// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/fused_chain.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <typeinfo>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mlx/backend/common/slicing.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::omarchy {

bool fused_chain_enabled() {
  return std::getenv("MLX_OMARCHY_FUSED_CHAIN") == nullptr ||
      env_flag("MLX_OMARCHY_FUSED_CHAIN");
}
namespace {

// Op codes: lockstep with the switch in shaders/fused_chain.comp. This
// is the chain's own packed-program space, mapped from primitives by
// chain_op_for below; the values are NOT the ElementwiseOperation enum
// from ../primitives.cpp (they diverge from 11 up).
enum ChainOperation : uint32_t {
  ChainAdd = 0,
  ChainMultiply = 1,
  ChainDivide = 2,
  ChainMaximum = 3,
  ChainExp = 4,
  ChainSigmoid = 5,
  ChainSquare = 6,
  ChainSqrt = 7,
  ChainRsqrt = 8,
  ChainSubtract = 9,
  ChainNegative = 10,
  ChainMinimum = 11,
  ChainTanh = 12,
};

// Instruction packing: op[7:0] | a[15:8] | b[23:16] | dst[31:24].
constexpr uint8_t kChainLeafBase = 0x10;
constexpr uint32_t kMaxChainLeaves = 3;
constexpr uint32_t kMaxChainInstrs = 8;

constexpr uint32_t pack_instruction(
    uint32_t op,
    uint32_t a,
    uint32_t b,
    uint32_t dst) {
  return (op & 0xffu) | ((a & 0xffu) << 8) | ((b & 0xffu) << 16) |
      ((dst & 0xffu) << 24);
}

std::optional<uint32_t> chain_op_for(const Primitive& p) {
  const auto& t = typeid(p);
  if (t == typeid(Add)) {
    return ChainAdd;
  }
  if (t == typeid(Multiply)) {
    return ChainMultiply;
  }
  if (t == typeid(Divide)) {
    return ChainDivide;
  }
  if (t == typeid(Maximum)) {
    return ChainMaximum;
  }
  if (t == typeid(Exp)) {
    return ChainExp;
  }
  if (t == typeid(Sigmoid)) {
    return ChainSigmoid;
  }
  if (t == typeid(Square)) {
    return ChainSquare;
  }
  if (t == typeid(Sqrt)) {
    return ChainSqrt;
  }
  if (t == typeid(Subtract)) {
    return ChainSubtract;
  }
  if (t == typeid(Negative)) {
    return ChainNegative;
  }
  if (t == typeid(Minimum)) {
    return ChainMinimum;
  }
  if (t == typeid(Tanh)) {
    return ChainTanh;
  }
  return std::nullopt;
}

ComputeBinding chain_binding(const array& value) {
  auto* buffer = static_cast<const VulkanBuffer*>(value.buffer().ptr());
  return {buffer->buffer, 0, buffer->size, buffer};
}

// Leaf broadcast addressing modes; lockstep with fused_chain.comp.
constexpr uint32_t kLeafDirect = 0;
constexpr uint32_t kLeafModLast = 1;
constexpr uint32_t kLeafDivLast = 2;
constexpr uint32_t kLeafScalar = 3;

std::optional<uint32_t> leaf_mode_for(
    const array& leaf,
    uint32_t count,
    uint32_t last_dim) {
  const size_t data_size = leaf.data_size();
  if (data_size == count) {
    return kLeafDirect;
  }
  if (data_size == 1) {
    return kLeafScalar;
  }
  if (last_dim == 0) {
    return std::nullopt;
  }
  const auto& shape = leaf.shape();
  // ModLast (address = index % last_dim) is correct exactly when every
  // output element (r, c) reads flat[c]: the leaf must be a single
  // physical row of last_dim elements (data_size == last_dim) whose
  // outer axes are pure broadcasts. Strides, not flags, prove that:
  // the last axis needs unit stride, and every outer axis needs stride
  // 0 (broadcast view) or a singleton shape (stride irrelevant). A
  // column broadcast - (64, 1) leaf expanded to (64, 64), strides
  // (1, 0) - has the same shape, data_size, and contiguous flag as the
  // row broadcast (0, 1) but its element (r, c) reads flat[r], which
  // index % last_dim cannot express; admitting it misindexed every
  // product (hardware-confirmed: fused sum 1.32 vs per-node -9.23).
  // The 2026-09-18 shape-only guard refused both broadcasts and cost
  // the row case its fusion (v0.7.0 recert fc pin anomaly); the
  // stride check refuses exactly the unsafe one.
  if (data_size == last_dim) {
    const auto& strides = leaf.strides();
    if (strides.empty() || strides.back() != 1) {
      return std::nullopt;
    }
    for (size_t i = 0; i + 1 < strides.size(); ++i) {
      if (strides[i] != 0 && shape[i] != 1) {
        return std::nullopt;
      }
    }
    return kLeafModLast;
  }
  // DivLast (address = index / last_dim) repeats each leaf element
  // last_dim consecutive outputs, which matches only a column-vector
  // leaf broadcast along its last axis (shape (..., count/last_dim, 1)).
  // A wide last axis such as (N, 1, L) needs the combined mapping
  // (index / (count/data_size)) * L + index % L, which the packed leaf
  // ABI cannot express; selecting DivLast from the data_size alone read
  // k[N, dv] for every dk and corrupted the GDN state update (F7: NaN
  // logits from step 4 on Bonsai-2-27B, reproduced standalone).
  if (count % last_dim == 0 && data_size == count / last_dim) {
    if (shape.back() == 1) {
      return kLeafDivLast;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

} // namespace

struct FusedChainImpl {
  // Stable graph IDs replace references to scheduler tape entries: eager
  // entries are detached immediately after gpu::eval returns.
  std::vector<std::uintptr_t> node_ids;
  std::vector<array> node_arrays;
  std::vector<uint32_t> program;
  std::vector<array> leaves;
  std::vector<uint32_t> leaf_offsets;
  std::vector<uint32_t> leaf_modes;
  uint32_t count = 0;
  uint32_t last_dim = 0;
  Shape eval_shape;
  Dtype dtype = float32;
  std::optional<array> tail_array;
  bool gate_enabled = false;
  bool open = false;
  bool saw_tape_output = false;
};

FusedChain::FusedChain(bool gate_enabled)
    : impl_(std::make_unique<FusedChainImpl>()) {
  // The gate is read ONCE per tape evaluation by the owner
  // (eval_compiled_tape) and handed in, so the off path costs no env
  // lookups at all.
  impl_->gate_enabled = gate_enabled;
}
FusedChain::~FusedChain() = default;
FusedChain::FusedChain(FusedChain&& other)
    : impl_(std::move(other.impl_)) {}

FusedChain& FusedChain::operator=(FusedChain&& other) {
  if (this != &other) {
    // Hand the slot over without instantiating unique_ptr's deleter on
    // the incomplete impl type in caller translation units.
    auto* handed_over = other.impl_.release();
    impl_.reset(handed_over);
  }
  return *this;
}

bool FusedChain::can_start(const array& node) {
  // Pure op/dtype check; the DEFAULT-OFF gate lives in the
  // constructor argument (MLX_OMARCHY_FUSED_CHAIN, decided by the tape
  // interpreter).
  if (!node.has_primitive()) {
    return false;
  }
  if (node.dtype() != float32 && node.dtype() != float16 &&
      node.dtype() != bfloat16) {
    return false;
  }
  return chain_op_for(node.primitive()).has_value();
}

size_t FusedChain::size() const {
  return impl_->node_ids.size();
}

std::uintptr_t FusedChain::tail_id() const {
  return impl_->node_ids.back();
}

bool FusedChain::carries(std::uintptr_t id) const {
  for (auto node_id : impl_->node_ids) {
    if (node_id == id) {
      return true;
    }
  }
  return false;
}

bool FusedChain::try_add(
    const array& node,
    const std::vector<array>& node_inputs,
    bool is_tape_output) {
  if (impl_->saw_tape_output) {
    return false;
  }
  if (impl_->node_ids.size() >= kMaxChainInstrs) {
    return false;
  }
  // Cached gate: read once per tape evaluation (constructor), not once
  // per node on the decode hot path.
  if (!impl_->gate_enabled) {
    return false;
  }
  if (!can_start(node)) {
    return false;
  }
  const auto op = chain_op_for(node.primitive()).value();
  // Eval-time broadcast shape of this node's output, derived from the
  // resolved inputs exactly as the per-node interpreter derives its
  // output shape. A shapeless tape serves decode from a prefill trace,
  // so node.shape() can be a stale trace shape; chain uniformity and
  // the dispatch count must key on what actually evaluates. The tail
  // contributes the chain's eval shape, not its tracing shape.
  auto is_prev = [&](const array& in) {
    return !impl_->node_ids.empty() && in.id() == impl_->node_ids.back();
  };
  Shape eval_shape;
  {
    int nd = 0;
    for (const auto& in : node_inputs) {
      nd = std::max(nd, static_cast<int>(
                            is_prev(in) ? impl_->eval_shape.size()
                                        : in.ndim()));
    }
    eval_shape.resize(nd, 0);
    for (const auto& in : node_inputs) {
      const bool prev = is_prev(in);
      const Shape& shape = prev ? impl_->eval_shape : in.shape();
      auto dd = nd - static_cast<int>(shape.size());
      for (int i = dd; i < nd; ++i) {
        eval_shape[i] = std::max(eval_shape[i], shape[i - dd]);
      }
    }
  }
  if (impl_->open) {
    if (eval_shape != impl_->eval_shape || node.dtype() != impl_->dtype) {
      return false;
    }
  }

  uint32_t count = 0;
  uint32_t last_dim = 0;
  if (impl_->open) {
    count = impl_->count;
    last_dim = impl_->last_dim;
  } else {
    if (eval_shape.empty()) {
      // A scalar chain has nothing to fuse.
      return false;
    }
    int64_t n = 1;
    for (auto d : eval_shape) {
      n *= d;
    }
    if (n > std::numeric_limits<uint32_t>::max() || n == 0) {
      return false;
    }
    count = static_cast<uint32_t>(n);
    last_dim = static_cast<uint32_t>(eval_shape.back());
  }

  if (!impl_->open && node.dtype() != float32) {
    const auto& capabilities = device().capabilities();
    if ((node.dtype() == float16 && !capabilities.shader_float16) ||
        (node.dtype() == bfloat16 && !capabilities.shader_int16) ||
        !capabilities.storage_buffer_16bit_access) {
      return false;
    }
  }

  auto encode_leaf = [&](const array& in) -> std::optional<uint32_t> {
    const auto mode = leaf_mode_for(in, count, last_dim);
    if (!mode) {
      return std::nullopt;
    }
    // Direct leaves are read as linear row-major buffers, so storage
    // order must equal logical order: a column-contiguous (e.g.
    // transposed) leaf is `contiguous` but would fuse in storage order
    // and return permuted values (upstream test_compile_dynamic_dims).
    // Refuse those and let the per-node path preserve the layout.
    // Tiled (mod-last/div-last) leaves read a smaller shared buffer a
    // broadcast produced in logical order, so any contiguous source
    // works. Scalar leaves read only index 0 and are layout-independent.
    if (mode.value() == kLeafDirect) {
      if (!in.flags().row_contiguous) {
        return std::nullopt;
      }
    } else if (mode.value() != kLeafScalar && !in.flags().contiguous) {
      return std::nullopt;
    }
    const size_t item_offset = in.offset() / in.itemsize();
    if (item_offset > std::numeric_limits<uint32_t>::max() / 2) {
      return std::nullopt;
    }
    for (size_t i = 0; i < impl_->leaves.size(); ++i) {
      if (impl_->leaves[i].id() == in.id() &&
          impl_->leaf_offsets[i] == item_offset &&
          impl_->leaf_modes[i] == mode.value()) {
        return kChainLeafBase + static_cast<uint32_t>(i);
      }
    }
    if (impl_->leaves.size() >= kMaxChainLeaves) {
      return std::nullopt;
    }
    uint32_t span;
    switch (mode.value()) {
      case kLeafDirect:
        span = count;
        break;
      case kLeafModLast:
        span = last_dim;
        break;
      case kLeafDivLast:
        span = count / last_dim;
        break;
      default:
        span = 1;
        break;
    }
    if (!compute_index_span_fits(item_offset, span)) {
      return std::nullopt;
    }
    auto* leaf_vk = static_cast<const VulkanBuffer*>(in.buffer().ptr());
    const uint64_t byte_end =
        item_offset * in.itemsize() + span * in.itemsize();
    if (byte_end > leaf_vk->size) {
      return std::nullopt;
    }
    impl_->leaves.push_back(in);
    impl_->leaf_offsets.push_back(static_cast<uint32_t>(item_offset));
    impl_->leaf_modes.push_back(mode.value());
    return kChainLeafBase + static_cast<uint32_t>(impl_->leaves.size() - 1);
  };

  // Operand resolution: one operand may be the previous member's output
  // (its register); everything else must be an addressable leaf.
  const uint32_t dst = static_cast<uint32_t>(impl_->node_ids.size());
  const uint32_t prev_reg = dst > 0 ? dst - 1 : std::numeric_limits<uint32_t>::max();
  std::optional<uint32_t> a;
  std::optional<uint32_t> b;
  if (impl_->open && !is_prev(node_inputs[0]) &&
      (node_inputs.size() < 2 || !is_prev(node_inputs[1]))) {
    // EXTENSIONS must consume the tail's register. A same-shape node
    // sharing no data with the tail would ride the chain as a NON-tail
    // interior member; closing the chain materializes the tail only,
    // so a consumer of that member outside the chain would resolve
    // nothing. Refusing here makes the interpreter close the chain
    // first; the sibling then opens a fresh chain of its own.
    return false;
  }
  // Leaves pushed for a REJECTED member are rolled back below, so the
  // chain never carries orphan slots.
  const size_t leaf_base = impl_->leaves.size();
  if (node_inputs.size() == 1) {
    a = is_prev(node_inputs[0]) ? prev_reg : encode_leaf(node_inputs[0]);
    b = a;
  } else if (node_inputs.size() == 2) {
    if (is_prev(node_inputs[0])) {
      a = prev_reg;
      b = encode_leaf(node_inputs[1]);
    } else if (is_prev(node_inputs[1])) {
      b = prev_reg;
      a = encode_leaf(node_inputs[0]);
    } else {
      // Chain head only (extensions were refused above): both
      // operands are addressable leaves.
      a = encode_leaf(node_inputs[0]);
      b = encode_leaf(node_inputs[1]);
    }
  } else {
    return false;
  }
  if (!a || !b) {
    // Erase, not resize: array is not default-constructible.
    impl_->leaves.erase(
        impl_->leaves.begin() + static_cast<std::ptrdiff_t>(leaf_base),
        impl_->leaves.end());
    impl_->leaf_offsets.resize(leaf_base);
    impl_->leaf_modes.resize(leaf_base);
    return false;
  }


  impl_->program.push_back(pack_instruction(op, a.value(), b.value(), dst));
  impl_->node_ids.push_back(node.id());
  impl_->node_arrays.push_back(node);
  impl_->dtype = node.dtype();
  impl_->tail_array = node;
  if (!impl_->open) {
    impl_->count = count;
    impl_->last_dim = last_dim;
    impl_->eval_shape = eval_shape;
    impl_->open = true;
  }
  if (is_tape_output) {
    impl_->saw_tape_output = true;
  }
  return true;
}

bool FusedChain::tail_is_tape_output() const {
  return impl_->saw_tape_output;
}

namespace {

// The SwiGLU program (r0 = sigmoid(g); r1 = g * r0; out = r1 * u, two
// direct leaves) gets the straight-line four-wide shaders/swiglu.comp
// with the interpreter's exact rounding, materialized intermediates
// included. Returns the (gate, up) leaf slots when the chain has that
// shape and the alignment the kernel needs, else nullopt and the
// interpreter runs.
std::optional<std::pair<uint32_t, uint32_t>> swiglu_leaves(
    const FusedChainImpl& chain) {
  if (chain.program.size() != 3 || chain.leaves.size() != 2 ||
      chain.node_ids.size() != 3 || (chain.count & 3u) != 0u ||
      chain.dtype == float32) {
    return std::nullopt;
  }
  for (uint32_t mode : chain.leaf_modes) {
    if (mode != kLeafDirect) {
      return std::nullopt;
    }
  }
  auto field = [&](size_t i, int shift) {
    return (chain.program[i] >> shift) & 0xffu;
  };
  uint32_t op0 = field(0, 0), a0 = field(0, 8), d0 = field(0, 24);
  uint32_t op1 = field(1, 0), a1 = field(1, 8), b1 = field(1, 16),
           d1 = field(1, 24);
  uint32_t op2 = field(2, 0), a2 = field(2, 8), b2 = field(2, 16),
           d2 = field(2, 24);
  if (op0 != ChainSigmoid || op1 != ChainMultiply || op2 != ChainMultiply ||
      a0 < kChainLeafBase || d2 != chain.node_ids.size() - 1) {
    return std::nullopt;
  }
  uint32_t gate = a0 - kChainLeafBase;
  // Multiplication commutes exactly, so either operand order matches.
  bool mul1_ok = (a1 == a0 && b1 == d0) || (a1 == d0 && b1 == a0);
  uint32_t up = a2 == d1 ? b2 : (b2 == d1 ? a2 : 0u);
  if (!mul1_ok || up < kChainLeafBase || up - kChainLeafBase == gate) {
    return std::nullopt;
  }
  up -= kChainLeafBase;
  if (gate >= chain.leaves.size() || up >= chain.leaves.size() ||
      (chain.leaf_offsets[gate] & 3u) != 0u ||
      (chain.leaf_offsets[up] & 3u) != 0u) {
    return std::nullopt;
  }
  return std::make_pair(gate, up);
}

// The standalone silu program (r0 = sigmoid(x); out = x * r0, one direct
// leaf). The generic interpreter runs it at ~8% of the memory roofline.
bool silu_leaf(const FusedChainImpl& chain) {
  if (chain.program.size() != 2 || chain.leaves.size() != 1 ||
      chain.node_ids.size() != 2 || (chain.count & 3u) != 0u ||
      chain.dtype == float32 || chain.leaf_modes[0] != kLeafDirect ||
      (chain.leaf_offsets[0] & 3u) != 0u) {
    return false;
  }
  auto field = [&](size_t i, int shift) {
    return (chain.program[i] >> shift) & 0xffu;
  };
  uint32_t op0 = field(0, 0), a0 = field(0, 8), d0 = field(0, 24);
  uint32_t op1 = field(1, 0), a1 = field(1, 8), b1 = field(1, 16),
           d1 = field(1, 24);
  return op0 == ChainSigmoid && op1 == ChainMultiply && a0 >= kChainLeafBase &&
      ((a1 == a0 && b1 == d0) || (a1 == d0 && b1 == a0)) &&
      d1 == chain.node_ids.size() - 1;
}

void dispatch_chain(
    FusedChainImpl& chain,
    array& out,
    const Stream& stream,
    bool materialize_intermediates) {
  auto& encoder = get_command_encoder(stream);
  out.set_data(allocator().malloc(out.nbytes()));
  if (materialize_intermediates) {
    for (size_t i = 0; i + 1 < chain.node_arrays.size(); ++i) {
      chain.node_arrays[i].set_data(
          allocator().malloc(chain.node_arrays[i].nbytes()));
    }
  }
  const bool materialize_nodes =
      materialize_intermediates && chain.node_arrays.size() == 3;
  if (auto leaves = swiglu_leaves(chain);
      leaves && materialize_nodes == materialize_intermediates) {
    ComputeParams params;
    params.count = chain.count;
    params.operation = chain.leaf_offsets[leaves->first];
    params.lhs_size = chain.leaf_offsets[leaves->second];
    params.rhs_size = materialize_nodes ? 1u : 0u;
    if (out.dtype() == float16 && fused_trio_enabled()) {
      // The trio pipeline keeps the same near-copied swiglu arithmetic
      // on the shared decode pipeline; out rides binding 4 and the
      // materialized intermediates bindings 5/6 (the fast_trio.comp
      // slot map). The ComputeParams swiglu mapping is unchanged.
      std::array<ComputeBinding, 7> trio_bindings{
          chain_binding(chain.leaves[leaves->first]),
          chain_binding(chain.leaves[leaves->second]),
          chain_binding(out),
          chain_binding(out),
          chain_binding(out),
          chain_binding(materialize_nodes ? chain.node_arrays[0] : out),
          chain_binding(materialize_nodes ? chain.node_arrays[1] : out)};
      encoder.dispatch_compute(
          ComputeKernel::FastTrioSwigluF16,
          trio_bindings,
          params,
          compute_dispatch_group_count(chain.count / 4u));
      return;
    }
    std::array<ComputeBinding, 5> bindings{
        chain_binding(chain.leaves[leaves->first]),
        chain_binding(chain.leaves[leaves->second]),
        chain_binding(out),
        chain_binding(materialize_nodes ? chain.node_arrays[0] : out),
        chain_binding(materialize_nodes ? chain.node_arrays[1] : out)};
    encoder.dispatch_compute(
        out.dtype() == float16 ? ComputeKernel::SwigluF16
                               : ComputeKernel::SwigluBF16,
        bindings,
        params,
        compute_dispatch_group_count(chain.count / 4u));
    return;
  }
  // Standalone silu (r0 = sigmoid(x); out = x * r0; one direct leaf): the
  // straight-line swiglu kernel built with up == 1.0, so the stored bits are
  // the interpreter's (x * r0 rounded, then * 1.0 exactly).
  if (!materialize_intermediates && silu_leaf(chain)) {
    ComputeParams params;
    params.count = chain.count;
    params.operation = chain.leaf_offsets[0];
    params.lhs_size = chain.leaf_offsets[0];
    params.rhs_size = 0u;
    std::array<ComputeBinding, 5> bindings{
        chain_binding(chain.leaves[0]),
        chain_binding(chain.leaves[0]),
        chain_binding(out),
        chain_binding(out),
        chain_binding(out)};
    encoder.dispatch_compute(
        out.dtype() == float16 ? ComputeKernel::SiluF16
                               : ComputeKernel::SiluBF16,
        bindings,
        params,
        compute_dispatch_group_count(chain.count / 4u));
    return;
  }

  const size_t program_bytes = chain.program.size() * sizeof(uint32_t);
  Buffer program_buffer = allocator().malloc(program_bytes);
  auto* program_vk = static_cast<VulkanBuffer*>(program_buffer.ptr());
  std::memcpy(program_vk->data, chain.program.data(), program_bytes);
  array program_keeper(
      Shape{static_cast<int>(chain.program.size())}, uint32, nullptr, {});
  array::Flags keeper_flags;
  keeper_flags.contiguous = true;
  keeper_flags.row_contiguous = true;
  keeper_flags.col_contiguous = true;
  program_keeper.set_data(
      program_buffer,
      program_keeper.size(),
      Strides{1},
      keeper_flags,
      0);
  encoder.add_temporary(program_keeper);

  ComputeParams params;
  params.count = chain.count;
  params.operation = static_cast<uint32_t>(chain.program.size());
  params.lhs_size = chain.last_dim;
  params.rhs_size = static_cast<uint32_t>(chain.node_ids.size() - 1);
  params.reduce_size = chain.leaf_offsets.size() > 0 ? chain.leaf_offsets[0] : 0;
  params.output_size = chain.leaf_offsets.size() > 1 ? chain.leaf_offsets[1] : 0;
  params.lhs_offset = chain.leaf_offsets.size() > 2 ? chain.leaf_offsets[2] : 0;
  params.rhs_offset = chain.leaf_modes.size() > 0 ? chain.leaf_modes[0] : 0;
  params.output_offset = chain.leaf_modes.size() > 1 ? chain.leaf_modes[1] : 0;
  params.aux_size = chain.leaf_modes.size() > 2 ? chain.leaf_modes[2] : 0;
  params.aux_offset = materialize_intermediates ? 1u : 0u;

  std::array<ComputeBinding, kMaxChainLeaves + 3> bindings{
      chain_binding(out),
      chain_binding(out),
      chain_binding(out),
      chain_binding(program_keeper),
      chain_binding(out),
      chain_binding(out)};
  for (size_t i = 0; i < chain.leaves.size(); ++i) {
    bindings[i] = chain_binding(chain.leaves[i]);
  }
  if (materialize_intermediates && !chain.node_arrays.empty()) {
    bindings[2] = chain_binding(chain.node_arrays[0]);
    if (chain.node_arrays.size() > 2) {
      bindings[5] = chain_binding(chain.node_arrays[1]);
    }
  }

  auto kernel = ComputeKernel::FusedChainF32;
  if (out.dtype() == float16) {
    kernel = ComputeKernel::FusedChainF16;
  } else if (out.dtype() == bfloat16) {
    kernel = ComputeKernel::FusedChainBF16;
  }
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      compute_dispatch_group_count(chain.count));
}

} // namespace

std::optional<array> FusedChain::evaluate(const Stream& stream) {
  if (impl_->node_ids.empty()) {
    return std::nullopt;
  }
  // A graph-free value, like an eager node after the evaluator detaches
  // it. The tail's operands include the previous member's tracing-graph
  // array (the chain continuation), so an output that kept them as
  // inputs led any nested eval over a consumer's inputs - the Sin/Cos
  // argument gate's settle() - into the trace and its placeholder
  // inputs: "[eval] Attempting to eval an array without a primitive".
  array out(impl_->eval_shape, impl_->dtype, nullptr, {});
  dispatch_chain(*impl_, out, stream, false);
  out.set_status(array::Status::evaluated);
  return out;
}

void FusedChain::evaluate_tail(const Stream& stream) {
  if (impl_->node_ids.empty() || !impl_->tail_array) {
    return;
  }
  dispatch_chain(*impl_, *impl_->tail_array, stream, true);
}

namespace {

enum class EagerStep : uint8_t { skip, sigmoid, gate_mul, output_mul };

struct EagerRole {
  std::uintptr_t group;
  EagerStep step;
};

// A planned decode RoPE pair (query, key): pending until the first
// member evaluates, then done (one FastTrioRopePairF16 dispatch wrote both
// outputs) or failed (both nodes take the ordinary RoPE path).
struct RopePair {
  std::array<array, 2> nodes;
  enum class State : uint8_t { pending, done, failed } state{State::pending};
};

// A planned decode GEMV group: pending until its first member
// evaluates, then done (every member and epilogue is written by the
// one dispatch) or failed (every node takes its own path).
struct GemvGroup {
  std::vector<GemvFusionMember> members;
  // SwiGLU store epilogue fold: set at plan time when the group is
  // exactly a chain's gate and up projections. The one dispatch writes
  // silu(gate) * up into this array and no swiglu dispatch exists.
  std::optional<array> swiglu_out;
  // Out-gate prologue fold: set at plan time when the group's shared x
  // is the deleted Multiply(Sigmoid(gate), out) output. The one
  // dispatch reads gate and out directly and rebuilds x per workgroup.
  std::optional<OutgatePlan> outgate;
  enum class State : uint8_t { pending, done, failed } state{State::pending};
};

struct DenseGemvGroup {
  std::vector<array> nodes;
  std::optional<array> input;
  // Planned producer-direct write of a member's output row into the
  // values cache copy (indexed like |nodes|; empty keeps the member's
  // own buffer).
  std::vector<std::optional<KvDirectWindow>> sum_windows;
  enum class State : uint8_t { pending, done, failed } state{State::pending};
};

struct SliceUpdatePair {
  explicit SliceUpdatePair(const array& first, const array& second)
      : nodes{first, second} {}

  std::array<array, 2> nodes;
  enum class State : uint8_t { pending, deferred, done, failed } state{
      State::pending};
  // Producer-direct plan: windows[0] is the RoPE (keys) target,
  // windows[1] the GEMV Add-epilogue (values) target. When both
  // producers commit their stores the merged pair dispatch is skipped;
  // any abort unwinds to that dispatch unchanged.
  bool direct{false};
  std::array<std::optional<KvDirectWindow>, 2> windows;
  // Set when the GEMV group stored the values sum into windows[1]:
  // windows[1].node holds the rows, its sum buffer never materializes,
  // and the keys side must now commit too (it runs later by data
  // dependency).
  bool values_committed{false};
};

struct EagerFusionState {
  std::unordered_map<std::uintptr_t, EagerRole> roles;
  std::unordered_map<std::uintptr_t, FusedChain> chains;
  std::unordered_map<std::uintptr_t, size_t> gemv_roles;
  std::vector<GemvGroup> gemv_groups;
  std::unordered_map<std::uintptr_t, size_t> dense_gemv_roles;
  std::vector<DenseGemvGroup> dense_gemv_groups;
  std::unordered_map<std::uintptr_t, size_t> slice_update_roles;
  std::vector<SliceUpdatePair> slice_update_pairs;
  std::unordered_map<std::uintptr_t, size_t> rope_redirect_roles;
  std::unordered_map<std::uintptr_t, size_t> reshape_redirect_roles;
  std::unordered_map<std::uintptr_t, size_t> rope_pair_roles;
  std::vector<RopePair> rope_pairs;
};

thread_local EagerFusionState* eager_state = nullptr;

bool is_op(const array* node, const std::type_info& op) {
  return node && node->has_primitive() && typeid(node->primitive()) == op;
}
// Two RoPE offset inputs can share one buffer binding when they are the
// same node, or when both are host-constant int scalars with an equal
// value (the mlx_lm decode shape: cache.offset is a Python int, so each
// rope call materializes its own scalar array). Host-constant means
// status available and no primitive - nothing in flight can be writing
// it, the same test rope_trig_gate uses before reading an offset.
bool rope_offset_shared(const array& a, const array& b) {
  if (a.id() == b.id()) {
    return true;
  }
  return a.size() == 1 && b.size() == 1 && a.dtype() == int32 &&
      b.dtype() == int32 && a.status() == array::Status::available &&
      !a.has_primitive() && b.status() == array::Status::available &&
      !b.has_primitive() && a.item<int>() == b.item<int>();
}
// Joins the captured rope neighbor names for the plan trace line.
std::string joined_rope_neighbors(
    const std::vector<std::string>& names) {
  std::string joined;
  for (const auto& name : names) {
    joined += name;
    joined += ",";
  }
  return joined;
}
// Plan-time trace for the RoPE pair scan (MLX_OMARCHY_TRIO_TRACE=1):
// one line per decode-shaped plan, capped, so a no-claim is diagnosable.
struct RopePairScanTrace {
  size_t rope_nodes = 0;
  size_t rope_forward_scalar = 0;
  size_t rej_not_rope = 0;
  size_t rej_node_inputs = 0;
  size_t rej_node_offset = 0;
  size_t rej_node_dtype = 0;
  size_t rej_node_notforward = 0;
  size_t rej_state = 0;
  std::vector<std::string> rope_neighbor_after;
  std::vector<std::string> rope_shapes;
  size_t rej_offset = 0;
  size_t rej_shape = 0;
  size_t rej_claimed = 0;
  size_t rej_redirect = 0;
  size_t pairs = 0;
};

const array& dense_gemv_source(const array& x) {
  if (is_op(&x, typeid(Flatten)) && x.inputs().size() == 1 &&
      x.dtype() == x.inputs()[0].dtype() && x.size() == x.inputs()[0].size()) {
    return x.inputs()[0];
  }
  return x;
}

// Producer-direct KV write planning. Classification of one SliceUpdate
// pair member's update producer: the keys member's producer is a RoPE
// node, the values member's producer is a fused GEMV Add epilogue read
// through a provable chain of view-only ops (Transpose/Reshape).
enum class DirectKind : uint8_t { none, keys_rope, values_sum, dense_values };

struct DirectPlan {
  std::optional<KvDirectWindow> window;
  DirectKind kind{DirectKind::none};
  size_t group_index{0};
  size_t member_index{0};
  // View-only ops (Transpose/Reshape) between the pair's update input
  // and the Add sum, update first. All become window aliases.
  std::vector<array> view_chain;
};

// Element geometry of a paste window inside a SliceUpdate output:
// element offset and per-update-axis strides, plus the span the window
// covers. Arrays are attached later by aggregate init (KvDirectWindow
// holds arrays and is not default-constructible).
struct DirectGeometry {
  uint32_t offset{0};
  uint32_t strides[4]{};
  uint32_t ndim{0};
};

// Shared layout contract of a direct-write window: |member| is a full
// row-contiguous f16 or bf16 copy of the cache, and the paste window
// (from the member's own SliceUpdate geometry) fits inside it.
std::optional<DirectGeometry> direct_window_geometry(const array& member) {
  const auto& base = member.inputs()[0];
  const auto& upd = member.inputs()[1];
  if (
      (member.dtype() != float16 && member.dtype() != bfloat16) ||
      member.dtype() != base.dtype() || member.dtype() != upd.dtype() ||
      !base.flags().row_contiguous ||
      base.size() != base.data_size() ||
      base.offset() % base.itemsize() != 0 || upd.size() == 0 ||
      upd.ndim() == 0 || upd.ndim() > 4) {
    return std::nullopt;
  }
  const auto& primitive_state =
      static_cast<const SliceUpdate&>(member.primitive()).state();
  auto [offset, strides] = prepare_slice(
      member,
      std::get<1>(primitive_state),
      std::get<3>(primitive_state));
  uint64_t span = 0;
  DirectGeometry geometry;
  geometry.ndim = static_cast<uint32_t>(upd.ndim());
  for (int axis = 0; axis < upd.ndim(); ++axis) {
    if (upd.shape(axis) <= 0 || strides[axis] < 0 ||
        static_cast<uint64_t>(upd.shape(axis)) >
            std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(strides[axis]) >
            std::numeric_limits<uint32_t>::max() ||
        (axis == upd.ndim() - 1 && strides[axis] != 1)) {
      return std::nullopt;
    }
    span += static_cast<uint64_t>(upd.shape(axis) - 1) * strides[axis];
    geometry.strides[axis] = static_cast<uint32_t>(strides[axis]);
  }
  if (offset > std::numeric_limits<uint32_t>::max() ||
      span > std::numeric_limits<uint32_t>::max() - offset ||
      offset + span >= member.size()) {
    return std::nullopt;
  }
  geometry.offset = static_cast<uint32_t>(offset);
  return geometry;
}

// Keys: the RoPE kernel writes (matrix, time, feature) output strides,
// so the window's (batch, kv-head) axes must form one regular matrix
// axis. The RoPE fence (forward, scalar offset, fused-path conditions)
// is re-checked at dispatch; aborting there unwinds to the merged pair
// dispatch.
DirectPlan plan_keys_window(
    const array& member,
    const array* update,
    const std::unordered_map<std::uintptr_t, size_t>& uses,
    const std::unordered_map<std::uintptr_t, size_t>& view_uses) {
  DirectPlan plan;
  if (!is_op(update, typeid(fast::RoPE))) {
    return plan;
  }
  // rope_rms_norm (the fused q/k RMSNorm + RoPE) rides the same RoPE
  // primitive with the norm weight as a third input; the redirected
  // dispatch is norm-aware and gates inputs.size() == (with_norm ? 3 : 2)
  // itself, so both shapes plan identically.
  const bool with_norm =
      static_cast<const fast::RoPE&>(update->primitive()).has_norm();
  if (update->inputs().size() != (with_norm ? 3u : 2u)) {
    return plan;
  }
  auto geometry = direct_window_geometry(member);
  if (!geometry) {
    return plan;
  }
  const auto& upd = member.inputs()[1];
  // In place only when the cache buffer's single in-tape consumer is
  // this SliceUpdate: no other node may still need the pre-update rows.
  // In place when every consumer of the cache buffer is a pure view
  // (the returned keys[..., :offset] slice reads post-write bytes).
  auto use_it = uses.find(member.inputs()[0].id());
  auto view_it = view_uses.find(member.inputs()[0].id());
  bool in_place = use_it != uses.end() && use_it->second > 0 &&
      view_it != view_uses.end() && view_it->second == use_it->second;
  uint32_t matrix_stride;
  if (upd.ndim() == 4) {
    uint64_t heads = static_cast<uint64_t>(upd.shape(1));
    if (heads == 0 ||
        static_cast<uint64_t>(geometry->strides[0]) !=
            heads * static_cast<uint64_t>(geometry->strides[1])) {
      return plan;
    }
    matrix_stride = geometry->strides[1];
  } else {
    matrix_stride = geometry->strides[0];
  }
  plan.window = KvDirectWindow{
      member,
      member.inputs()[0],
      geometry->offset,
      {matrix_stride, geometry->strides[upd.ndim() - 2], 1, 0},
      geometry->ndim,
      /*row_gap=*/0,
      /*head_dim=*/0,
      in_place};
  plan.kind = DirectKind::keys_rope;
  return plan;
}

// Values: the epilogue sum is one flat row of n_kv * head_dim columns;
// its store maps column c to offset + (c / head_dim) * row_gap +
// (c % head_dim), which needs a single batch and a window whose inner
// axis is contiguous.
DirectPlan plan_values_window(
    const array& member,
    const array* update,
    const std::unordered_map<std::uintptr_t, size_t>& uses,
    const std::vector<GemvGroup>& groups,
    const std::unordered_map<std::uintptr_t, size_t>& dense_roles,
    std::vector<DenseGemvGroup>& dense_groups,
    const std::unordered_map<std::uintptr_t, size_t>& view_uses) {
  DirectPlan plan;
  auto use_count = [&](const array& value) {
    auto it = uses.find(value.id());
    return it == uses.end() ? size_t{0} : it->second;
  };
  // Walk the chain of view-only ops (Transpose/Reshape) from the
  // pair's update input down to the Add sum. Every view must have
  // exactly one consumer: the direct plan leaves no evaluated copy
  // behind that could serve a second one.
  std::vector<const array*> chain;
  const array* node = update;
  while (is_op(node, typeid(Reshape)) || is_op(node, typeid(Transpose))) {
    if (use_count(*node) != 1) {
      return plan;
    }
    chain.push_back(node);
    node = &node->inputs()[0];
  }
  if (dense_roles.count(node->id()) == 0 &&
      !is_op(node, typeid(QuantizedMatmul)) &&
      (!is_op(node, typeid(Add)) || use_count(*node) != 1)) {
    return plan;
  }
  const array* sum = node;
  bool found = false;
  // Dense-terminal: the sum is itself a dense bf16 decode GEMV group
  // member's output row (no Add epilogue). The group stores it through
  // the same producer-direct window contract.
  auto dense_it = dense_roles.find(sum->id());
  if (dense_it != dense_roles.end()) {
    if (use_count(*sum) != 1) {
      return plan;
    }
    const auto& group_nodes = dense_groups[dense_it->second].nodes;
    for (size_t mi = 0; mi < group_nodes.size(); ++mi) {
      if (group_nodes[mi].id() == sum->id()) {
        plan.group_index = dense_it->second;
        plan.member_index = mi;
        plan.kind = DirectKind::dense_values;
        found = true;
      }
    }
    if (!found) {
      return plan;
    }
  } else {
    for (size_t gi = 0; gi < groups.size() && !found; ++gi) {
      for (size_t mi = 0; mi < groups[gi].members.size(); ++mi) {
        const auto& candidate = groups[gi].members[mi].epilogue;
        if (candidate && candidate->id() == sum->id()) {
          plan.group_index = gi;
          plan.member_index = mi;
          plan.kind = DirectKind::values_sum;
          found = true;
        }
      }
    }
    // Raw-terminal: the sum is a fused GEMV group member's own output
    // row (v projections carry no Add epilogue). The dispatch stores
    // the rounded output into the window directly; the member's own
    // buffer stays a scratch write nothing reads.
    for (size_t gi = 0; gi < groups.size() && !found; ++gi) {
      for (size_t mi = 0; mi < groups[gi].members.size(); ++mi) {
        if (groups[gi].members[mi].node.id() == sum->id()) {
          if (use_count(*sum) != 1) {
            return plan;
          }
          plan.group_index = gi;
          plan.member_index = mi;
          plan.kind = DirectKind::values_sum;
          found = true;
        }
      }
    }
    if (!found) {
      return plan;
    }
  }
  const auto& base = member.inputs()[0];
  const auto& upd = member.inputs()[1];
  if (upd.ndim() != 4 || upd.shape(0) != 1 || base.shape(0) != 1 ||
      static_cast<size_t>(upd.shape(1)) * upd.shape(3) != sum->size()) {
    return plan;
  }
  auto geometry = direct_window_geometry(member);
  if (!geometry) {
    return plan;
  }
  uint32_t head_dim = static_cast<uint32_t>(upd.shape(3));
  if (head_dim == 0) {
    return plan;
  }
  // Prove the chain re-indexes the sum without moving bytes and in
  // the store order the GEMV writes: element (head, dim) of the
  // update must read the sum buffer at head * head_dim + dim.
  // Compose the map from the freshly written row-contiguous sum
  // forward through the chain. Reshape composes only from a
  // contiguous map (a reshape of anything else materializes);
  // Transpose always composes.
  auto contiguous_strides = [](const Shape& shape) {
    Strides strides(shape.size(), 0);
    strides.back() = 1;
    for (int ax = static_cast<int>(shape.size()) - 2; ax >= 0; --ax) {
      strides[ax] = strides[ax + 1] * shape[ax + 1];
    }
    return strides;
  };
  Shape map_shape(sum->shape());
  Strides map_strides = contiguous_strides(map_shape);
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    const array& view = **it;
    if (is_op(&view, typeid(Transpose))) {
      const auto& axes =
          static_cast<const Transpose&>(view.primitive()).state();
      if (axes.size() != map_shape.size()) {
        return plan;
      }
      Shape next_shape(map_shape.size());
      Strides next_strides(map_shape.size());
      for (size_t ax = 0; ax < axes.size(); ++ax) {
        if (axes[ax] < 0 ||
            static_cast<size_t>(axes[ax]) >= map_shape.size()) {
          return plan;
        }
        next_shape[ax] = map_shape[axes[ax]];
        next_strides[ax] = map_strides[axes[ax]];
      }
      map_shape = std::move(next_shape);
      map_strides = std::move(next_strides);
    } else {
      if (map_strides != contiguous_strides(map_shape)) {
        return plan;
      }
      map_shape = view.shape();
      map_strides = contiguous_strides(map_shape);
    }
  }
  if (map_shape != upd.shape() || map_strides[3] != 1 ||
      map_strides[1] != static_cast<ptrdiff_t>(head_dim)) {
    return plan;
  }
  for (const array* view : chain) {
    plan.view_chain.push_back(*view);
  }
  plan.window = KvDirectWindow{
      member,
      base,
      geometry->offset,
      {geometry->strides[0],
       geometry->strides[1],
       geometry->strides[2],
       geometry->strides[3]},
      geometry->ndim,
      /*row_gap=*/geometry->strides[1],
      /*head_dim=*/head_dim,
      /*in_place=*/[&] {
        auto use_it = uses.find(base.id());
        auto view_it = view_uses.find(base.id());
        return use_it != uses.end() && use_it->second > 0 &&
            view_it != view_uses.end() &&
            view_it->second == use_it->second;
      }()};
  if (plan.kind == DirectKind::none) {
    plan.kind = DirectKind::values_sum;
  }
  return plan;
}

} // namespace

EagerFusionScope::EagerFusionScope(const std::deque<array>& tape)
    : previous_(eager_state) {
  eager_state = nullptr;
  if (!fused_chain_enabled()) {
    return;
  }

  auto* state = new EagerFusionState;
  eager_state = state;
  std::unordered_map<std::uintptr_t, const array*> nodes;
  std::unordered_map<std::uintptr_t, size_t> uses;
  // Per-input count of consumers that are pure views (Slice, Transpose,
  // Reshape). Views materialize nothing and read their input whenever
  // the consuming graph reads them, so an array whose consumers are ALL
  // views is safely writable in place by a later producer (the views
  // observe the post-write bytes).
  std::unordered_map<std::uintptr_t, size_t> view_uses;
  nodes.reserve(tape.size());
  for (const auto& node : tape) {
    nodes.emplace(node.id(), &node);
    bool node_is_view = is_op(&node, typeid(Slice)) ||
        is_op(&node, typeid(Transpose)) || is_op(&node, typeid(Reshape));
    for (const auto& input : node.inputs()) {
      ++uses[input.id()];
      if (node_is_view) {
        ++view_uses[input.id()];
      }
    }
  }
  auto lookup = [&](const array& ref) -> const array* {
    auto it = nodes.find(ref.id());
    return it == nodes.end() ? nullptr : it->second;
  };
  // Swiglu chains planned for the GEMV store epilogue fold: the gate
  // and up leaves plus the tail array the fold will store into.
  struct SwigluPlan {
    std::uintptr_t gate_id;
    std::uintptr_t up_id;
    std::uintptr_t sigmoid_id;
    std::uintptr_t inner_id;
    array out;
  };
  std::vector<SwigluPlan> swiglu_plans;
  std::unordered_set<std::uintptr_t> claimed;
  for (const auto& tail : tape) {
    if (!is_op(&tail, typeid(Multiply)) || tail.inputs().size() != 2 ||
        claimed.count(tail.id())) {
      continue;
    }
    const array* left = lookup(tail.inputs()[0]);
    const array* right = lookup(tail.inputs()[1]);
    const array* inner = is_op(left, typeid(Multiply)) ? left :
        (is_op(right, typeid(Multiply)) ? right : nullptr);
    if (!inner || inner->inputs().size() != 2 || uses[inner->id()] != 1) {
      continue;
    }
    const array* inner_left = lookup(inner->inputs()[0]);
    const array* inner_right = lookup(inner->inputs()[1]);
    const array* sigmoid = is_op(inner_left, typeid(Sigmoid)) ? inner_left :
        (is_op(inner_right, typeid(Sigmoid)) ? inner_right : nullptr);
    if (!sigmoid || sigmoid->inputs().size() != 1 ||
        uses[sigmoid->id()] != 1) {
      continue;
    }
    const array* up = inner == left ? right : left;
    const array& gate = sigmoid == inner_left ? inner->inputs()[1]
                                               : inner->inputs()[0];
    if (gate.id() != sigmoid->inputs()[0].id() ||
        tail.dtype() != inner->dtype() || tail.dtype() != sigmoid->dtype() ||
        tail.primitive().stream() != inner->primitive().stream() ||
        tail.primitive().stream() != sigmoid->primitive().stream() ||
        claimed.count(inner->id()) || claimed.count(sigmoid->id())) {
      continue;
    }
    if (up != nullptr && fused_gemv_swiglu_enabled() &&
        tail.dtype() != float32) {
      swiglu_plans.push_back(SwigluPlan{
          gate.id(), up->id(), sigmoid->id(), inner->id(), tail});
    }
    const auto group = tail.id();
    state->roles.emplace(
        sigmoid->id(), EagerRole{group, EagerStep::sigmoid});
    state->roles.emplace(
        inner->id(), EagerRole{group, EagerStep::gate_mul});
    state->roles.emplace(
        tail.id(), EagerRole{group, EagerStep::output_mul});
    claimed.insert(sigmoid->id());
    claimed.insert(inner->id());
    claimed.insert(tail.id());
  }
  // Out-gate prologue plans: tail = Multiply(Sigmoid(gate), out) whose
  // sigmoid the tail alone reads and whose own single consumer is a
  // QuantizedMatmul (the group attach below verifies both). The group
  // dispatch rebuilds tail's value per workgroup with the elementwise
  // arithmetic and rounding, so the sigmoid and multiply dispatches the
  // pattern would create are deleted. Roles are registered only when a
  // group actually adopts the plan; otherwise every node evaluates
  // ordinarily.
  std::unordered_map<std::uintptr_t, OutgatePlan> outgate_plans;
  const bool outgate_on = outgate_fold_enabled();
  for (const auto& tail : tape) {
    if (!outgate_on || !is_op(&tail, typeid(Multiply)) ||
        tail.inputs().size() != 2 ||
        claimed.count(tail.id()) || uses[tail.id()] != 1 ||
        tail.dtype() == float32) {
      continue;
    }
    const array* left = lookup(tail.inputs()[0]);
    const array* right = lookup(tail.inputs()[1]);
    const array* sigmoid = is_op(left, typeid(Sigmoid)) ? left
        : (is_op(right, typeid(Sigmoid)) ? right : nullptr);
    if (!sigmoid || sigmoid->inputs().size() != 1 ||
        uses[sigmoid->id()] != 1 || claimed.count(sigmoid->id()) ||
        tail.dtype() != sigmoid->dtype() ||
        tail.primitive().stream() != sigmoid->primitive().stream()) {
      continue;
    }
    const array& out = sigmoid == left ? tail.inputs()[1] : tail.inputs()[0];
    if (out.id() == sigmoid->inputs()[0].id()) {
      continue;  // x * sigmoid(x) is silu, not an out-gate product
    }
    outgate_plans.emplace(
        tail.id(),
        OutgatePlan{sigmoid->inputs()[0], out, tail, sigmoid->id()});
    claimed.insert(sigmoid->id());
    claimed.insert(tail.id());
  }
  for (size_t i = 1; i < tape.size(); ++i) {
    const array& first = tape[i - 1];
    const array& second = tape[i];
    if (!is_op(&first, typeid(SliceUpdate)) ||
        !is_op(&second, typeid(SliceUpdate)) ||
        first.inputs().size() != 2 || second.inputs().size() != 2 ||
        first.dtype() != second.dtype() ||
        (first.dtype() != float16 && first.dtype() != bfloat16) ||
        first.shape() != second.shape() ||
        first.inputs()[0].shape() != second.inputs()[0].shape() ||
        first.inputs()[1].shape() != second.inputs()[1].shape() ||
        first.primitive().stream() != second.primitive().stream() ||
        static_cast<const SliceUpdate&>(first.primitive()).state() !=
            static_cast<const SliceUpdate&>(second.primitive()).state() ||
        std::get<0>(static_cast<const SliceUpdate&>(first.primitive()).state()) !=
            SliceUpdate::None ||
        claimed.count(first.id()) || claimed.count(second.id())) {
      continue;
    }
    size_t index = state->slice_update_pairs.size();
    state->slice_update_roles.emplace(first.id(), index);
    state->slice_update_roles.emplace(second.id(), index);
    state->slice_update_pairs.emplace_back(first, second);
    claimed.insert(first.id());
    claimed.insert(second.id());
    ++i;
  }

  if (!fused_gemv_enabled()) {
    return;
  }

  // Decode GEMV groups. Candidates: affine transposed 4-bit/group-64
  // QuantizedMatmul nodes whose x is a single row and whose weight,
  // scale, and bias inputs were evaluated before this eval began (the
  // group dispatches when its FIRST member evaluates, so a later
  // member's inputs must already be readable). Nodes sharing one x form
  // groups of up to kQmmVecMultiWeights in tape order. A member's Add
  // epilogue is the Add that is the node's only consumer, same dtype
  // and size; its other operand may be any array that is ready when the
  // group dispatches (checked then, refused otherwise). A group of one
  // member without an epilogue gains nothing and is not planned.
  std::unordered_map<std::uintptr_t, const array*> single_consumer;
  for (const auto& node : tape) {
    for (const auto& input : node.inputs()) {
      if (uses[input.id()] == 1) {
        single_consumer[input.id()] = &node;
      }
    }
  }
  std::unordered_map<std::uintptr_t, std::vector<const array*>> by_x;
  std::vector<std::uintptr_t> x_order;
  for (const auto& node : tape) {
    if (!is_op(&node, typeid(QuantizedMatmul)) ||
        node.inputs().size() != 4 || claimed.count(node.id())) {
      continue;
    }
    auto [group_size, bits, mode, transpose] =
        static_cast<const QuantizedMatmul&>(node.primitive()).state();
    if (mode != QuantizationMode::Affine || !transpose || bits != 4 ||
        group_size != 64 || lookup(node.inputs()[1]) ||
        lookup(node.inputs()[2]) || lookup(node.inputs()[3])) {
      continue;
    }
    const array& x = node.inputs()[0];
    if (x.ndim() < 2 || x.shape(-2) < 1 ||
        x.shape(-2) > static_cast<int>(kQmmVecTokenRowsMax) ||
        x.size() !=
            static_cast<size_t>(x.shape(-2)) *
                static_cast<size_t>(x.shape(-1))) {
      continue;
    }
    if (x.shape(-2) > 1 && !gemv_token_multi_enabled()) {
      continue;
    }
    auto [it, inserted] = by_x.try_emplace(x.id());
    if (inserted) {
      x_order.push_back(x.id());
    }
    it->second.push_back(&node);
  }
  for (auto x_id : x_order) {
    const auto& nodes = by_x[x_id];
    auto plan_it = outgate_plans.find(x_id);
    bool x_has_plan = plan_it != outgate_plans.end();
    // Token rows share the single-token group planner; the out-gate
    // prologue is the one fold the token column does not carry, so a
    // multi-row x never adopts an outgate plan (a dispatch-time refusal
    // after adoption is fatal by contract).
    const int x_rows = nodes.empty() ? 1 : nodes.front()->inputs()[0].shape(-2);
    if (x_rows > 1) {
      x_has_plan = false;
    }
    for (size_t start = 0; start < nodes.size();
         start += kQmmVecMultiWeights) {
      GemvGroup group;
      bool worth = x_has_plan && start == 0;
      for (size_t i = start; i < nodes.size() && i < start + kQmmVecMultiWeights;
           ++i) {
        const array& node = *nodes[i];
        GemvFusionMember member{node, std::nullopt, std::nullopt};
        if (auto consumer = single_consumer.find(node.id());
            consumer != single_consumer.end()) {
          const array* add = consumer->second;
          if (is_op(add, typeid(Add)) && add->inputs().size() == 2 &&
              add->dtype() == node.dtype() &&
              add->size() ==
                  static_cast<size_t>(x_rows) * node.size() &&
              add->primitive().stream() == node.primitive().stream() &&
              !claimed.count(add->id())) {
            const array* other = add->inputs()[0].id() == node.id()
                ? &add->inputs()[1]
                : &add->inputs()[0];
            const array* other_node = lookup(*other);
            // A bias arrives as Broadcast(bias) to the row's rank, a
            // view that adds no elements at one row (and only repeats
            // the n-vector across token rows at x_rows > 1); it sits in
            // the tape after the member in eval order, so read the bias
            // itself.
            if (is_op(other_node, typeid(Broadcast)) &&
                other_node->inputs().size() == 1 &&
                other_node->inputs()[0].size() == node.size() &&
                other_node->size() ==
                    static_cast<size_t>(x_rows) * node.size() &&
                other_node->inputs()[0].dtype() == other_node->dtype()) {
              other = &other_node->inputs()[0];
              other_node = lookup(*other);
            }
            if (other->id() != node.id() &&
                (!other_node ||
                 other_node->primitive().stream() ==
                     node.primitive().stream())) {
              member.epilogue = *add;
              member.addend = *other;
            }
          }
        }
        worth = worth || member.epilogue.has_value();
        group.members.push_back(std::move(member));
      }
      if (group.members.size() < 2 && !worth) {
        continue;
      }
      if (worth && start == 0 && x_has_plan) {
        group.outgate = std::move(plan_it->second);
      }
      size_t index = state->gemv_groups.size();
      for (const auto& member : group.members) {
        state->gemv_roles.emplace(member.node.id(), index);
        claimed.insert(member.node.id());
        if (member.epilogue) {
          state->gemv_roles.emplace(member.epilogue->id(), index);
          claimed.insert(member.epilogue->id());
        }
      }
      state->gemv_groups.push_back(std::move(group));
    }
  }
  // Out-gate attach validation. A group carrying a plan keeps it only
  // when the gate's only reader is the plan's sigmoid and out's only
  // reader the deleted multiply, and the folding producers share the
  // member stream. The sigmoid and multiply nodes then map to the
  // group as skip roles — true exactly when the group dispatched, so a
  // failed contract drops them back onto the ordinary path and the
  // multiply materializes x as before. (EagerRole.group carries the
  // gemv-group index for skip roles.)
  std::unordered_map<std::uintptr_t, size_t> outgate_reader_count;
  for (const auto& node : tape) {
    for (const auto& input : node.inputs()) {
      ++outgate_reader_count[input.id()];
    }
  }
  for (size_t gi = 0; gi < state->gemv_groups.size(); ++gi) {
    auto& group = state->gemv_groups[gi];
    if (!group.outgate) {
      continue;
    }
    OutgatePlan& plan = *group.outgate;
    auto gate_node = lookup(plan.gate);
    auto out_node = lookup(plan.out);
    const Stream& member_stream = group.members[0].node.primitive().stream();
    bool streams_ok =
        (!gate_node ||
         gate_node->primitive().stream() == member_stream) &&
        (!out_node || out_node->primitive().stream() == member_stream);
    if (!streams_ok || outgate_reader_count[plan.gate.id()] != 1 ||
        outgate_reader_count[plan.out.id()] != 1) {
      group.outgate.reset();
      continue;
    }
    // Everything the dispatch will check must be decided here: after
    // adoption the deleted nodes never evaluate, so a dispatch-time
    // refusal would leave their product unmaterialized. Whole-dense on
    // the two vectors is implied for tape-produced inputs (fresh whole
    // buffers, offset 0, row-contiguous) and the dispatch re-checks it;
    // the only genuinely open condition, input_ready, is guaranteed by
    // same-stream tape ordering.
    auto& encoder = get_command_encoder(member_stream);
    const auto& caps = encoder.device().capabilities();
    bool subgroup_ready = caps.subgroup_size == 32u &&
        (caps.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
    if (plan.x.dtype() != bfloat16 || !subgroup_ready ||
        encoder.device().compute().binding_limit() <
            kQmmVecMultiBindings + 3 ||
        plan.gate.dtype() != plan.x.dtype() ||
        plan.out.dtype() != plan.x.dtype() ||
        plan.x.ndim() < 2 || plan.x.shape(-2) != 1) {
      group.outgate.reset();
      continue;
    }
    state->roles.emplace(
        plan.sigmoid_id, EagerRole{gi, EagerStep::skip});
    state->roles.emplace(plan.x.id(), EagerRole{gi, EagerStep::skip});
  }
  // SwiGLU store epilogue fold. When a planned group is exactly the
  // gate and up projections of a planned swiglu chain — two members,
  // no Add epilogues, equal lengths, f16/bf16 — the group's dispatch
  // computes both dots per workgroup and stores only silu(gate) * up.
  // The gate leaf is read by the chain's sigmoid AND gate multiply,
  // the up leaf by the tail multiply; when those are each leaf's ONLY
  // readers, aliasing both projections onto the product is safe and
  // the chain's three nodes move to the group's roles: the first
  // member's eval fires the one dispatch and the standalone swiglu
  // dispatch never exists. Any runtime contract failure un-plans the
  // whole group onto the per-node path.
  if (!swiglu_plans.empty()) {
    std::unordered_map<std::uintptr_t, std::vector<std::uintptr_t>>
        consumers;
    for (const auto& node : tape) {
      for (const auto& input : node.inputs()) {
        consumers[input.id()].push_back(node.id());
      }
    }
    auto only_readers = [&](std::uintptr_t leaf,
                            std::vector<std::uintptr_t> want) {
      auto it = consumers.find(leaf);
      if (it == consumers.end()) {
        return false;
      }
      std::sort(it->second.begin(), it->second.end());
      std::sort(want.begin(), want.end());
      return it->second == want;
    };
    for (size_t gi = 0; gi < state->gemv_groups.size(); ++gi) {
      auto& group = state->gemv_groups[gi];
      if (group.members.size() != 2 || group.members[0].epilogue ||
          group.members[1].epilogue ||
          group.members[0].node.dtype() == float32) {
        continue;
      }
      const std::uintptr_t gate_id = group.members[0].node.id();
      const std::uintptr_t up_id = group.members[1].node.id();
      if (group.members[0].node.inputs()[1].shape(0) !=
          group.members[1].node.inputs()[1].shape(0)) {
        continue;
      }
      for (const auto& plan : swiglu_plans) {
        // The tape may hold the up projection before the gate one.
        // Weight slot 0 must be the gate for the epilogue's silu, so a
        // swapped group is reordered; the per-slot binding/output
        // wiring is positional, which makes the swap a no-op for the
        // plain multi-weight path.
        bool straight = plan.gate_id == gate_id && plan.up_id == up_id;
        bool swapped = plan.gate_id == up_id && plan.up_id == gate_id;
        if ((!straight && !swapped) ||
            !only_readers(plan.gate_id, {plan.sigmoid_id, plan.inner_id}) ||
            !only_readers(plan.up_id, {plan.out.id()})) {
          continue;
        }
        if (swapped) {
          std::swap(group.members[0], group.members[1]);
        }
        group.swiglu_out = plan.out;
        for (std::uintptr_t id :
             {plan.sigmoid_id, plan.inner_id, plan.out.id()}) {
          state->roles.erase(id);
          state->gemv_roles.emplace(id, gi);
        }
        break;
      }
    }
  }
  std::unordered_map<std::uintptr_t, std::vector<const array*>> dense_by_x;
  std::vector<std::uintptr_t> dense_x_order;
  for (const auto& node : tape) {
    if (!is_op(&node, typeid(Matmul)) || node.inputs().size() != 2 ||
        node.dtype() != bfloat16 || claimed.count(node.id())) {
      continue;
    }
    const array& x = node.inputs()[0];
    if (x.ndim() < 2 || x.shape(-2) != 1 ||
        x.size() != static_cast<size_t>(x.shape(-1))) {
      continue;
    }
    const array& source = dense_gemv_source(x);
    auto [it, inserted] = dense_by_x.try_emplace(source.id());
    if (inserted) {
      dense_x_order.push_back(source.id());
    }
    it->second.push_back(&node);
  }
  for (auto x_id : dense_x_order) {
    const auto& nodes = dense_by_x[x_id];
    for (size_t start = 0; start + 1 < nodes.size();
         start += kDenseVecMultiWeights) {
      DenseGemvGroup group;
      group.input = dense_gemv_source(nodes[start]->inputs()[0]);
      for (size_t i = start;
           i < nodes.size() && i < start + kDenseVecMultiWeights;
           ++i) {
        group.nodes.push_back(*nodes[i]);
      }
      if (group.nodes.size() < 2) {
        continue;
      }
      const size_t index = state->dense_gemv_groups.size();
      for (const auto& node : group.nodes) {
        state->dense_gemv_roles.emplace(node.id(), index);
        claimed.insert(node.id());
      }
      state->dense_gemv_groups.push_back(std::move(group));
    }
  }
  // Producer-direct KV cache writes: when one pair member's new rows
  // come from a RoPE node (keys) and the other's from a fused GEMV
  // Add epilogue through a provable chain of view-only ops (values),
  // both producers store their rows straight into the updated cache
  // copies and the merged pair dispatch is deleted. Any structural
  // mismatch keeps the ordinary plan; runtime aborts unwind to it as
  // well.
  if (kv_direct_enabled()) {
    for (size_t index = 0; index < state->slice_update_pairs.size();
         ++index) {
      auto& pair = state->slice_update_pairs[index];
      DirectPlan plans[2];
      bool classifiable = true;
      for (int side = 0; side < 2 && classifiable; ++side) {
        const array* update = lookup(pair.nodes[side].inputs()[1]);
        auto use_it =
            update ? uses.find(update->id()) : uses.end();
        if (!update || use_it == uses.end() || use_it->second != 1) {
          classifiable = false;
          break;
        }
        plans[side] = plan_keys_window(
                pair.nodes[side], update, uses, view_uses);
        if (plans[side].kind == DirectKind::none) {
          plans[side] = plan_values_window(
              pair.nodes[side],
              update,
              uses,
              state->gemv_groups,
              state->dense_gemv_roles,
              state->dense_gemv_groups,
              view_uses);
        }
      }
      if (!classifiable || plans[0].kind == DirectKind::none ||
          plans[1].kind == DirectKind::none ||
          plans[0].kind == plans[1].kind) {
        static int kv_trace = 0;
        if (std::getenv("MLX_OMARCHY_KV_TRACE") != nullptr && kv_trace < 6) {
          ++kv_trace;
          std::fprintf(
              stderr,
              "[kv-plan] pair %zu: classifiable=%d kinds=%d,%d "
              "dense_groups=%zu\n",
              index,
              static_cast<int>(classifiable),
              static_cast<int>(plans[0].kind),
              static_cast<int>(plans[1].kind),
              state->dense_gemv_groups.size());
          // MLX_OMARCHY_KV_TRACE detail: per-side update primitive,
          // shapes, strides, and the values-side terminal check, so a
          // refusal names its structural cause.
          {
            for (int side = 0; side < 2; ++side) {
              const array& member = pair.nodes[side];
              const array* upd = lookup(member.inputs()[1]);
              if (!upd) {
                std::fprintf(stderr, "[kv-plan]   side %d: update gone\n", side);
                continue;
              }
              const char* opname = "no-primitive";
              if (upd->has_primitive()) {
                opname = typeid(upd->primitive()).name();
              }
              std::fprintf(
                  stderr,
                  "[kv-plan]   side %d: upd=%s shape=[", side, opname);
              for (auto d : upd->shape()) {
                std::fprintf(stderr, "%d,", static_cast<int>(d));
              }
              std::fprintf(stderr, "] strides=[");
              for (auto s : upd->strides()) {
                std::fprintf(stderr, "%lld,", static_cast<long long>(s));
              }
              std::fprintf(stderr, "] inputs=%zu", upd->inputs().size());
              const array& base = member.inputs()[0];
              std::fprintf(stderr, " base shape=[");
              for (auto d : base.shape()) {
                std::fprintf(stderr, "%d,", static_cast<int>(d));
              }
              std::fprintf(stderr, "]\n");
            }
          }
        }
        continue;
      }
      int rope_side = plans[0].kind == DirectKind::keys_rope ? 0 : 1;
      int sum_side = 1 - rope_side;
      DirectPlan& sum_plan = plans[sum_side];
      // Copy the values window into its GEMV member BEFORE the pair
      // moves it: an optional move leaves the source empty, and the
      // member's window must carry the arrays themselves.
      if (sum_plan.kind == DirectKind::dense_values) {
        auto& windows =
            state->dense_gemv_groups[sum_plan.group_index].sum_windows;
        if (windows.size() <
            state->dense_gemv_groups[sum_plan.group_index].nodes.size()) {
          windows.resize(
              state->dense_gemv_groups[sum_plan.group_index].nodes.size());
        }
        windows[sum_plan.member_index] = sum_plan.window;
      } else {
        state->gemv_groups[sum_plan.group_index]
            .members[sum_plan.member_index]
            .sum_window = sum_plan.window;
      }
      pair.windows[0] = std::move(plans[rope_side].window);
      pair.windows[1] = std::move(sum_plan.window);
      state->rope_redirect_roles.emplace(
          pair.nodes[rope_side].inputs()[1].id(), index);
      for (const auto& view : sum_plan.view_chain) {
        state->reshape_redirect_roles.emplace(view.id(), index);
      }
      pair.direct = true;
    }
  }
  // Decode RoPE pair plan: two adjacent forward RoPE nodes sharing one
  // scalar offset input, one stream, float16 storage, identical rope
  // state (dims, traditional, base, scale, forward; no freqs), and no
  // passthrough tail (dims == last-axis extent) fuse into ONE
  // FastTrioRopePairF16 rope-pair dispatch when the first member evaluates.
  // The keys side keeps any planned producer-direct KV window; the
  // query side must not be one. Any runtime refusal un-plans the pair
  // and both nodes take the ordinary RoPE path unchanged.
  // (bf16 streams are covered by the F6 SliceUpdate pair widening below,
  // not by this f16-only rope-pair plan.)
  if (fused_trio_enabled()) {
    RopePairScanTrace trace;
    // The RoPE nodes of one decode step are NOT tape-adjacent: each
    // rope call materializes its scalar offset between them and the
    // cache write views interleave, so collect the rope subsequence in
    // tape order and pair consecutive members (query, key).
    std::vector<const array*> rope_nodes;
    for (const auto& node : tape) {
      if (!is_op(&node, typeid(fast::RoPE))) {
        continue;
      }
      if (node.inputs().size() != 2) {
        trace.rej_node_inputs += 1;
        continue;
      }
      if (node.inputs()[1].size() != 1) {
        trace.rej_node_offset += 1;
        continue;
      }
      if (node.dtype() != float16) {
        trace.rej_node_dtype += 1;
        continue;
      }
      if (!std::get<5>(
              static_cast<const fast::RoPE&>(node.primitive()).state())) {
        trace.rej_node_notforward += 1;
        continue;
      }
      if (trace.rope_shapes.size() < 6) {
        trace.rope_shapes.push_back(
            "idx" + std::to_string(static_cast<long>(&node - &tape.front())) +
            ":" + std::to_string(node.shape(-2)) + "x" +
            std::to_string(node.shape(-1)) + ":f" +
            std::to_string(std::get<1>(
                static_cast<const fast::RoPE&>(node.primitive()).state())) +
            ":off" +
            std::to_string(node.inputs()[1].status() ==
                           array::Status::available));
      }
      rope_nodes.push_back(&node);
    }
    trace.rope_nodes = rope_nodes.size();
    // Greedy sliding pairing: a stray unpairable rope node (47 vs 48 in
    // one measured plan) shifts fixed-parity pairs onto cross-layer
    // neighbors the runtime would have to refuse, so slide by one on a
    // failed pair instead.
    size_t i = 1;
    while (i < rope_nodes.size()) {
      const array& first = *rope_nodes[i - 1];
      const array& second = *rope_nodes[i];
      const auto& rope_first =
          static_cast<const fast::RoPE&>(first.primitive());
      const auto& rope_second =
          static_cast<const fast::RoPE&>(second.primitive());
      if (rope_second.state() != rope_first.state() ||
          std::get<1>(rope_first.state()) % 2 != 0 ||
          first.primitive().stream() != second.primitive().stream()) {
        trace.rej_state += 2;
        i += 1;
        continue;
      }
      if (!rope_offset_shared(first.inputs()[1], second.inputs()[1])) {
        trace.rej_offset += 2;
        i += 1;
        continue;
      }
      if (first.shape().size() != second.shape().size() ||
          first.shape(-1) != second.shape(-1) ||
          first.shape(-2) != second.shape(-2) ||
          first.shape(-1) != std::get<1>(rope_first.state())) {
        trace.rej_shape += 2;
        i += 1;
        continue;
      }
      if (claimed.count(first.id()) || claimed.count(second.id())) {
        trace.rej_claimed += 2;
        i += 1;
        continue;
      }
      if (state->rope_redirect_roles.count(first.id())) {
        trace.rej_redirect += 2;
        i += 1;
        continue;
      }
      const size_t index = state->rope_pairs.size();
      state->rope_pair_roles.emplace(first.id(), index);
      state->rope_pair_roles.emplace(second.id(), index);
      state->rope_pairs.push_back(RopePair{{first, second}});
      claimed.insert(first.id());
      claimed.insert(second.id());
      trace.pairs += 1;
      i += 2;
    }
    static int trio_trace_plans = 0;
    if (std::getenv("MLX_OMARCHY_TRIO_TRACE") != nullptr &&
        trace.rope_nodes > 0 && trio_trace_plans < 3) {
      ++trio_trace_plans;
      std::fprintf(
          stderr,
          "[trio-plan] rope_nodes=%zu rope_forward_scalar=%zu "
          "rej_node_inputs=%zu rej_node_offset=%zu rej_node_dtype=%zu "
          "rej_node_notforward=%zu rej_state=%zu rej_offset=%zu "
          "rej_shape=%zu rej_claimed=%zu rej_redirect=%zu "
          "pairs_planned=%zu rope_neighbor_after=%s rope_shapes=%s\n",
          trace.rope_nodes,
          trace.rope_forward_scalar,
          trace.rej_node_inputs,
          trace.rej_node_offset,
          trace.rej_node_dtype,
          trace.rej_node_notforward,
          trace.rej_state,
          trace.rej_offset,
          trace.rej_shape,
          trace.rej_claimed,
          trace.rej_redirect,
          trace.pairs,
          joined_rope_neighbors(trace.rope_neighbor_after).c_str(),
          joined_rope_neighbors(trace.rope_shapes).c_str());
    }
  }
}

EagerFusionScope::~EagerFusionScope() {
  delete eager_state;
  eager_state = static_cast<EagerFusionState*>(previous_);
}

// Out-gate prologue fold (sigmoid*multiply recomputed per o_proj workgroup).
// Chip-keyed like the other decode-path defaults: it gained on G13C (jw16,
// M1 Max) but every o_proj workgroup redoes the 4096-element gate on
// G13G (T8103), where decode dropped ~1.7% versus the wheel without it
// (H147). MLX_OMARCHY_OUTGATE_FOLD=0/1 forces either arm.
bool outgate_fold_enabled() {
  const char* v = std::getenv("MLX_OMARCHY_OUTGATE_FOLD");
  if (v != nullptr && *v != '\0') {
    return !(v[0] == '0' && v[1] == '\0');
  }
  const std::string& name = device().capabilities().device_name;
  bool legacy_g13 = name.find("G13") != std::string::npos &&
      name.find("G13C") == std::string::npos;
  return !legacy_g13;
}

bool fused_gemv_enabled() {
  return fused_chain_enabled() &&
      (std::getenv("MLX_OMARCHY_FUSED_GEMV") == nullptr ||
       env_flag("MLX_OMARCHY_FUSED_GEMV"));
}

// MLX_OMARCHY_FUSED_GEMV_SWIGLU=0 keeps the SwiGLU store epilogue off
// (the MLX_OMARCHY_FUSED_GEMV gate also covers it); on by default.
bool fused_gemv_swiglu_enabled() {
  return fused_gemv_enabled() &&
      (std::getenv("MLX_OMARCHY_FUSED_GEMV_SWIGLU") == nullptr ||
       env_flag("MLX_OMARCHY_FUSED_GEMV_SWIGLU"));
}

// MLX_OMARCHY_QMM_VEC_TOKEN_MULTI=0 keeps the grouped GEMV planner at
// the strict single-row fence (q_len 2..16 composes); on by default.
bool gemv_token_multi_enabled() {
  return fused_gemv_enabled() &&
      (std::getenv("MLX_OMARCHY_QMM_VEC_TOKEN_MULTI") == nullptr ||
       env_flag("MLX_OMARCHY_QMM_VEC_TOKEN_MULTI"));
}

bool fused_trio_enabled() {
  return fused_chain_enabled() &&
      (std::getenv("MLX_OMARCHY_FUSED_TRIO") == nullptr ||
       env_flag("MLX_OMARCHY_FUSED_TRIO"));
}

bool kv_direct_enabled() {
  return fused_chain_enabled() &&
      (std::getenv("MLX_OMARCHY_KV_DIRECT") == nullptr ||
       env_flag("MLX_OMARCHY_KV_DIRECT"));
}

KvDirectWindow* find_rope_kv_redirect(const array& out) {
  if (!eager_state) {
    return nullptr;
  }
  auto it = eager_state->rope_redirect_roles.find(out.id());
  if (it == eager_state->rope_redirect_roles.end()) {
    return nullptr;
  }
  auto& pair = eager_state->slice_update_pairs[it->second];
  if (!pair.direct || pair.state != SliceUpdatePair::State::pending ||
      !pair.values_committed) {
    return nullptr;
  }
  return &pair.windows[0].value();
}

void commit_rope_kv_redirect(const array& out) {
  if (!eager_state) {
    return;
  }
  auto it = eager_state->rope_redirect_roles.find(out.id());
  if (it == eager_state->rope_redirect_roles.end()) {
    return;
  }
  auto& pair = eager_state->slice_update_pairs[it->second];
  if (pair.direct && pair.state == SliceUpdatePair::State::pending) {
    pair.state = SliceUpdatePair::State::done;
  }
}

void abort_kv_direct() {
  if (!eager_state) {
    return;
  }
  for (auto& pair : eager_state->slice_update_pairs) {
    if (pair.direct && pair.state == SliceUpdatePair::State::pending) {
      pair.state = SliceUpdatePair::State::failed;
    }
  }
}

void commit_values_kv_write(const array& sum_node) {
  if (!eager_state) {
    return;
  }
  for (auto& pair : eager_state->slice_update_pairs) {
    if (pair.direct && pair.windows[1]->node.id() == sum_node.id()) {
      pair.values_committed = true;
      return;
    }
  }
}

bool try_eval_eager_fusion(array& node, const Stream& stream) {
  if (!eager_state) {
    return false;
  }
  if (auto rope_pair = eager_state->rope_pair_roles.find(node.id());
      rope_pair != eager_state->rope_pair_roles.end()) {
    auto& pair = eager_state->rope_pairs[rope_pair->second];
    if (pair.state == RopePair::State::done) {
      return true;
    }
    if (pair.state == RopePair::State::failed) {
      return false;
    }
    if (pair.nodes[0].status() == array::Status::evaluated &&
        pair.nodes[1].status() == array::Status::evaluated) {
      // A settle pass already evaluated both members: the pair can pop
      // inside a nested settle scope before the outer tape reaches it,
      // and each scope plans the same pairs. One dispatch wrote both
      // outputs; re-firing would only rewrite them.
      pair.state = RopePair::State::done;
      return true;
    }
    bool fired = dispatch_rope_pair(pair.nodes, stream);
    pair.state = fired ? RopePair::State::done : RopePair::State::failed;
    static int trio_run_trace = 0;
    if (!fired && std::getenv("MLX_OMARCHY_TRIO_TRACE") != nullptr &&
        trio_run_trace < 3) {
      ++trio_run_trace;
      std::fprintf(
          stderr, "[trio-run] pair un-planned: runtime contract refused\n");
    }
    return pair.state == RopePair::State::done;
  }
  if (auto update = eager_state->slice_update_roles.find(node.id());
      update != eager_state->slice_update_roles.end()) {
    auto& pair = eager_state->slice_update_pairs[update->second];
    if (pair.direct) {
      if (pair.state == SliceUpdatePair::State::done) {
        return true;
      }
      // The producers did not commit (abort or fence): un-plan the
      // direct write and run the ordinary merged pair dispatch below.
      pair.direct = false;
      pair.state = SliceUpdatePair::State::pending;
    }
    if (pair.state == SliceUpdatePair::State::done) {
      return true;
    }
    if (pair.state == SliceUpdatePair::State::failed) {
      return false;
    }
    auto result = dispatch_slice_update_pair(pair.nodes, stream);
    if (result == SliceUpdatePairDispatch::done) {
      pair.state = SliceUpdatePair::State::done;
      return true;
    }
    if (pair.state == SliceUpdatePair::State::deferred) {
      throw std::runtime_error(
          "[mlx-omarchy] deferred SliceUpdate pair did not become ready");
    }
    if (result == SliceUpdatePairDispatch::not_ready &&
        node.id() == pair.nodes[0].id()) {
      pair.state = SliceUpdatePair::State::deferred;
      return true;
    }
    pair.state = SliceUpdatePair::State::failed;
    return false;
  }
  if (auto reshape = eager_state->reshape_redirect_roles.find(node.id());
      reshape != eager_state->reshape_redirect_roles.end()) {
    auto& pair = eager_state->slice_update_pairs[reshape->second];
    if (!pair.values_committed) {
      return false;
    }
    const auto& window = *pair.windows[1];
    Strides view_strides(window.strides, window.strides + window.ndim);
    array::Flags flags;
    flags.contiguous = false;
    flags.row_contiguous = false;
    flags.col_contiguous = false;
    node.copy_shared_buffer(
        window.base, view_strides, flags, node.data_size(), window.offset);
    return true;
  }
  if (auto dense = eager_state->dense_gemv_roles.find(node.id());
      dense != eager_state->dense_gemv_roles.end()) {
    auto& group = eager_state->dense_gemv_groups[dense->second];
    if (group.state == DenseGemvGroup::State::pending) {
      // Producer-direct values windows: same contract as the quantized
      // path. In place, the member's rows land straight in the live
      // cache; otherwise a fresh copy is enqueued first so the rows
      // land in order. A refusal un-plans the whole direct write and
      // unwinds to the merged pair dispatch.
      for (auto& window : group.sum_windows) {
        if (!window) {
          continue;
        }
        auto& w = *window;
        if (w.node.data_shared_ptr() != nullptr ||
            w.base.data_shared_ptr() == nullptr ||
            (w.base.dtype() != float16 && w.base.dtype() != bfloat16) ||
            !w.base.flags().row_contiguous ||
            w.base.size() != w.base.data_size() ||
            w.base.offset() % w.base.itemsize() != 0 ||
            w.base.data_shared_ptr() == nullptr) {
          abort_kv_direct();
          group.sum_windows.clear();
          break;
        }
        if (w.in_place) {
          w.node.copy_shared_buffer(
              w.base, w.base.strides(), w.base.flags(), w.base.data_size());
        } else {
          // Not provably exclusive: materialize a fresh cache copy and
          // let the window scatter target it (the merged pair dispatch
          // this replaces does exactly the same copy).
          w.node.set_data(allocator().malloc(w.node.nbytes()));
          copy_gpu(
              w.base,
              w.node,
              w.base.flags().contiguous ? CopyType::Vector : CopyType::General,
              stream);
        }
        get_command_encoder(stream).add_temporary(w.node);
        get_command_encoder(stream).add_temporary(w.base);
        commit_values_kv_write(w.node);
      }
      group.state = dispatch_dense_gemv_group(
                        group.nodes,
                        *group.input,
                        group.sum_windows.empty()
                            ? nullptr
                            : group.sum_windows.data(),
                        stream)
          ? DenseGemvGroup::State::done
          : DenseGemvGroup::State::failed;
    }
    return group.state == DenseGemvGroup::State::done;
  }
  if (auto gemv = eager_state->gemv_roles.find(node.id());
      gemv != eager_state->gemv_roles.end()) {
    auto& group = eager_state->gemv_groups[gemv->second];
    if (group.state == GemvGroup::State::pending) {
      group.state = dispatch_quantized_gemv_group(
                        group.members,
                        group.swiglu_out ? &*group.swiglu_out : nullptr,
                        stream,
                        group.outgate ? &*group.outgate : nullptr)
          ? GemvGroup::State::done
          : GemvGroup::State::failed;
      if (group.state == GemvGroup::State::failed && group.outgate) {
        // The deleted sigmoid and multiply never ran and cannot be
        // replayed here; a refusal after adoption is a planner bug,
        // not a fallback path (plan-time validation covers everything
        // but same-stream readiness, which tape ordering guarantees).
        throw std::runtime_error(
            "[mlx-omarchy] out-gate prologue group refused after fold "
            "adoption");
      }
    }
    return group.state == GemvGroup::State::done;
  }
  auto role_it = eager_state->roles.find(node.id());
  if (role_it == eager_state->roles.end()) {
    return false;
  }
  const auto role = role_it->second;
  if (role.step == EagerStep::skip) {
    // A node deleted by an adopted out-gate plan: true while its group
    // is pending or done (the group's dispatch owns the work and
    // materializes x; EagerRole.group carries the gemv group index),
    // false on a failed contract so the node evaluates ordinarily and
    // the multiply produces x as before.
    return eager_state->gemv_groups[role.group].state !=
        GemvGroup::State::failed;
  }
  if (role.step == EagerStep::sigmoid) {
    // The chain interpreter binds up to kMaxChainLeaves + 3 buffers.
    if (device().compute().binding_limit() < kMaxChainLeaves + 3) {
      return false;
    }
    auto [chain_it, inserted] =
        eager_state->chains.try_emplace(role.group, true);
    if (!inserted ||
        !chain_it->second.try_add(node, node.inputs(), false)) {
      eager_state->chains.erase(role.group);
      return false;
    }
    return true;
  }

  auto chain_it = eager_state->chains.find(role.group);
  if (chain_it == eager_state->chains.end()) {
    return false;
  }
  if (!chain_it->second.try_add(node, node.inputs(), false)) {
    chain_it->second.evaluate_tail(stream);
    eager_state->chains.erase(chain_it);
    return false;
  }
  if (role.step == EagerStep::output_mul) {
    chain_it->second.evaluate_tail(stream);
    eager_state->chains.erase(chain_it);
  }
  return true;
}
} // namespace mlx::core::omarchy
