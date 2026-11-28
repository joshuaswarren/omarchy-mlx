// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/fused_chain.h"

#include <algorithm>
#include <array>
#include <chrono>
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
#include <utility>
#include <vector>

#include "mlx/backend/common/slicing.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/tape_index.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::omarchy {

bool fused_chain_enabled() {
  return omarchy::env_str("MLX_OMARCHY_FUSED_CHAIN") == nullptr ||
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
    // mx.rsqrt is Sqrt with the recip flag; mapping it to ChainSqrt returned sqrt(x).
    return static_cast<const Sqrt&>(p).state() ? ChainRsqrt : ChainSqrt;
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
    if (out.dtype() == float16 && fused_trio_enabled(encoder.device())) {
      // The trio pipeline keeps the same near-copied swiglu arithmetic
      // on the shared decode pipeline; out rides binding 4 and the
      // materialized intermediates bindings 5/6 (the fast_trio.comp
      // slot map). The ComputeParams swiglu mapping is unchanged.
      std::array<ComputeBinding, 7> trio_bindings{
          binding(chain.leaves[leaves->first]),
          binding(chain.leaves[leaves->second]),
          binding(out),
          binding(out),
          binding(out),
          binding(materialize_nodes ? chain.node_arrays[0] : out),
          binding(materialize_nodes ? chain.node_arrays[1] : out)};
      encoder.dispatch_compute(
          ComputeKernel::FastTrioSwigluF16,
          trio_bindings,
          params,
          compute_dispatch_group_count(chain.count / 4u));
      return;
    }
    std::array<ComputeBinding, 5> bindings{
        binding(chain.leaves[leaves->first]),
        binding(chain.leaves[leaves->second]),
        binding(out),
        binding(materialize_nodes ? chain.node_arrays[0] : out),
        binding(materialize_nodes ? chain.node_arrays[1] : out)};
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
        binding(chain.leaves[0]),
        binding(chain.leaves[0]),
        binding(out),
        binding(out),
        binding(out)};
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
  std::memcpy(program_buffer.raw_ptr(), chain.program.data(), program_bytes);
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
      binding(out),
      binding(out),
      binding(out),
      binding(program_keeper),
      binding(out),
      binding(out)};
  for (size_t i = 0; i < chain.leaves.size(); ++i) {
    bindings[i] = binding(chain.leaves[i]);
  }
  if (materialize_intermediates && !chain.node_arrays.empty()) {
    bindings[2] = binding(chain.node_arrays[0]);
    if (chain.node_arrays.size() > 2) {
      bindings[5] = binding(chain.node_arrays[1]);
    }
  }

  // MLX_OMARCHY_FUSED_CHAIN_LANES=128 selects the four-subgroup
  // variants. The 256-lane and 128-lane kernels must produce
  // byte-identical output; the fused-chain test asserts it.
  const char* lanes_env = omarchy::env_str("MLX_OMARCHY_FUSED_CHAIN_LANES");
  const bool lanes128 = lanes_env && std::strcmp(lanes_env, "128") == 0;
  auto kernel = lanes128 ? ComputeKernel::FusedChainF32L128
                         : ComputeKernel::FusedChainF32;
  if (out.dtype() == float16) {
    kernel = lanes128 ? ComputeKernel::FusedChainF16L128
                      : ComputeKernel::FusedChainF16;
  } else if (out.dtype() == bfloat16) {
    kernel = lanes128 ? ComputeKernel::FusedChainBF16L128
                      : ComputeKernel::FusedChainBF16;
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
  // Indices into rope_norms for norm-carrying members (the fold the
  // pair dispatch absorbs); -1 means the plain 2-input form.
  std::array<int, 2> norm_folds{-1, -1};
  // dispatching marks a pair whose dispatch is on the stack: a nested
  // eval reaching either member must not re-fire dispatch_rope_pair
  // (its chain settle walks ancestors that include pair members, so a
  // re-entrant pair dispatch would recurse forever). The re-entrant
  // eval falls through to the fold arm or the ordinary path instead,
  // which materializes the node; the in-flight pair kernel then writes
  // bit-identical data over it.
  enum class State : uint8_t { pending, dispatching, done, failed } state{
      State::pending};
};

// A planned residual Add -> RMSNorm fold (M2D-2): the add produces the
// bf16 residual and the norm reads exactly it, same stream and shape.
// Pending until one FastAddRmsNormAppleBF16 dispatch at the add's tape
// position writes both outputs, then done; failed un-plans and both
// nodes take their ordinary paths.
struct AddNormPair {
  std::array<array, 2> nodes;
  enum class State : uint8_t { pending, done, failed } state{State::pending};
};

// A planned RMSNorm -> RoPE fold: the norm's only consumer is a plain
// two-input RoPE, reached directly or through one view op. Pending
// until the rope evaluates, then done (one FastRopeNorm dispatch read
// the pre-norm input and wrote the rope output) or failed (the norm and
// rope both take their own paths).
struct RopeNormFold {
  array norm;
  array rope;
  // The view node between norm and rope (Transpose/Reshape); absent
  // when the rope input is the norm output itself. Its layout is
  // replayed over the norm's input buffer for the fused dispatch.
  std::optional<array> view;
  // Eval detaches tape nodes (clears inputs and primitive), so the
  // fold keeps the pieces dispatch and the failure path need.
  array x;
  array w;
  std::shared_ptr<Primitive> norm_prim;
  std::shared_ptr<Primitive> view_prim;
  float eps;
  enum class State : uint8_t { pending, done, failed } state{State::pending};
};

// A planned quantized-embedding prologue fold: a dequantize-direction
// fast::Quantize whose three inputs each trace through single-use view
// ops to a Gather taking one full row of a table by the same index
// array. Pending until the Quantize evaluates, then done (one
// gather-dequant dispatch wrote the output) or failed (the gathers,
// views and dequantize replay on their own paths).
struct PrologueFold {
  array quant;
  array ids;
  array w;
  array scales;
  array biases;
  // Skipped tape nodes in dependency order (per arm: gather first,
  // then its views outward to the quant input) with the pieces the
  // failure replay needs: tape nodes are detached by eval time.
  struct ReplayStep {
    array node;
    std::shared_ptr<Primitive> prim;
    std::vector<array> inputs;
  };
  std::vector<ReplayStep> replay;
  int bits = 0;
  int group_size = 0;
  int64_t words_per_row = 0;
  int64_t groups_per_row = 0;
  int64_t nrows = 0;
  // Optional fused downstream fast::RMSNorm (MLX_OMARCHY_PROLOGUE_NORM):
  // writes both embed_out (quant) and norm_out in one 256-thread dispatch.
  std::optional<array> norm_out;
  std::optional<array> norm_weight;
  float norm_eps = 0.0f;
  bool norm_done = false;
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
  // Trailing fast::RMSNorm epilogue over the single member's Add sum.
  std::optional<ResNormPlan> res_norm;
  bool res_norm_done{false};
  // Fused RMSNorm+RoPE q/k epilogue over two member outputs; the keys
  // side stores through the pair's producer-direct window.
  std::optional<RopeEpiPlan> rope_epi;
  bool rope_epi_done{false};
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

// Decode-rows SDPA out-transpose fold: the verify route's (1,H,S,D)
// output feeds a single-consumer Transpose(0,2,1,3) whose view a
// materializing Reshape copies into (1,S,H*D). The rows kernel can
// write the transposed layout straight into the reshape's buffer
// (out_strides[1] head stride D, out_strides[2] row stride H*D),
// deleting the copy dispatch. reshape is the Reshape node; chain
// holds the view nodes between it and the attention so their evals
// skip once the fold fires.
struct SdpaTransposeFold {
  array reshape;
  std::vector<array> chain;
  bool fired{false};
};

struct SdpaOutgateFold {
  SdpaOutgatePlan plan;
  bool fired{false};
};

struct SdpaOnlineRopePlan {
  array q_rope;
  array k_rope;
  array sdpa;
  array q_raw;
  array k_raw;
  array offset;
  std::optional<array> q_weight;
  std::optional<array> k_weight;
  std::optional<array> q_view;
  std::optional<array> k_view;
  std::shared_ptr<Primitive> q_view_prim;
  std::shared_ptr<Primitive> k_view_prim;
  std::shared_ptr<Primitive> q_rope_prim;
  std::shared_ptr<Primitive> k_rope_prim;
  std::vector<array> q_rope_inputs;
  std::vector<array> k_rope_inputs;
  int q_norm_fold{-1};
  int k_norm_fold{-1};
  size_t pair_index{0};
  bool has_norm{false};
  float eps{0.0f};
  float base{0.0f};
  float scale{1.0f};
  int dims{0};
  SdpaOnlineRopeCtx ctx;
  bool q_skipped{false};
  bool k_skipped{false};
  enum class State : uint8_t { pending, k_bound, done, failed } state{
      State::pending};
};

struct GdnInProjBaFold {
  enum class State : uint8_t { pending, done, failed };
  GdnInProjBaPlan plan;
  array a_node;
  std::shared_ptr<Primitive> a_prim;
  std::vector<array> a_inputs;
  array b_node;
  std::shared_ptr<Primitive> b_prim;
  std::vector<array> b_inputs;
  State state{State::pending};
};

struct EagerFusionState {
  std::unordered_map<std::uintptr_t, EagerRole> roles;
  std::unordered_map<std::uintptr_t, FusedChain> chains;
  TapeIndex<size_t> gemv_roles;
  std::vector<GemvGroup> gemv_groups;
  TapeIndex<size_t> res_norm_skip_roles;
  std::vector<SdpaTransposeFold> sdpa_transpose_folds;
  // attention out array id -> fold index; reshape/view ids -> fold index
  std::unordered_map<std::uintptr_t, size_t> sdpa_transpose_plans;
  std::unordered_map<std::uintptr_t, size_t> sdpa_transpose_skip_roles;
  std::vector<SdpaOutgateFold> sdpa_outgate_folds;
  std::unordered_map<std::uintptr_t, size_t> sdpa_outgate_plans;
  std::unordered_map<std::uintptr_t, size_t> sdpa_outgate_skip_roles;
  std::vector<SdpaOnlineRopePlan> sdpa_online_rope_plans;
  std::unordered_map<std::uintptr_t, size_t> sdpa_online_rope_skip_roles;
  std::unordered_map<std::uintptr_t, size_t> sdpa_online_rope_su_roles;
  std::unordered_map<std::uintptr_t, size_t> sdpa_online_rope_roles;
  TapeIndex<size_t> dense_gemv_roles;
  std::vector<DenseGemvGroup> dense_gemv_groups;
  TapeIndex<size_t> slice_update_roles;
  std::vector<SliceUpdatePair> slice_update_pairs;
  TapeIndex<size_t> rope_redirect_roles;
  TapeIndex<size_t> reshape_redirect_roles;
  TapeIndex<size_t> rope_pair_roles;
  std::vector<RopePair> rope_pairs;
  TapeIndex<size_t> add_norm_roles;
  std::vector<AddNormPair> add_norm_pairs;
  TapeIndex<size_t> rope_norm_skip_roles;
  TapeIndex<size_t> rope_norm_roles;
  std::vector<RopeNormFold> rope_norms;
  TapeIndex<size_t> prologue_skip_roles;
  TapeIndex<size_t> prologue_fold_roles;
  TapeIndex<size_t> prologue_norm_skip_roles;
  std::vector<PrologueFold> prologue_folds;
  TapeIndex<size_t> gdn_ba_skip_roles;
  TapeIndex<size_t> gdn_ba_roles;
  std::vector<GdnInProjBaFold> gdn_ba_folds;

  void prepare(std::size_t tape_size) {
    gemv_roles.prepare(tape_size);
    res_norm_skip_roles.prepare(tape_size);
    dense_gemv_roles.prepare(tape_size);
    slice_update_roles.prepare(tape_size);
    rope_redirect_roles.prepare(tape_size);
    reshape_redirect_roles.prepare(tape_size);
    rope_pair_roles.prepare(tape_size);
    add_norm_roles.prepare(tape_size);
    rope_norm_skip_roles.prepare(tape_size);
    rope_norm_roles.prepare(tape_size);
    prologue_skip_roles.prepare(tape_size);
    prologue_fold_roles.prepare(tape_size);
    prologue_norm_skip_roles.prepare(tape_size);
    gdn_ba_skip_roles.prepare(tape_size);
    gdn_ba_roles.prepare(tape_size);
  }

  // Called when the scope ends and the state returns to the pool: drops
  // every array reference the plans held so buffers can be reused
  // between evals. Container capacity is kept for the next scope.
  void reset() {
    roles.clear();
    chains.clear();
    gemv_groups.clear();
    dense_gemv_groups.clear();
    slice_update_pairs.clear();
    rope_pairs.clear();
    add_norm_pairs.clear();
    rope_norms.clear();
    prologue_folds.clear();
    gdn_ba_folds.clear();
    sdpa_transpose_folds.clear();
    sdpa_transpose_plans.clear();
    sdpa_transpose_skip_roles.clear();
    sdpa_outgate_folds.clear();
    sdpa_outgate_plans.clear();
    sdpa_outgate_skip_roles.clear();
    sdpa_online_rope_plans.clear();
    sdpa_online_rope_skip_roles.clear();
    sdpa_online_rope_su_roles.clear();
    sdpa_online_rope_roles.clear();
  }
};

thread_local EagerFusionState* eager_state = nullptr;
thread_local double eager_dispatch_us = 0.0;
thread_local size_t eager_dispatch_calls = 0;

// One spare state per live scope, depth-capped: a decode eval builds the
// same ~10 role tables every step, and unordered_map pays a node
// allocation per insert plus bucket growth each time. Reused states keep
// their bucket/vector capacity so a steady-state eval allocates no map
// nodes at all for the TapeIndex tables and none for growth elsewhere.
// Nested scopes (a settle eval inside a dispatch) take distinct pool
// entries, so live tables are never shared.
std::vector<std::unique_ptr<EagerFusionState>>& fusion_state_pool() {
  static thread_local std::vector<std::unique_ptr<EagerFusionState>> pool;
  return pool;
}

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

// A producer-direct window may write its cache in place only when this
// SliceUpdate holds the last handle to the cache and to its storage: no
// other tape node, view, snapshot, or caller can then still read the
// pre-update rows (MLX buffer donation). Ask before a plan copies the
// base handle, which would count as a second holder.
// MLX_OMARCHY_KV_INPLACE=0 forces the copy.
bool kv_window_in_place(const array& member) {
  const bool env_on = omarchy::env_str("MLX_OMARCHY_KV_INPLACE") == nullptr ||
      env_flag("MLX_OMARCHY_KV_INPLACE");
  const array& base = member.inputs()[0];
  const bool donatable = base.is_donatable();
  // MLX_OMARCHY_KV_TRACE: donation is the in-place gate; a refusal that
  // only shows up as a missing alias names no cause without the split.
  if (omarchy::env_str("MLX_OMARCHY_KV_TRACE") != nullptr) {
    auto data = base.data_shared_ptr();
    std::fprintf(
        stderr,
        "[kv-inplace] env=%d donatable=%d data_uses=%ld\n",
        static_cast<int>(env_on),
        static_cast<int>(donatable),
        data ? static_cast<long>(data.use_count()) : -1L);
  }
  return env_on && donatable;
}

// Keys: the RoPE kernel writes (matrix, time, feature) output strides,
// so the window's (batch, kv-head) axes must form one regular matrix
// axis. The RoPE fence (forward, scalar offset, fused-path conditions)
// is re-checked at dispatch; aborting there unwinds to the merged pair
// dispatch.
DirectPlan plan_keys_window(const array& member, const array* update) {
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
  const bool in_place = kv_window_in_place(member);
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
    const TapeIndex<size_t>& uses,
    const std::vector<GemvGroup>& groups,
    const TapeIndex<size_t>& dense_roles,
    std::vector<DenseGemvGroup>& dense_groups) {
  DirectPlan plan;
  auto use_count = [&](const array& value) {
    auto* it = uses.find(value.id());
    return it == nullptr ? size_t{0} : *it;
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
  if (dense_it != nullptr) {
    if (use_count(*sum) != 1) {
      return plan;
    }
    const auto& group_nodes = dense_groups[*dense_it].nodes;
    for (size_t mi = 0; mi < group_nodes.size(); ++mi) {
      if (group_nodes[mi].id() == sum->id()) {
        plan.group_index = *dense_it;
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
      static_cast<size_t>(upd.shape(1)) * upd.shape(2) * upd.shape(3) !=
          sum->size()) {
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
  // The seq axis must map onto contiguous sum rows: element (head, m,
  // dim) of the update reads the sum buffer at m * n_kv*head_dim +
  // head * head_dim + dim, which is the multi-token window store the
  // rows kernel emits.
  if (map_shape != upd.shape() || map_strides[3] != 1 ||
      map_strides[1] != static_cast<ptrdiff_t>(head_dim) ||
      map_strides[2] !=
          static_cast<ptrdiff_t>(
              static_cast<uint64_t>(upd.shape(1)) * head_dim)) {
    return plan;
  }
  for (const array* view : chain) {
    plan.view_chain.push_back(*view);
  }
  const bool in_place = kv_window_in_place(member);
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
      in_place};
  if (plan.kind == DirectKind::none) {
    plan.kind = DirectKind::values_sum;
  }
  return plan;
}

// The device that executes a tape node: its primitive's stream decides,
// not the default device index. Nodes on a CPU stream have none, and no
// fused route plans them.
const Device* stream_device(const array& node) {
  const Stream& stream = node.primitive().stream();
  if (stream.device.type != mlx::core::Device::gpu) {
    return nullptr;
  }
  return &device(stream.device.index);
}

bool node_gemv_enabled(const array& node) {
  const Device* d = stream_device(node);
  return d != nullptr && fused_gemv_enabled(*d);
}

// Epilogue-only group route (MLX_OMARCHY_QMM_EPILOGUE_GEMV): on parts where
// the multi-weight fused GEMV defaults off but the residual-norm epilogue
// defaults on (G14C), single-member groups carrying an Add epilogue still
// plan; multi-member x-sharing folds stay off. See device.h.
bool gemv_epilogue_only_route_enabled(const Device& device) {
  if (!fused_chain_enabled()) {
    return false;
  }
  const char* env = omarchy::env_str("MLX_OMARCHY_QMM_EPILOGUE_GEMV");
  if (env != nullptr && env[0] != '\0') {
    return std::strcmp(env, "0") != 0;
  }
  return gemv_epilogue_only_part(device.capabilities().device_name);
}

bool node_epilogue_gemv_enabled(const array& node) {
  const Device* d = stream_device(node);
  return d != nullptr && gemv_epilogue_only_route_enabled(*d);
}

bool node_trio_enabled(const array& node) {
  const Device* d = stream_device(node);
  return d != nullptr && fused_trio_enabled(*d);
}

// ---------------------------------------------------------------------------
// Per-shape fold-plan cache (thread-local).
//
// Autoregressive decode repeats the exact same graph topology on every token
// while only array handles, KV-cache write offsets, and KV-view slice lengths
// advance. We hash the tape's full structural + attribute + config signature
// (masking only the dynamic KV sequence axis on 4D Slice views over 4D
// SliceUpdate nodes while encoding SDPA online-rope regime transitions),
// cache the resulting fold plan as tape-index templates, and instantiate a
// fresh EagerFusionState on cache hit in linear time over the planned folds.
// Runtime-dependent checks (SliceUpdate state match, KV direct-window geometry
// and buffer donation, shared RoPE offsets, RopeNorm view flags) are
// revalidated on every replay; any mismatch declines the cached plan cleanly
// and falls through to the full planner.
// ---------------------------------------------------------------------------

struct FoldKeyHasher {
  uint64_t lo = 0x243f6a8885a308d3ull;
  uint64_t hi = 0x13198a2e03707344ull;

  void mix(uint64_t v) {
    lo ^= v + 0x9e3779b97f4a7c15ull + (lo << 6) + (lo >> 2);
    lo *= 0xbf58476d1ce4e5b9ull;
    lo ^= lo >> 31;
    hi ^= (v ^ lo) + 0x517cc1b727220a95ull + (hi << 7) + (hi >> 3);
    hi *= 0x94d049bb133111ebull;
    hi ^= hi >> 31;
  }

  void mix_bytes(const void* data, size_t n) {
    mix(static_cast<uint64_t>(n));
    const auto* p = static_cast<const uint8_t*>(data);
    size_t i = 0;
    while (i + 8 <= n) {
      uint64_t w = 0;
      std::memcpy(&w, p + i, 8);
      mix(w);
      i += 8;
    }
    if (i < n) {
      uint64_t tail = 0;
      std::memcpy(&tail, p + i, n - i);
      mix(tail);
    }
  }

  void mix_float(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    mix(static_cast<uint64_t>(bits));
  }
};

bool fold_planner_trace_active() {
  return std::getenv("MLX_OMARCHY_SWIGLU_TRACE") != nullptr ||
      omarchy::env_str("MLX_OMARCHY_KV_TRACE") != nullptr ||
      std::getenv("MLX_OMARCHY_ROPE_EPI_TRACE") != nullptr ||
      omarchy::env_str("MLX_OMARCHY_ROPE_NORM_TRACE") != nullptr ||
      omarchy::env_str("MLX_OMARCHY_SDPA_ROPE_TRACE") != nullptr ||
      omarchy::env_str("MLX_OMARCHY_TRIO_TRACE") != nullptr;
}

void hash_fold_config(FoldKeyHasher& h) {
  auto mix_env = [&](const char* val) {
    if (val == nullptr) {
      h.mix(0x100000001ull);
    } else {
      h.mix(0x200000002ull);
      h.mix_bytes(val, std::strlen(val));
    }
  };
  mix_env(omarchy::env_str("MLX_OMARCHY_FUSED_CHAIN"));
  mix_env(omarchy::env_str("MLX_OMARCHY_NO_FUSION_GATE"));
  mix_env(omarchy::env_str("MLX_OMARCHY_OUTGATE_FOLD"));
  mix_env(omarchy::env_str("MLX_OMARCHY_GDN_INPROJ_BA_FOLD"));
  mix_env(omarchy::env_str("MLX_OMARCHY_GDN_COALESCED"));
  mix_env(omarchy::env_str("MLX_OMARCHY_FUSED_GEMV"));
  mix_env(omarchy::env_str("MLX_OMARCHY_QMM_EPILOGUE_GEMV"));
  mix_env(omarchy::env_str("MLX_OMARCHY_GEMV_BALANCED_QKV"));
  mix_env(omarchy::env_str("MLX_OMARCHY_FUSED_GEMV_SWIGLU"));
  mix_env(omarchy::env_str("MLX_OMARCHY_QMM_VEC_TOKEN_MULTI"));
  mix_env(omarchy::env_str("MLX_OMARCHY_QMM_VEC_TOKEN_FOLD"));
  mix_env(omarchy::env_str("MLX_OMARCHY_QMM_VEC_Q4_WORD"));
  mix_env(omarchy::env_str("MLX_OMARCHY_KV_DIRECT"));
  mix_env(omarchy::env_str("MLX_OMARCHY_QMM_RESIDUAL_NORM"));
  mix_env(std::getenv("MLX_OMARCHY_ROPE_EPI"));
  mix_env(omarchy::env_str("MLX_OMARCHY_SDPA_OUTGATE"));
  mix_env(omarchy::env_str("MLX_OMARCHY_ADD_NORM_FUSE"));
  mix_env(omarchy::env_str("MLX_OMARCHY_ROPE_NORM_FOLD"));
  mix_env(omarchy::env_str("MLX_OMARCHY_SDPA_ONLINE_ROPE"));
  mix_env(omarchy::env_str("MLX_OMARCHY_SDPA_ONLINE"));
  mix_env(omarchy::env_str("MLX_OMARCHY_FUSED_TRIO"));
  mix_env(omarchy::env_str("MLX_OMARCHY_ROPE_NORM_PAIR"));
  mix_env(omarchy::env_str("MLX_OMARCHY_PROLOGUE_FOLD"));
  mix_env(std::getenv("MLX_OMARCHY_PROLOGUE_NORM"));
}

constexpr uint32_t kExternalSlotBit = 0x80000000u;

FoldPlanKey compute_fold_plan_key_impl(
    const std::deque<array>& tape,
    TapeIndex<uint32_t>& id_map) {
  id_map.prepare(tape.size() * 2 + 16);
  const uint32_t n = static_cast<uint32_t>(tape.size());
  for (uint32_t i = 0; i < n; ++i) {
    id_map.put_if_absent(tape[i].id(), i);
  }

  FoldKeyHasher h;
  hash_fold_config(h);
  h.mix(static_cast<uint64_t>(n));

  uint32_t next_ext = 0;
  for (uint32_t i = 0; i < n; ++i) {
    const array& node = tape[i];
    const bool has_prim = node.has_primitive();
    const std::type_info* ti = nullptr;
    if (has_prim) {
      Primitive& p = node.primitive();
      ti = &typeid(p);
      h.mix(static_cast<uint64_t>(ti->hash_code()));
      const Stream& st = p.stream();
      h.mix(
          (static_cast<uint64_t>(static_cast<int>(st.device.type)) << 32) |
          static_cast<uint32_t>(st.device.index));
      h.mix(static_cast<uint64_t>(st.index));
    } else {
      h.mix(0xfeedface00000001ull);
    }

    h.mix(static_cast<uint64_t>(node.dtype().val()));
    h.mix(static_cast<uint64_t>(node.ndim()));
    h.mix(static_cast<uint64_t>(node.siblings().size()));

    // KV-cache slice views (4D Slice over a 4D SliceUpdate) grow axis 2 by +1
    // each decode token. Mask axis 2 to (len > 0) here and encode the exact
    // SDPA online-rope k_len regime on the consuming SDPA node below.
    bool mask_kv_seq_axis = false;
    if (ti != nullptr && *ti == typeid(Slice) && node.ndim() == 4 &&
        !node.inputs().empty()) {
      const uint32_t* src_pos = id_map.find(node.inputs()[0].id());
      if (src_pos != nullptr && (*src_pos & kExternalSlotBit) == 0 &&
          is_op(&tape[*src_pos], typeid(SliceUpdate)) &&
          tape[*src_pos].ndim() == 4) {
        mask_kv_seq_axis = true;
      }
    }
    h.mix(mask_kv_seq_axis ? 0x511ce001ull : 0x511ce000ull);
    for (int ax = 0; ax < node.ndim(); ++ax) {
      if (mask_kv_seq_axis && ax == 2) {
        h.mix(node.shape(2) > 0 ? 1ull : 0ull);
      } else {
        h.mix(static_cast<uint64_t>(node.shape(ax)));
      }
    }

    const auto& inputs = node.inputs();
    h.mix(static_cast<uint64_t>(inputs.size()));
    for (const auto& inp : inputs) {
      if (const uint32_t* slot = id_map.find(inp.id()); slot != nullptr) {
        h.mix(static_cast<uint64_t>(*slot));
      } else {
        const uint32_t ext_slot = kExternalSlotBit | next_ext++;
        id_map.put_if_absent(inp.id(), ext_slot);
        h.mix(static_cast<uint64_t>(ext_slot));
        h.mix(static_cast<uint64_t>(inp.dtype().val()));
        h.mix(static_cast<uint64_t>(inp.ndim()));
        for (int ax = 0; ax < inp.ndim(); ++ax) {
          h.mix(static_cast<uint64_t>(inp.shape(ax)));
        }
      }
    }

    if (ti == nullptr) {
      continue;
    }
    Primitive& p = node.primitive();
    if (*ti == typeid(QuantizedMatmul)) {
      auto [group_size, bits, mode, transpose] =
          static_cast<const QuantizedMatmul&>(p).state();
      h.mix(static_cast<uint64_t>(group_size));
      h.mix(static_cast<uint64_t>(bits));
      h.mix(static_cast<uint64_t>(static_cast<int>(mode)));
      h.mix(transpose ? 1ull : 0ull);
    } else if (*ti == typeid(fast::RMSNorm)) {
      h.mix_float(static_cast<const fast::RMSNorm&>(p).state().second);
    } else if (*ti == typeid(fast::RoPE)) {
      const auto& rp = static_cast<const fast::RoPE&>(p);
      const auto& st = rp.state();
      h.mix(rp.has_norm() ? 1ull : 0ull);
      if (rp.has_norm()) {
        h.mix_float(rp.norm_eps());
      }
      h.mix(static_cast<uint64_t>(std::get<1>(st)));
      h.mix(std::get<2>(st) ? 1ull : 0ull);
      h.mix_float(std::get<3>(st));
      h.mix_float(std::get<4>(st));
      h.mix(std::get<5>(st) ? 1ull : 0ull);
      h.mix((p.stream().device.type == mlx::core::Device::gpu &&
             fast::RoPE::use_fallback(p.stream()))
                ? 1ull
                : 0ull);
    } else if (*ti == typeid(Transpose)) {
      const auto& axes = static_cast<const Transpose&>(p).state();
      h.mix(static_cast<uint64_t>(axes.size()));
      for (int ax : axes) {
        h.mix(static_cast<uint64_t>(ax));
      }
    } else if (*ti == typeid(SliceUpdate)) {
      const auto& st = static_cast<const SliceUpdate&>(p).state();
      h.mix(static_cast<uint64_t>(static_cast<int>(std::get<0>(st))));
      if (!inputs.empty()) {
        const array& base = inputs[0];
        uint64_t bflags = (base.flags().row_contiguous ? 1ull : 0ull) |
            (base.size() == base.data_size() ? 2ull : 0ull) |
            ((base.itemsize() != 0 && base.offset() % base.itemsize() == 0)
                 ? 4ull
                 : 0ull);
        h.mix(bflags);
      }
    } else if (*ti == typeid(Gather)) {
      const auto& gst = static_cast<const Gather&>(p).state();
      h.mix(static_cast<uint64_t>(gst.first.size()));
      for (int ax : gst.first) {
        h.mix(static_cast<uint64_t>(ax));
      }
      h.mix(static_cast<uint64_t>(gst.second.size()));
      for (auto sz : gst.second) {
        h.mix(static_cast<uint64_t>(sz));
      }
    } else if (*ti == typeid(fast::Quantize)) {
      const auto& qst = static_cast<const fast::Quantize&>(p).state();
      h.mix(static_cast<uint64_t>(std::get<1>(qst)));
      h.mix(static_cast<uint64_t>(std::get<2>(qst)));
      h.mix(static_cast<uint64_t>(static_cast<int>(std::get<3>(qst))));
      h.mix(std::get<4>(qst) ? 1ull : 0ull);
    } else if (*ti == typeid(fast::GdnConvDeltaUpdate)) {
      h.mix(static_cast<uint64_t>(
          static_cast<const fast::GdnConvDeltaUpdate&>(p).key_dim()));
    } else if (*ti == typeid(fast::ScaledDotProductAttention)) {
      if (node.ndim() == 4 && inputs.size() == 3 &&
          (node.shape(3) == 128 || node.shape(3) == 256) &&
          inputs[1].ndim() == 4) {
        const int64_t k_len = inputs[1].shape(2);
        uint64_t regime = (k_len > 0) ? 1ull : 0ull;
        if (const Device* d = stream_device(node);
            d != nullptr && k_len > 0) {
          const auto& caps = d->capabilities();
          if (sdpa_online_rope_enabled(
                  caps, static_cast<uint32_t>(k_len), node.shape(3), false)) {
            regime |= 2ull;
          }
          if (sdpa_online_rope_enabled(
                  caps, static_cast<uint32_t>(k_len), node.shape(3), true)) {
            regime |= 4ull;
          }
        }
        h.mix(regime);
      }
    }
  }
  return FoldPlanKey{h.lo, h.hi, tape.size()};
}

// Tape-position reference to any array referenced by a cached fold plan:
// either a tape node itself (input_idx == kSelf) or an input slot of a tape
// node. Holds no array handles so cached templates never retain GPU buffers.
struct NodeRef {
  static constexpr uint32_t kSelf = std::numeric_limits<uint32_t>::max();
  uint32_t tape_idx{0};
  uint32_t input_idx{kSelf};

  const array& resolve(const std::deque<array>& tape) const {
    return input_idx == kSelf ? tape[tape_idx]
                              : tape[tape_idx].inputs()[input_idx];
  }
};

struct RoleTmpl {
  uint32_t node_idx{0};
  uint32_t group_ref{0};
  EagerStep step{EagerStep::skip};
};

struct GemvMemberTmpl {
  uint32_t node_idx{0};
  int32_t epilogue_idx{-1};
  std::optional<NodeRef> addend;
};

struct OutgateTmpl {
  NodeRef gate;
  NodeRef out;
  uint32_t x_idx{0};
  uint32_t sigmoid_idx{0};
};

struct ResNormTmpl {
  uint32_t norm_idx{0};
  NodeRef weight;
  float eps{0.0f};
  bool dead_qmm_out{false};
};

struct RopeEpiSideTmpl {
  uint32_t rope_idx{0};
  size_t member_index{0};
  std::vector<uint32_t> view_chain_indices;
  Shape view_shape;
  Strides view_strides;
  int64_t view_offset{0};
  size_t pair_index{0};
};

struct RopeEpiTmpl {
  RopeEpiSideTmpl q;
  RopeEpiSideTmpl k;
  int dims{0};
  bool traditional{false};
  float base{0.0f};
  float scale{1.0f};
  float norm_eps{0.0f};
};

struct GemvGroupTmpl {
  std::vector<GemvMemberTmpl> members;
  int32_t swiglu_out_idx{-1};
  std::vector<uint32_t> swiglu_role_indices;
  std::optional<OutgateTmpl> outgate;
  std::optional<ResNormTmpl> res_norm;
  std::optional<RopeEpiTmpl> rope_epi;
};

struct DenseGemvGroupTmpl {
  std::vector<uint32_t> node_indices;
  NodeRef input;
};

struct SliceUpdatePairTmpl {
  uint32_t first_idx{0};
  uint32_t second_idx{0};
  bool direct{false};
  uint8_t rope_side{0};
  DirectKind sum_kind{DirectKind::none};
  size_t sum_group_index{0};
  size_t sum_member_index{0};
  uint32_t rope_idx{0};
  std::vector<uint32_t> sum_view_chain_indices;
};

struct SdpaTransposeTmpl {
  uint32_t sdpa_idx{0};
  uint32_t reshape_idx{0};
  std::vector<uint32_t> chain_indices;
};

struct SdpaOutgateTmpl {
  uint32_t sdpa_idx{0};
  uint32_t tail_idx{0};
  NodeRef gate_src;
  std::vector<uint32_t> skip_indices;
};

struct SdpaOnlineRopeTmpl {
  uint32_t q_rope_idx{0};
  uint32_t k_rope_idx{0};
  uint32_t sdpa_idx{0};
  NodeRef q_raw;
  NodeRef k_raw;
  NodeRef offset;
  std::optional<NodeRef> q_weight;
  std::optional<NodeRef> k_weight;
  int32_t q_view_idx{-1};
  int32_t k_view_idx{-1};
  int q_norm_fold{-1};
  int k_norm_fold{-1};
  size_t pair_index{0};
  bool has_norm{false};
  float eps{0.0f};
  float base{0.0f};
  float scale{1.0f};
  int dims{0};
};

struct RopePairTmpl {
  uint32_t first_idx{0};
  uint32_t second_idx{0};
  std::array<int, 2> norm_folds{-1, -1};
};

struct AddNormPairTmpl {
  uint32_t add_idx{0};
  uint32_t norm_idx{0};
};

struct RopeNormTmpl {
  uint32_t norm_idx{0};
  uint32_t rope_idx{0};
  int32_t view_idx{-1};
  NodeRef x;
  NodeRef w;
  float eps{0.0f};
};

struct PrologueFoldTmpl {
  uint32_t quant_idx{0};
  NodeRef ids;
  NodeRef w;
  NodeRef scales;
  NodeRef biases;
  std::vector<uint32_t> replay_indices;
  int bits{0};
  int group_size{0};
  int64_t words_per_row{0};
  int64_t groups_per_row{0};
  int64_t nrows{0};
  int32_t norm_out_idx{-1};
  std::optional<NodeRef> norm_weight;
  float norm_eps{0.0f};
};

struct GdnInProjBaTmpl {
  uint32_t gdn_idx{0};
  uint32_t a_idx{0};
  uint32_t b_idx{0};
  NodeRef x;
  NodeRef w_a;
  NodeRef s_a;
  NodeRef b_a;
  NodeRef w_b;
  NodeRef s_b;
  NodeRef b_b;
  uint32_t k{0};
  uint32_t hv{0};
};

struct FoldPlanTemplate {
  std::vector<RoleTmpl> roles;
  std::vector<GemvGroupTmpl> gemv_groups;
  std::vector<DenseGemvGroupTmpl> dense_gemv_groups;
  std::vector<SliceUpdatePairTmpl> slice_update_pairs;
  std::vector<SdpaTransposeTmpl> sdpa_transpose_folds;
  std::vector<SdpaOutgateTmpl> sdpa_outgate_folds;
  std::vector<SdpaOnlineRopeTmpl> sdpa_online_rope_plans;
  std::vector<RopePairTmpl> rope_pairs;
  std::vector<AddNormPairTmpl> add_norm_pairs;
  std::vector<RopeNormTmpl> rope_norms;
  std::vector<PrologueFoldTmpl> prologue_folds;
  std::vector<GdnInProjBaTmpl> gdn_ba_folds;
};

struct FoldPlanCacheEntry {
  FoldPlanKey key;
  FoldPlanTemplate tmpl;
};

struct FoldPlanCacheState {
  static constexpr size_t kMaxEntries = 16;
  std::vector<FoldPlanCacheEntry> entries;
  uint64_t hits{0};
  uint64_t misses{0};
  uint64_t revalidation_fallbacks{0};
};

FoldPlanCacheState& fold_plan_cache_state() {
  static thread_local FoldPlanCacheState cache;
  return cache;
}

const FoldPlanTemplate* fold_plan_cache_lookup(const FoldPlanKey& key) {
  auto& cache = fold_plan_cache_state();
  for (size_t i = 0; i < cache.entries.size(); ++i) {
    if (cache.entries[i].key == key) {
      if (i > 0) {
        std::rotate(
            cache.entries.begin(),
            cache.entries.begin() + static_cast<ptrdiff_t>(i),
            cache.entries.begin() + static_cast<ptrdiff_t>(i + 1));
      }
      return &cache.entries[0].tmpl;
    }
  }
  return nullptr;
}

void fold_plan_cache_insert(const FoldPlanKey& key, FoldPlanTemplate&& tmpl) {
  auto& cache = fold_plan_cache_state();
  for (size_t i = 0; i < cache.entries.size(); ++i) {
    if (cache.entries[i].key == key) {
      cache.entries[i].tmpl = std::move(tmpl);
      if (i > 0) {
        std::rotate(
            cache.entries.begin(),
            cache.entries.begin() + static_cast<ptrdiff_t>(i),
            cache.entries.begin() + static_cast<ptrdiff_t>(i + 1));
      }
      return;
    }
  }
  if (cache.entries.size() >= FoldPlanCacheState::kMaxEntries) {
    cache.entries.pop_back();
  }
  cache.entries.insert(
      cache.entries.begin(), FoldPlanCacheEntry{key, std::move(tmpl)});
}

std::optional<uint32_t> find_tape_pos(
    std::uintptr_t id,
    const TapeIndex<uint32_t>& id_map) {
  const uint32_t* p = id_map.find(id);
  if (p == nullptr || (*p & kExternalSlotBit) != 0) {
    return std::nullopt;
  }
  return *p;
}

std::optional<NodeRef> find_node_ref(
    const array& target,
    uint32_t hint_idx,
    const std::deque<array>& tape,
    const TapeIndex<uint32_t>& id_map) {
  if (auto pos = find_tape_pos(target.id(), id_map)) {
    return NodeRef{*pos, NodeRef::kSelf};
  }
  const auto& inps = tape[hint_idx].inputs();
  for (uint32_t k = 0; k < static_cast<uint32_t>(inps.size()); ++k) {
    if (inps[k].id() == target.id()) {
      return NodeRef{hint_idx, k};
    }
  }
  // One-hop view unwrap (e.g. Broadcast(bias), Flatten(x), Reshape(gate_src)).
  for (uint32_t k = 0; k < static_cast<uint32_t>(inps.size()); ++k) {
    if (auto mid = find_tape_pos(inps[k].id(), id_map)) {
      const auto& minps = tape[*mid].inputs();
      for (uint32_t m = 0; m < static_cast<uint32_t>(minps.size()); ++m) {
        if (minps[m].id() == target.id()) {
          return NodeRef{*mid, m};
        }
      }
    }
  }
  return std::nullopt;
}

std::optional<FoldPlanTemplate> extract_fold_plan_template(
    const EagerFusionState& state,
    const std::deque<array>& tape,
    const TapeIndex<uint32_t>& id_map) {
  FoldPlanTemplate tmpl;
  const uint32_t n = static_cast<uint32_t>(tape.size());

  // 1. Roles (SwiGLU chain roles + Outgate/RopeEpi skip roles).
  tmpl.roles.reserve(state.roles.size());
  for (const auto& [id, role] : state.roles) {
    auto node_idx = find_tape_pos(id, id_map);
    if (!node_idx) {
      return std::nullopt;
    }
    uint32_t group_ref = 0;
    if (role.step == EagerStep::skip) {
      group_ref = static_cast<uint32_t>(role.group);
    } else {
      auto tail_idx = find_tape_pos(role.group, id_map);
      if (!tail_idx) {
        return std::nullopt;
      }
      group_ref = *tail_idx;
    }
    tmpl.roles.push_back(RoleTmpl{*node_idx, group_ref, role.step});
  }

  // 2. Quantized GEMV groups.
  tmpl.gemv_groups.reserve(state.gemv_groups.size());
  std::vector<std::vector<uint32_t>> extra_gemv_roles(state.gemv_groups.size());
  for (uint32_t i = 0; i < n; ++i) {
    if (const size_t* gi = state.gemv_roles.find(tape[i].id())) {
      const auto& g = state.gemv_groups[*gi];
      bool is_member_or_ep = false;
      for (const auto& m : g.members) {
        if (m.node.id() == tape[i].id() ||
            (m.epilogue && m.epilogue->id() == tape[i].id())) {
          is_member_or_ep = true;
          break;
        }
      }
      if (!is_member_or_ep) {
        extra_gemv_roles[*gi].push_back(i);
      }
    }
  }
  for (size_t gi = 0; gi < state.gemv_groups.size(); ++gi) {
    const auto& g = state.gemv_groups[gi];
    GemvGroupTmpl gt;
    gt.members.reserve(g.members.size());
    for (const auto& m : g.members) {
      auto node_idx = find_tape_pos(m.node.id(), id_map);
      if (!node_idx) {
        return std::nullopt;
      }
      GemvMemberTmpl mt;
      mt.node_idx = *node_idx;
      if (m.epilogue) {
        auto ep_idx = find_tape_pos(m.epilogue->id(), id_map);
        if (!ep_idx) {
          return std::nullopt;
        }
        mt.epilogue_idx = static_cast<int32_t>(*ep_idx);
        if (m.addend) {
          auto addend_ref = find_node_ref(*m.addend, *ep_idx, tape, id_map);
          if (!addend_ref) {
            return std::nullopt;
          }
          mt.addend = *addend_ref;
        }
      }
      gt.members.push_back(std::move(mt));
    }
    if (g.swiglu_out) {
      auto sw_idx = find_tape_pos(g.swiglu_out->id(), id_map);
      if (!sw_idx) {
        return std::nullopt;
      }
      gt.swiglu_out_idx = static_cast<int32_t>(*sw_idx);
      gt.swiglu_role_indices = std::move(extra_gemv_roles[gi]);
    }
    if (g.outgate) {
      auto x_idx = find_tape_pos(g.outgate->x.id(), id_map);
      auto sig_idx = find_tape_pos(g.outgate->sigmoid_id, id_map);
      if (!x_idx || !sig_idx) {
        return std::nullopt;
      }
      auto gate_ref = find_node_ref(g.outgate->gate, *sig_idx, tape, id_map);
      auto out_ref = find_node_ref(g.outgate->out, *x_idx, tape, id_map);
      if (!gate_ref || !out_ref) {
        return std::nullopt;
      }
      gt.outgate = OutgateTmpl{*gate_ref, *out_ref, *x_idx, *sig_idx};
    }
    if (g.res_norm) {
      auto norm_idx = find_tape_pos(g.res_norm->out.id(), id_map);
      if (!norm_idx) {
        return std::nullopt;
      }
      auto w_ref = find_node_ref(g.res_norm->weight, *norm_idx, tape, id_map);
      if (!w_ref) {
        return std::nullopt;
      }
      gt.res_norm =
          ResNormTmpl{*norm_idx, *w_ref, g.res_norm->eps, g.res_norm->dead_qmm_out};
    }
    if (g.rope_epi) {
      auto extract_side = [&](const RopeEpiSide& s) -> std::optional<RopeEpiSideTmpl> {
        auto r_idx = find_tape_pos(s.rope.id(), id_map);
        if (!r_idx) {
          return std::nullopt;
        }
        RopeEpiSideTmpl st;
        st.rope_idx = *r_idx;
        st.member_index = s.member_index;
        st.view_chain_indices.reserve(s.view_chain.size());
        for (const auto& v : s.view_chain) {
          auto v_idx = find_tape_pos(v.id(), id_map);
          if (!v_idx) {
            return std::nullopt;
          }
          st.view_chain_indices.push_back(*v_idx);
        }
        st.view_shape = s.view_shape;
        st.view_strides = s.view_strides;
        st.view_offset = s.view_offset;
        st.pair_index = s.pair_index;
        return st;
      };
      auto qs = extract_side(g.rope_epi->q);
      auto ks = extract_side(g.rope_epi->k);
      if (!qs || !ks) {
        return std::nullopt;
      }
      gt.rope_epi = RopeEpiTmpl{
          std::move(*qs),
          std::move(*ks),
          g.rope_epi->dims,
          g.rope_epi->traditional,
          g.rope_epi->base,
          g.rope_epi->scale,
          g.rope_epi->norm_eps};
    }
    tmpl.gemv_groups.push_back(std::move(gt));
  }

  // 3. Dense GEMV groups.
  tmpl.dense_gemv_groups.reserve(state.dense_gemv_groups.size());
  for (const auto& dg : state.dense_gemv_groups) {
    if (dg.nodes.empty() || !dg.input) {
      return std::nullopt;
    }
    DenseGemvGroupTmpl dgt;
    dgt.node_indices.reserve(dg.nodes.size());
    for (const auto& nd : dg.nodes) {
      auto idx = find_tape_pos(nd.id(), id_map);
      if (!idx) {
        return std::nullopt;
      }
      dgt.node_indices.push_back(*idx);
    }
    auto in_ref = find_node_ref(*dg.input, dgt.node_indices[0], tape, id_map);
    if (!in_ref) {
      return std::nullopt;
    }
    dgt.input = *in_ref;
    tmpl.dense_gemv_groups.push_back(std::move(dgt));
  }

  // 4. SliceUpdate pairs + producer-direct KV bindings.
  std::vector<std::vector<uint32_t>> views_by_pair(
      state.slice_update_pairs.size());
  for (uint32_t i = 0; i < n; ++i) {
    if (const size_t* pi = state.reshape_redirect_roles.find(tape[i].id())) {
      views_by_pair[*pi].push_back(i);
    }
  }
  tmpl.slice_update_pairs.reserve(state.slice_update_pairs.size());
  for (size_t pi = 0; pi < state.slice_update_pairs.size(); ++pi) {
    const auto& pair = state.slice_update_pairs[pi];
    auto first_idx = find_tape_pos(pair.nodes[0].id(), id_map);
    auto second_idx = find_tape_pos(pair.nodes[1].id(), id_map);
    if (!first_idx || !second_idx) {
      return std::nullopt;
    }
    SliceUpdatePairTmpl pt;
    pt.first_idx = *first_idx;
    pt.second_idx = *second_idx;
    pt.direct = pair.direct;
    if (pair.direct) {
      if (!pair.windows[0] || !pair.windows[1]) {
        return std::nullopt;
      }
      pt.rope_side =
          (pair.windows[0]->node.id() == pair.nodes[0].id()) ? 0 : 1;
      auto r_idx = find_tape_pos(
          pair.nodes[pt.rope_side].inputs()[1].id(), id_map);
      if (!r_idx) {
        return std::nullopt;
      }
      pt.rope_idx = *r_idx;
      pt.sum_view_chain_indices = std::move(views_by_pair[pi]);

      const std::uintptr_t sum_su_id = pair.windows[1]->node.id();
      bool found_sum = false;
      for (size_t gi = 0; gi < state.gemv_groups.size() && !found_sum; ++gi) {
        for (size_t mi = 0; mi < state.gemv_groups[gi].members.size(); ++mi) {
          const auto& sw = state.gemv_groups[gi].members[mi].sum_window;
          if (sw && sw->node.id() == sum_su_id) {
            pt.sum_kind = DirectKind::values_sum;
            pt.sum_group_index = gi;
            pt.sum_member_index = mi;
            found_sum = true;
            break;
          }
        }
      }
      for (size_t gi = 0; gi < state.dense_gemv_groups.size() && !found_sum;
           ++gi) {
        const auto& sws = state.dense_gemv_groups[gi].sum_windows;
        for (size_t mi = 0; mi < sws.size(); ++mi) {
          if (sws[mi] && sws[mi]->node.id() == sum_su_id) {
            pt.sum_kind = DirectKind::dense_values;
            pt.sum_group_index = gi;
            pt.sum_member_index = mi;
            found_sum = true;
            break;
          }
        }
      }
      if (!found_sum) {
        return std::nullopt;
      }
    }
    tmpl.slice_update_pairs.push_back(std::move(pt));
  }

  // 5. SDPA out-transpose folds.
  tmpl.sdpa_transpose_folds.resize(state.sdpa_transpose_folds.size());
  for (size_t fi = 0; fi < state.sdpa_transpose_folds.size(); ++fi) {
    const auto& f = state.sdpa_transpose_folds[fi];
    auto r_idx = find_tape_pos(f.reshape.id(), id_map);
    if (!r_idx) {
      return std::nullopt;
    }
    tmpl.sdpa_transpose_folds[fi].reshape_idx = *r_idx;
    tmpl.sdpa_transpose_folds[fi].chain_indices.reserve(f.chain.size());
    for (const auto& c : f.chain) {
      auto c_idx = find_tape_pos(c.id(), id_map);
      if (!c_idx) {
        return std::nullopt;
      }
      tmpl.sdpa_transpose_folds[fi].chain_indices.push_back(*c_idx);
    }
  }
  for (const auto& [id, fi] : state.sdpa_transpose_plans) {
    auto s_idx = find_tape_pos(id, id_map);
    if (!s_idx) {
      return std::nullopt;
    }
    tmpl.sdpa_transpose_folds[fi].sdpa_idx = *s_idx;
  }

  // 6. SDPA out-gate folds.
  tmpl.sdpa_outgate_folds.resize(state.sdpa_outgate_folds.size());
  for (const auto& [id, fi] : state.sdpa_outgate_plans) {
    auto s_idx = find_tape_pos(id, id_map);
    if (!s_idx) {
      return std::nullopt;
    }
    tmpl.sdpa_outgate_folds[fi].sdpa_idx = *s_idx;
  }
  for (const auto& [id, fi] : state.sdpa_outgate_skip_roles) {
    auto sk_idx = find_tape_pos(id, id_map);
    if (!sk_idx) {
      return std::nullopt;
    }
    tmpl.sdpa_outgate_folds[fi].skip_indices.push_back(*sk_idx);
  }
  for (size_t fi = 0; fi < state.sdpa_outgate_folds.size(); ++fi) {
    const auto& f = state.sdpa_outgate_folds[fi];
    auto tail_idx = find_tape_pos(f.plan.tail.id(), id_map);
    if (!tail_idx) {
      return std::nullopt;
    }
    tmpl.sdpa_outgate_folds[fi].tail_idx = *tail_idx;
    uint32_t sig_hint = *tail_idx;
    for (uint32_t si : tmpl.sdpa_outgate_folds[fi].skip_indices) {
      if (is_op(&tape[si], typeid(Sigmoid))) {
        sig_hint = si;
        break;
      }
    }
    auto g_ref = find_node_ref(f.plan.gate_src, sig_hint, tape, id_map);
    if (!g_ref) {
      return std::nullopt;
    }
    tmpl.sdpa_outgate_folds[fi].gate_src = *g_ref;
  }

  // 7. Add -> RMSNorm pairs.
  tmpl.add_norm_pairs.reserve(state.add_norm_pairs.size());
  for (const auto& p : state.add_norm_pairs) {
    auto a_idx = find_tape_pos(p.nodes[0].id(), id_map);
    auto n_idx = find_tape_pos(p.nodes[1].id(), id_map);
    if (!a_idx || !n_idx) {
      return std::nullopt;
    }
    tmpl.add_norm_pairs.push_back(AddNormPairTmpl{*a_idx, *n_idx});
  }

  // 8. RMSNorm -> RoPE folds.
  tmpl.rope_norms.reserve(state.rope_norms.size());
  for (const auto& f : state.rope_norms) {
    auto n_idx = find_tape_pos(f.norm.id(), id_map);
    auto r_idx = find_tape_pos(f.rope.id(), id_map);
    if (!n_idx || !r_idx) {
      return std::nullopt;
    }
    int32_t view_idx = -1;
    if (f.view) {
      auto v_idx = find_tape_pos(f.view->id(), id_map);
      if (!v_idx) {
        return std::nullopt;
      }
      view_idx = static_cast<int32_t>(*v_idx);
    }
    auto x_ref = find_node_ref(f.x, *n_idx, tape, id_map);
    auto w_ref = find_node_ref(f.w, *n_idx, tape, id_map);
    if (!x_ref || !w_ref) {
      return std::nullopt;
    }
    tmpl.rope_norms.push_back(
        RopeNormTmpl{*n_idx, *r_idx, view_idx, *x_ref, *w_ref, f.eps});
  }

  // 9. SDPA online-rope plans.
  tmpl.sdpa_online_rope_plans.reserve(state.sdpa_online_rope_plans.size());
  for (const auto& p : state.sdpa_online_rope_plans) {
    auto qr_idx = find_tape_pos(p.q_rope.id(), id_map);
    auto kr_idx = find_tape_pos(p.k_rope.id(), id_map);
    auto s_idx = find_tape_pos(p.sdpa.id(), id_map);
    if (!qr_idx || !kr_idx || !s_idx) {
      return std::nullopt;
    }
    const uint32_t q_hint = p.q_norm_fold >= 0
        ? tmpl.rope_norms[p.q_norm_fold].norm_idx
        : *qr_idx;
    const uint32_t k_hint = p.k_norm_fold >= 0
        ? tmpl.rope_norms[p.k_norm_fold].norm_idx
        : *kr_idx;
    auto q_raw_ref = find_node_ref(p.q_raw, q_hint, tape, id_map);
    auto k_raw_ref = find_node_ref(p.k_raw, k_hint, tape, id_map);
    auto off_ref = find_node_ref(p.offset, *qr_idx, tape, id_map);
    if (!q_raw_ref || !k_raw_ref || !off_ref) {
      return std::nullopt;
    }
    std::optional<NodeRef> qw_ref;
    std::optional<NodeRef> kw_ref;
    if (p.q_weight) {
      qw_ref = find_node_ref(*p.q_weight, q_hint, tape, id_map);
      if (!qw_ref) {
        return std::nullopt;
      }
    }
    if (p.k_weight) {
      kw_ref = find_node_ref(*p.k_weight, k_hint, tape, id_map);
      if (!kw_ref) {
        return std::nullopt;
      }
    }
    int32_t qv_idx = -1;
    int32_t kv_idx = -1;
    if (p.q_view) {
      auto v = find_tape_pos(p.q_view->id(), id_map);
      if (!v) {
        return std::nullopt;
      }
      qv_idx = static_cast<int32_t>(*v);
    }
    if (p.k_view) {
      auto v = find_tape_pos(p.k_view->id(), id_map);
      if (!v) {
        return std::nullopt;
      }
      kv_idx = static_cast<int32_t>(*v);
    }
    tmpl.sdpa_online_rope_plans.push_back(SdpaOnlineRopeTmpl{
        *qr_idx,
        *kr_idx,
        *s_idx,
        *q_raw_ref,
        *k_raw_ref,
        *off_ref,
        qw_ref,
        kw_ref,
        qv_idx,
        kv_idx,
        p.q_norm_fold,
        p.k_norm_fold,
        p.pair_index,
        p.has_norm,
        p.eps,
        p.base,
        p.scale,
        p.dims});
  }

  // 10. RoPE pairs.
  tmpl.rope_pairs.reserve(state.rope_pairs.size());
  for (const auto& p : state.rope_pairs) {
    auto f_idx = find_tape_pos(p.nodes[0].id(), id_map);
    auto s_idx = find_tape_pos(p.nodes[1].id(), id_map);
    if (!f_idx || !s_idx) {
      return std::nullopt;
    }
    tmpl.rope_pairs.push_back(RopePairTmpl{*f_idx, *s_idx, p.norm_folds});
  }

  // 11. Quantized-embedding prologue folds.
  tmpl.prologue_folds.reserve(state.prologue_folds.size());
  for (const auto& f : state.prologue_folds) {
    auto q_idx = find_tape_pos(f.quant.id(), id_map);
    if (!q_idx || f.replay.empty()) {
      return std::nullopt;
    }
    PrologueFoldTmpl pt;
    pt.quant_idx = *q_idx;
    pt.replay_indices.reserve(f.replay.size());
    std::vector<uint32_t> gather_indices;
    for (const auto& step : f.replay) {
      auto s_idx = find_tape_pos(step.node.id(), id_map);
      if (!s_idx) {
        return std::nullopt;
      }
      pt.replay_indices.push_back(*s_idx);
      if (is_op(&tape[*s_idx], typeid(Gather))) {
        gather_indices.push_back(*s_idx);
      }
    }
    if (gather_indices.size() != 3) {
      return std::nullopt;
    }
    auto ids_ref = find_node_ref(f.ids, gather_indices[0], tape, id_map);
    auto w_ref = find_node_ref(f.w, gather_indices[0], tape, id_map);
    auto s_ref = find_node_ref(f.scales, gather_indices[1], tape, id_map);
    auto b_ref = find_node_ref(f.biases, gather_indices[2], tape, id_map);
    if (!ids_ref || !w_ref || !s_ref || !b_ref) {
      return std::nullopt;
    }
    pt.ids = *ids_ref;
    pt.w = *w_ref;
    pt.scales = *s_ref;
    pt.biases = *b_ref;
    pt.bits = f.bits;
    pt.group_size = f.group_size;
    pt.words_per_row = f.words_per_row;
    pt.groups_per_row = f.groups_per_row;
    pt.nrows = f.nrows;
    if (f.norm_out && f.norm_weight) {
      auto no_idx = find_tape_pos(f.norm_out->id(), id_map);
      if (!no_idx) {
        return std::nullopt;
      }
      auto nw_ref = find_node_ref(*f.norm_weight, *no_idx, tape, id_map);
      if (!nw_ref) {
        return std::nullopt;
      }
      pt.norm_out_idx = static_cast<int32_t>(*no_idx);
      pt.norm_weight = *nw_ref;
      pt.norm_eps = f.norm_eps;
    }
    tmpl.prologue_folds.push_back(std::move(pt));
  }

  // 12. GDN in_proj_a/b prologue folds.
  if (!state.gdn_ba_folds.empty()) {
    std::vector<int32_t> gdn_idx_by_fold(state.gdn_ba_folds.size(), -1);
    for (uint32_t i = 0; i < n; ++i) {
      if (is_op(&tape[i], typeid(fast::GdnConvDeltaUpdate))) {
        if (const size_t* fi = state.gdn_ba_roles.find(tape[i].id())) {
          gdn_idx_by_fold[*fi] = static_cast<int32_t>(i);
        }
      }
    }
    tmpl.gdn_ba_folds.reserve(state.gdn_ba_folds.size());
    for (size_t fi = 0; fi < state.gdn_ba_folds.size(); ++fi) {
      const auto& f = state.gdn_ba_folds[fi];
      auto a_idx = find_tape_pos(f.a_node.id(), id_map);
      auto b_idx = find_tape_pos(f.b_node.id(), id_map);
      if (gdn_idx_by_fold[fi] < 0 || !a_idx || !b_idx) {
        return std::nullopt;
      }
      auto x_ref = find_node_ref(f.plan.x, *a_idx, tape, id_map);
      auto wa_ref = find_node_ref(f.plan.w_a, *a_idx, tape, id_map);
      auto sa_ref = find_node_ref(f.plan.s_a, *a_idx, tape, id_map);
      auto ba_ref = find_node_ref(f.plan.b_a, *a_idx, tape, id_map);
      auto wb_ref = find_node_ref(f.plan.w_b, *b_idx, tape, id_map);
      auto sb_ref = find_node_ref(f.plan.s_b, *b_idx, tape, id_map);
      auto bb_ref = find_node_ref(f.plan.b_b, *b_idx, tape, id_map);
      if (!x_ref || !wa_ref || !sa_ref || !ba_ref || !wb_ref || !sb_ref ||
          !bb_ref) {
        return std::nullopt;
      }
      tmpl.gdn_ba_folds.push_back(GdnInProjBaTmpl{
          static_cast<uint32_t>(gdn_idx_by_fold[fi]),
          *a_idx,
          *b_idx,
          *x_ref,
          *wa_ref,
          *sa_ref,
          *ba_ref,
          *wb_ref,
          *sb_ref,
          *bb_ref,
          f.plan.k,
          f.plan.hv});
    }
  }

  return tmpl;
}

bool instantiate_fold_plan(
    const FoldPlanTemplate& tmpl,
    const std::deque<array>& tape,
    EagerFusionState& state) {
  // 1. Quantized GEMV groups.
  state.gemv_groups.reserve(tmpl.gemv_groups.size());
  for (size_t gi = 0; gi < tmpl.gemv_groups.size(); ++gi) {
    const auto& gt = tmpl.gemv_groups[gi];
    GemvGroup group;
    group.members.reserve(gt.members.size());
    for (const auto& mt : gt.members) {
      const array& node = tape[mt.node_idx];
      GemvFusionMember member{node, std::nullopt, std::nullopt};
      state.gemv_roles.put_if_absent(node.id(), gi);
      if (mt.epilogue_idx >= 0) {
        const array& ep = tape[mt.epilogue_idx];
        member.epilogue = ep;
        state.gemv_roles.put_if_absent(ep.id(), gi);
        if (mt.addend) {
          member.addend = mt.addend->resolve(tape);
        }
      }
      group.members.push_back(std::move(member));
    }
    if (gt.swiglu_out_idx >= 0) {
      group.swiglu_out = tape[gt.swiglu_out_idx];
      for (uint32_t ri : gt.swiglu_role_indices) {
        state.gemv_roles.put_if_absent(tape[ri].id(), gi);
      }
    }
    if (gt.outgate) {
      const array& x = tape[gt.outgate->x_idx];
      const array& sig = tape[gt.outgate->sigmoid_idx];
      group.outgate = OutgatePlan{
          gt.outgate->gate.resolve(tape),
          gt.outgate->out.resolve(tape),
          x,
          sig.id()};
    }
    if (gt.res_norm) {
      const array& norm = tape[gt.res_norm->norm_idx];
      group.res_norm = ResNormPlan{
          norm,
          gt.res_norm->weight.resolve(tape),
          gt.res_norm->eps,
          gt.res_norm->dead_qmm_out};
      state.res_norm_skip_roles.put_if_absent(norm.id(), gi);
    }
    if (gt.rope_epi) {
      auto build_side = [&](const RopeEpiSideTmpl& st) {
        std::vector<array> vchain;
        vchain.reserve(st.view_chain_indices.size());
        for (uint32_t vi : st.view_chain_indices) {
          vchain.push_back(tape[vi]);
        }
        return RopeEpiSide{
            tape[st.rope_idx],
            st.member_index,
            std::move(vchain),
            st.view_shape,
            st.view_strides,
            st.view_offset,
            std::nullopt,
            st.pair_index};
      };
      group.rope_epi = RopeEpiPlan{
          build_side(gt.rope_epi->q),
          build_side(gt.rope_epi->k),
          gt.rope_epi->dims,
          gt.rope_epi->traditional,
          gt.rope_epi->base,
          gt.rope_epi->scale,
          gt.rope_epi->norm_eps};
    }
    state.gemv_groups.push_back(std::move(group));
  }

  // 2. Dense GEMV groups.
  state.dense_gemv_groups.reserve(tmpl.dense_gemv_groups.size());
  for (size_t gi = 0; gi < tmpl.dense_gemv_groups.size(); ++gi) {
    const auto& dgt = tmpl.dense_gemv_groups[gi];
    DenseGemvGroup group;
    group.input = dgt.input.resolve(tape);
    group.nodes.reserve(dgt.node_indices.size());
    for (uint32_t ni : dgt.node_indices) {
      const array& nd = tape[ni];
      group.nodes.push_back(nd);
      state.dense_gemv_roles.put_if_absent(nd.id(), gi);
    }
    state.dense_gemv_groups.push_back(std::move(group));
  }

  // 3. SliceUpdate pairs: revalidate runtime state match and recompute fresh
  // KV direct-write window offsets and buffer donation flags for this token.
  state.slice_update_pairs.reserve(tmpl.slice_update_pairs.size());
  for (size_t pi = 0; pi < tmpl.slice_update_pairs.size(); ++pi) {
    const auto& pt = tmpl.slice_update_pairs[pi];
    const array& first = tape[pt.first_idx];
    const array& second = tape[pt.second_idx];
    const auto& st0 = static_cast<const SliceUpdate&>(first.primitive()).state();
    const auto& st1 =
        static_cast<const SliceUpdate&>(second.primitive()).state();
    if (st0 != st1 || std::get<0>(st0) != SliceUpdate::None) {
      return false;
    }
    state.slice_update_roles.put_if_absent(first.id(), pi);
    state.slice_update_roles.put_if_absent(second.id(), pi);
    state.slice_update_pairs.emplace_back(first, second);
    if (pt.direct) {
      auto& pair = state.slice_update_pairs.back();
      const array& keys_member = pt.rope_side == 0 ? first : second;
      const array& vals_member = pt.rope_side == 0 ? second : first;
      auto keys_geom = direct_window_geometry(keys_member);
      auto vals_geom = direct_window_geometry(vals_member);
      if (!keys_geom || !vals_geom) {
        return false;
      }
      const array& k_upd = keys_member.inputs()[1];
      uint32_t matrix_stride = 0;
      if (k_upd.ndim() == 4) {
        uint64_t heads = static_cast<uint64_t>(k_upd.shape(1));
        if (heads == 0 ||
            static_cast<uint64_t>(keys_geom->strides[0]) !=
                heads * static_cast<uint64_t>(keys_geom->strides[1])) {
          return false;
        }
        matrix_stride = keys_geom->strides[1];
      } else {
        matrix_stride = keys_geom->strides[0];
      }
      const array& v_upd = vals_member.inputs()[1];
      if (v_upd.ndim() != 4 || v_upd.shape(3) <= 0) {
        return false;
      }
      const uint32_t head_dim = static_cast<uint32_t>(v_upd.shape(3));
      // Query donation BEFORE copying base handles into KvDirectWindow.
      const bool keys_in_place = kv_window_in_place(keys_member);
      const bool vals_in_place = kv_window_in_place(vals_member);
      KvDirectWindow keys_win{
          keys_member,
          keys_member.inputs()[0],
          keys_geom->offset,
          {matrix_stride, keys_geom->strides[k_upd.ndim() - 2], 1, 0},
          keys_geom->ndim,
          /*row_gap=*/0,
          /*head_dim=*/0,
          keys_in_place};
      KvDirectWindow vals_win{
          vals_member,
          vals_member.inputs()[0],
          vals_geom->offset,
          {vals_geom->strides[0],
           vals_geom->strides[1],
           vals_geom->strides[2],
           vals_geom->strides[3]},
          vals_geom->ndim,
          /*row_gap=*/vals_geom->strides[1],
          /*head_dim=*/head_dim,
          vals_in_place};
      if (pt.sum_kind == DirectKind::dense_values) {
        auto& windows = state.dense_gemv_groups[pt.sum_group_index].sum_windows;
        if (windows.size() <
            state.dense_gemv_groups[pt.sum_group_index].nodes.size()) {
          windows.resize(
              state.dense_gemv_groups[pt.sum_group_index].nodes.size());
        }
        windows[pt.sum_member_index] = vals_win;
      } else {
        state.gemv_groups[pt.sum_group_index]
            .members[pt.sum_member_index]
            .sum_window = vals_win;
      }
      pair.windows[0] = std::move(keys_win);
      pair.windows[1] = std::move(vals_win);
      state.rope_redirect_roles.put_if_absent(tape[pt.rope_idx].id(), pi);
      for (uint32_t vi : pt.sum_view_chain_indices) {
        state.reshape_redirect_roles.put_if_absent(tape[vi].id(), pi);
      }
      pair.direct = true;
    }
  }

  // Attach freshly computed keys windows to any RopeEpiPlan.
  for (auto& group : state.gemv_groups) {
    if (group.rope_epi) {
      const auto& pair = state.slice_update_pairs[group.rope_epi->k.pair_index];
      if (!pair.direct || !pair.windows[0]) {
        return false;
      }
      group.rope_epi->k.window = pair.windows[0];
    }
  }

  // 4. Roles (SwiGLU chain roles + Outgate/RopeEpi skip roles).
  for (const auto& rt : tmpl.roles) {
    const std::uintptr_t group_val = (rt.step == EagerStep::skip)
        ? static_cast<std::uintptr_t>(rt.group_ref)
        : tape[rt.group_ref].id();
    state.roles.emplace(tape[rt.node_idx].id(), EagerRole{group_val, rt.step});
  }

  // 5. SDPA out-gate folds.
  state.sdpa_outgate_folds.reserve(tmpl.sdpa_outgate_folds.size());
  for (size_t fi = 0; fi < tmpl.sdpa_outgate_folds.size(); ++fi) {
    const auto& ft = tmpl.sdpa_outgate_folds[fi];
    state.sdpa_outgate_folds.push_back(SdpaOutgateFold{
        SdpaOutgatePlan{tape[ft.tail_idx], ft.gate_src.resolve(tape)}, false});
    state.sdpa_outgate_plans.emplace(tape[ft.sdpa_idx].id(), fi);
    for (uint32_t si : ft.skip_indices) {
      state.sdpa_outgate_skip_roles.emplace(tape[si].id(), fi);
    }
  }

  // 6. SDPA out-transpose folds.
  state.sdpa_transpose_folds.reserve(tmpl.sdpa_transpose_folds.size());
  for (size_t fi = 0; fi < tmpl.sdpa_transpose_folds.size(); ++fi) {
    const auto& ft = tmpl.sdpa_transpose_folds[fi];
    SdpaTransposeFold fold{tape[ft.reshape_idx], {}, false};
    fold.chain.reserve(ft.chain_indices.size());
    for (uint32_t ci : ft.chain_indices) {
      fold.chain.push_back(tape[ci]);
      state.sdpa_transpose_skip_roles.emplace(tape[ci].id(), fi);
    }
    state.sdpa_transpose_plans.emplace(tape[ft.sdpa_idx].id(), fi);
    state.sdpa_transpose_skip_roles.emplace(tape[ft.reshape_idx].id(), fi);
    state.sdpa_transpose_folds.push_back(std::move(fold));
  }

  // 7. Add -> RMSNorm pairs.
  state.add_norm_pairs.reserve(tmpl.add_norm_pairs.size());
  for (size_t pi = 0; pi < tmpl.add_norm_pairs.size(); ++pi) {
    const auto& pt = tmpl.add_norm_pairs[pi];
    const array& add = tape[pt.add_idx];
    const array& norm = tape[pt.norm_idx];
    state.add_norm_roles.put_if_absent(add.id(), pi);
    state.add_norm_roles.put_if_absent(norm.id(), pi);
    state.add_norm_pairs.push_back(AddNormPair{{add, norm}});
  }

  // 8. RMSNorm -> RoPE folds (revalidate view contiguity/offset).
  state.rope_norms.reserve(tmpl.rope_norms.size());
  for (size_t fi = 0; fi < tmpl.rope_norms.size(); ++fi) {
    const auto& ft = tmpl.rope_norms[fi];
    const array& n = tape[ft.norm_idx];
    const array& rope = tape[ft.rope_idx];
    const array& x = ft.x.resolve(tape);
    const array& w = ft.w.resolve(tape);
    std::optional<array> view_opt;
    std::shared_ptr<Primitive> view_prim;
    if (ft.view_idx >= 0) {
      const array& view = tape[ft.view_idx];
      if (!x.flags().row_contiguous ||
          (view.itemsize() != 0 && view.offset() % view.itemsize() != 0) ||
          view.shape(-1) != x.shape(-1)) {
        return false;
      }
      view_opt = view;
      view_prim = view.primitive_ptr();
      state.rope_norm_skip_roles.put_if_absent(view.id(), fi);
    }
    state.rope_norm_skip_roles.put_if_absent(n.id(), fi);
    state.rope_norm_roles.put_if_absent(rope.id(), fi);
    state.rope_norms.push_back(RopeNormFold{
        n,
        rope,
        std::move(view_opt),
        x,
        w,
        n.primitive_ptr(),
        std::move(view_prim),
        ft.eps});
  }

  // 9. SDPA online-rope plans (revalidate shared Q/K RoPE offsets).
  state.sdpa_online_rope_plans.reserve(tmpl.sdpa_online_rope_plans.size());
  for (size_t pi = 0; pi < tmpl.sdpa_online_rope_plans.size(); ++pi) {
    const auto& pt = tmpl.sdpa_online_rope_plans[pi];
    const array& q_rope = tape[pt.q_rope_idx];
    const array& k_rope = tape[pt.k_rope_idx];
    const array& sdpa = tape[pt.sdpa_idx];
    if (!rope_offset_shared(q_rope.inputs()[1], k_rope.inputs()[1])) {
      return false;
    }
    const auto& pair = state.slice_update_pairs[pt.pair_index];
    if (!pair.direct || !pair.windows[0] || !pair.windows[1]) {
      return false;
    }
    std::optional<array> q_weight;
    std::optional<array> k_weight;
    if (pt.q_weight) {
      q_weight = pt.q_weight->resolve(tape);
    }
    if (pt.k_weight) {
      k_weight = pt.k_weight->resolve(tape);
    }
    std::optional<array> q_view;
    std::optional<array> k_view;
    std::shared_ptr<Primitive> q_view_prim;
    std::shared_ptr<Primitive> k_view_prim;
    if (pt.q_view_idx >= 0) {
      q_view = tape[pt.q_view_idx];
      q_view_prim = tape[pt.q_view_idx].primitive_ptr();
    }
    if (pt.k_view_idx >= 0) {
      k_view = tape[pt.k_view_idx];
      k_view_prim = tape[pt.k_view_idx].primitive_ptr();
    }
    state.sdpa_online_rope_plans.push_back(SdpaOnlineRopePlan{
        q_rope,
        k_rope,
        sdpa,
        pt.q_raw.resolve(tape),
        pt.k_raw.resolve(tape),
        pt.offset.resolve(tape),
        std::move(q_weight),
        std::move(k_weight),
        std::move(q_view),
        std::move(k_view),
        std::move(q_view_prim),
        std::move(k_view_prim),
        q_rope.primitive_ptr(),
        k_rope.primitive_ptr(),
        q_rope.inputs(),
        k_rope.inputs(),
        pt.q_norm_fold,
        pt.k_norm_fold,
        pt.pair_index,
        pt.has_norm,
        pt.eps,
        pt.base,
        pt.scale,
        pt.dims,
        SdpaOnlineRopeCtx{},
        false,
        false,
        SdpaOnlineRopePlan::State::pending});
    state.sdpa_online_rope_skip_roles.emplace(q_rope.id(), pi);
    state.sdpa_online_rope_skip_roles.emplace(k_rope.id(), pi);
    state.sdpa_online_rope_su_roles.emplace(pair.nodes[0].id(), pi);
    state.sdpa_online_rope_su_roles.emplace(pair.nodes[1].id(), pi);
    state.sdpa_online_rope_roles.emplace(sdpa.id(), pi);
  }

  // 10. RoPE pairs (revalidate shared scalar offsets).
  state.rope_pairs.reserve(tmpl.rope_pairs.size());
  for (size_t pi = 0; pi < tmpl.rope_pairs.size(); ++pi) {
    const auto& pt = tmpl.rope_pairs[pi];
    const array& first = tape[pt.first_idx];
    const array& second = tape[pt.second_idx];
    if (!rope_offset_shared(first.inputs()[1], second.inputs()[1])) {
      return false;
    }
    state.rope_pair_roles.put_if_absent(first.id(), pi);
    state.rope_pair_roles.put_if_absent(second.id(), pi);
    RopePair pair{{first, second}};
    pair.norm_folds = pt.norm_folds;
    state.rope_pairs.push_back(std::move(pair));
  }

  // 11. Quantized-embedding prologue folds.
  state.prologue_folds.reserve(tmpl.prologue_folds.size());
  for (size_t fi = 0; fi < tmpl.prologue_folds.size(); ++fi) {
    const auto& pt = tmpl.prologue_folds[fi];
    const array& quant = tape[pt.quant_idx];
    std::vector<PrologueFold::ReplayStep> replay;
    replay.reserve(pt.replay_indices.size());
    for (uint32_t ri : pt.replay_indices) {
      const array& step_node = tape[ri];
      replay.push_back(PrologueFold::ReplayStep{
          step_node, step_node.primitive_ptr(), step_node.inputs()});
      state.prologue_skip_roles.put_if_absent(step_node.id(), fi);
    }
    state.prologue_fold_roles.put_if_absent(quant.id(), fi);
    std::optional<array> norm_out;
    std::optional<array> norm_weight;
    if (pt.norm_out_idx >= 0 && pt.norm_weight) {
      const array& no = tape[pt.norm_out_idx];
      norm_out = no;
      norm_weight = pt.norm_weight->resolve(tape);
      state.prologue_norm_skip_roles.put_if_absent(no.id(), fi);
    }
    state.prologue_folds.push_back(PrologueFold{
        quant,
        pt.ids.resolve(tape),
        pt.w.resolve(tape),
        pt.scales.resolve(tape),
        pt.biases.resolve(tape),
        std::move(replay),
        pt.bits,
        pt.group_size,
        pt.words_per_row,
        pt.groups_per_row,
        pt.nrows,
        std::move(norm_out),
        std::move(norm_weight),
        pt.norm_eps,
        false,
        PrologueFold::State::pending});
  }

  // 12. GDN in_proj_a/b prologue folds.
  state.gdn_ba_folds.reserve(tmpl.gdn_ba_folds.size());
  for (size_t fi = 0; fi < tmpl.gdn_ba_folds.size(); ++fi) {
    const auto& ft = tmpl.gdn_ba_folds[fi];
    const array& gdn = tape[ft.gdn_idx];
    const array& a_node = tape[ft.a_idx];
    const array& b_node = tape[ft.b_idx];
    state.gdn_ba_skip_roles.put_if_absent(a_node.id(), fi);
    state.gdn_ba_skip_roles.put_if_absent(b_node.id(), fi);
    state.gdn_ba_roles.put_if_absent(gdn.id(), fi);
    for (const auto& sib : gdn.siblings()) {
      state.gdn_ba_roles.put_if_absent(sib.id(), fi);
    }
    state.gdn_ba_folds.push_back(GdnInProjBaFold{
        GdnInProjBaPlan{
            ft.x.resolve(tape),
            ft.w_a.resolve(tape),
            ft.s_a.resolve(tape),
            ft.b_a.resolve(tape),
            ft.w_b.resolve(tape),
            ft.s_b.resolve(tape),
            ft.b_b.resolve(tape),
            ft.k,
            ft.hv},
        a_node,
        a_node.primitive_ptr(),
        a_node.inputs(),
        b_node,
        b_node.primitive_ptr(),
        b_node.inputs(),
        GdnInProjBaFold::State::pending});
  }

  return true;
}

void print_fold_timing(
    size_t tape_size,
    std::chrono::steady_clock::time_point plan_t0,
    const EagerFusionState& state,
    const char* cache_status) {
  const double plan_us =
      std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - plan_t0)
          .count();
  std::fprintf(
      stderr,
      "[fold-time] plan tape=%zu us=%.0f gemv=%zu dense=%zu swiglu-roles=%zu "
      "rope-pairs=%zu rope-norms=%zu prologues=%zu slice-pairs=%zu "
      "sdpa-folds=%zu rope-epis=%zu cache=%s\n",
      tape_size,
      plan_us,
      state.gemv_groups.size(),
      state.dense_gemv_groups.size(),
      state.roles.size(),
      state.rope_pairs.size(),
      state.rope_norms.size(),
      state.prologue_folds.size(),
      state.slice_update_pairs.size(),
      state.sdpa_transpose_folds.size(),
      std::count_if(
          state.gemv_groups.begin(),
          state.gemv_groups.end(),
          [](const auto& g) { return g.rope_epi.has_value(); }),
      cache_status);
}

} // namespace

bool fold_plan_cache_enabled() {
  const char* env = omarchy::env_str("MLX_OMARCHY_FOLD_PLAN_CACHE");
  return env == nullptr || env[0] == '\0' || std::strcmp(env, "0") != 0;
}

FoldPlanKey compute_fold_plan_key(const std::deque<array>& tape) {
  env_note_eval();
  static thread_local TapeIndex<uint32_t> standalone_id_map;
  return compute_fold_plan_key_impl(tape, standalone_id_map);
}

FoldPlanCacheStats fold_plan_cache_stats() {
  const auto& c = fold_plan_cache_state();
  return FoldPlanCacheStats{
      c.hits, c.misses, c.revalidation_fallbacks, c.entries.size()};
}

void clear_fold_plan_cache() {
  auto& c = fold_plan_cache_state();
  c.entries.clear();
  c.hits = 0;
  c.misses = 0;
  c.revalidation_fallbacks = 0;
}

double* eval_phase_slots() {
  static thread_local double slots[8];
  return slots;
}

EagerFusionScope::EagerFusionScope(const std::deque<array>& tape)
    : previous_(eager_state), t0_(std::chrono::steady_clock::now()) {
  eager_state = nullptr;
  env_note_eval();
  if (!fused_chain_enabled()) {
    return;
  }
  static const bool fold_timing =
      std::getenv("MLX_OMARCHY_FOLD_TIMING") != nullptr;
  const auto plan_t0 = std::chrono::steady_clock::now();

  // Prescan gate: every pass below needs an anchor primitive
  // (QuantizedMatmul or Matmul for GEMV groups, SliceUpdate for cache
  // pairs, Sigmoid for swiglu/out-gate plans, fast::RoPE for trio
  // plans, fast::Quantize for embedding prologue folds). Tapes with none
  // of these can fire no plan, so skip the node/use maps and all pattern
  // passes. Set MLX_OMARCHY_NO_FUSION_GATE to restore the unconditional scan.
  if (omarchy::env_str("MLX_OMARCHY_NO_FUSION_GATE") == nullptr) {
    bool anchor = false;
    for (const auto& node : tape) {
      if (is_op(&node, typeid(QuantizedMatmul)) ||
          is_op(&node, typeid(Matmul)) ||
          is_op(&node, typeid(SliceUpdate)) ||
          is_op(&node, typeid(Sigmoid)) ||
          is_op(&node, typeid(fast::RoPE)) ||
          is_op(&node, typeid(fast::Quantize))) {
        anchor = true;
        break;
      }
    }
    if (!anchor) {
      return;
    }
  }

  EagerFusionState* state;
  auto& pool = fusion_state_pool();
  if (pool.empty()) {
    state = new EagerFusionState;
  } else {
    state = pool.back().release();
    pool.pop_back();
  }
  eager_state = state;
  state->prepare(tape.size());

  const bool use_cache =
      fold_plan_cache_enabled() && !fold_planner_trace_active();
  static thread_local TapeIndex<uint32_t> key_id_map;
  FoldPlanKey plan_key{};
  const char* cache_status = "off";
  if (use_cache) {
    plan_key = compute_fold_plan_key_impl(tape, key_id_map);
    if (const FoldPlanTemplate* tmpl = fold_plan_cache_lookup(plan_key)) {
      if (instantiate_fold_plan(*tmpl, tape, *state)) {
        ++fold_plan_cache_state().hits;
        cache_status = "hit";
        if (fold_timing) {
          print_fold_timing(tape.size(), plan_t0, *state, cache_status);
        }
        return;
      }
      ++fold_plan_cache_state().revalidation_fallbacks;
      state->reset();
      state->prepare(tape.size());
    }
    ++fold_plan_cache_state().misses;
    cache_status = "miss";
  }
  // Tape-index scratch is thread-local and reused across passes. The maps
  // are only live for this planning section, so reuse is safe under nested
  // scopes: a nested scope's eval runs after this constructor returns,
  // never while it still holds the tables.
  static thread_local TapeIndex<const array*> nodes;
  static thread_local TapeIndex<size_t> uses;
  // consumer_of[input] = last tape node reading input; paired with the
  // uses count it is the single-consumer map without a second tape pass.
  static thread_local TapeIndex<const array*> consumer_of;
  static thread_local TapeIndex<std::array<std::uintptr_t, 2>> first_two_readers;
  static thread_local TapeIndex<char> claimed;
  nodes.prepare(tape.size());
  uses.prepare(tape.size() * 2);
  consumer_of.prepare(tape.size() * 2);
  first_two_readers.prepare(tape.size() * 2);
  claimed.prepare(tape.size());
  for (const auto& node : tape) {
    nodes.put_if_absent(node.id(), &node);
    for (const auto& input : node.inputs()) {
      size_t& u = uses[input.id()];
      if (u < 2) {
        first_two_readers[input.id()][u] = node.id();
      }
      ++u;
      consumer_of[input.id()] = &node;
    }
  }
  auto lookup = [&](const array& ref) -> const array* {
    auto* it = nodes.find(ref.id());
    return it == nullptr ? nullptr : *it;
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
  const bool swiglu_scan_trace =
      std::getenv("MLX_OMARCHY_SWIGLU_TRACE") != nullptr;
  for (const auto& tail : tape) {
    if (!is_op(&tail, typeid(Multiply)) || tail.inputs().size() != 2 ||
        claimed.count(tail.id())) {
      continue;
    }
    const array* left = lookup(tail.inputs()[0]);
    const array* right = lookup(tail.inputs()[1]);
    const array* inner = is_op(left, typeid(Multiply)) ? left :
        (is_op(right, typeid(Multiply)) ? right : nullptr);
    if (swiglu_scan_trace) {
      const char* ln = left && left->has_primitive()
          ? left->primitive().name()
          : "<none>";
      const char* rn = right && right->has_primitive()
          ? right->primitive().name()
          : "<none>";
      std::fprintf(
          stderr,
          "[swiglu-scan] mul in0=%s in1=%s inner=%s uses_i=%zu\n",
          ln,
          rn,
          inner ? "mul" : "none",
          inner ? uses[inner->id()] : 0ul);
    }
    if (!inner || inner->inputs().size() != 2 || uses[inner->id()] != 1) {
      continue;
    }
    const array* inner_left = lookup(inner->inputs()[0]);
    const array* inner_right = lookup(inner->inputs()[1]);
    const array* sigmoid = is_op(inner_left, typeid(Sigmoid)) ? inner_left :
        (is_op(inner_right, typeid(Sigmoid)) ? inner_right : nullptr);
    if (swiglu_scan_trace) {
      const char* iln = inner_left && inner_left->has_primitive()
          ? inner_left->primitive().name()
          : "<none>";
      const char* irn = inner_right && inner_right->has_primitive()
          ? inner_right->primitive().name()
          : "<none>";
      std::fprintf(
          stderr,
          "[swiglu-scan]   inner in0=%s in1=%s sig=%s uses_s=%zu\n",
          iln,
          irn,
          sigmoid ? "sig" : "none",
          sigmoid ? uses[sigmoid->id()] : 0ul);
    }
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
    if (up != nullptr && stream_device(tail) != nullptr &&
        fused_gemv_swiglu_enabled(*stream_device(tail)) &&
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
    claimed.put_if_absent(sigmoid->id(), 1);
    claimed.put_if_absent(inner->id(), 1);
    claimed.put_if_absent(tail.id(), 1);
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
  const char* outgate_raw = omarchy::env_str("MLX_OMARCHY_OUTGATE_FOLD");
  const bool outgate_explicit_on =
      outgate_raw != nullptr && outgate_raw[0] != '\0' &&
      !(outgate_raw[0] == '0' && outgate_raw[1] == '\0');
  const bool outgate_on = outgate_fold_enabled();
  for (const auto& tail : tape) {
    if (!outgate_on || !is_op(&tail, typeid(Multiply)) ||
        tail.inputs().size() != 2 ||
        claimed.count(tail.id()) || uses[tail.id()] != 1 ||
        tail.dtype() == float32) {
      continue;
    }
    if (!outgate_explicit_on) {
      if (const Device* td = stream_device(tail);
          td != nullptr && sdpa_outgate_enabled(td->capabilities())) {
        continue;
      }
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
    claimed.put_if_absent(sigmoid->id(), 1);
    claimed.put_if_absent(tail.id(), 1);
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
    state->slice_update_roles.put_if_absent(first.id(), index);
    state->slice_update_roles.put_if_absent(second.id(), index);
    state->slice_update_pairs.emplace_back(first, second);
    claimed.put_if_absent(first.id(), 1);
    claimed.put_if_absent(second.id(), 1);
    ++i;
  }

  // GDN decode in_proj_a/b prologue fold (MLX_OMARCHY_GDN_INPROJ_BA_FOLD=1):
  // claim single-consumer Q4/g64 affine QuantizedMatmul nodes producing a and b
  // for fast::GdnConvDeltaUpdate before GEMV grouping when the coalesced Sg32
  // kernel is eligible, so the tiny [Hv, Hv] in_proj_a/b dispatches fold into
  // idle Phase-1 subgroups 12/13 (zero extra barriers, zero extra shared mem)
  // and leave [in_proj_qkv, in_proj_z] as a clean balanced 2-member pair.
  if (gdn_inproj_ba_fold_enabled()) {
    const char* gdn_coal_env = omarchy::env_str("MLX_OMARCHY_GDN_COALESCED");
    const bool gdn_coal_set =
        gdn_coal_env != nullptr && gdn_coal_env[0] != '\0';
    constexpr VkSubgroupFeatureFlags kGdnSgFeatures =
        VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
        VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
        VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
    for (const auto& gdn : tape) {
      if (!is_op(&gdn, typeid(fast::GdnConvDeltaUpdate)) ||
          gdn.inputs().size() != 10 || claimed.count(gdn.id())) {
        continue;
      }
      const auto& stream = gdn.primitive().stream();
      if (stream.device.type != mlx::core::Device::gpu) {
        continue;
      }
      const auto& gdn_in_x = gdn.inputs()[1];
      const auto& gdn_in_w = gdn.inputs()[2];
      const auto& gdn_in_a = gdn.inputs()[3];
      if (gdn_in_x.ndim() != 3 || gdn_in_x.shape(0) != 1 ||
          gdn_in_w.ndim() < 2 || gdn_in_w.shape(1) != 4 ||
          gdn_in_a.ndim() != 3) {
        continue;
      }
      const auto& gdn_prim =
          static_cast<const fast::GdnConvDeltaUpdate&>(gdn.primitive());
      const int gdn_hv = gdn_in_a.shape(2);
      const int gdn_hk = gdn_prim.key_dim() / 128;
      const bool a_log_f32 = gdn.inputs()[5].dtype() == float32;
      auto& dev = device(stream.device.index);
      const auto& caps = dev.capabilities();
      const bool gdn_coal_want = gdn_coal_set
          ? (std::strcmp(gdn_coal_env, "0") != 0)
          : (a_log_f32 ? omarchy::gdn_conv_coalesced_f32a_default(caps.device_name)
                       : omarchy::gdn_conv_coalesced_default(
                             caps.device_name, gdn_hk, gdn_hv));
      if (!gdn_coal_want || caps.subgroup_size != 32u ||
          (caps.subgroup_operations & kGdnSgFeatures) != kGdnSgFeatures ||
          caps.max_compute_work_group_invocations < 1024u ||
          caps.max_compute_work_group_size[0] < 1024u ||
          dev.compute().binding_limit() < 18u) {
        continue;
      }
      const array* a_node = lookup(gdn.inputs()[3]);
      const array* b_node = lookup(gdn.inputs()[4]);
      if (a_node == nullptr || b_node == nullptr || a_node == b_node ||
          !is_op(a_node, typeid(QuantizedMatmul)) ||
          !is_op(b_node, typeid(QuantizedMatmul)) ||
          a_node->inputs().size() != 4 || b_node->inputs().size() != 4 ||
          claimed.count(a_node->id()) || claimed.count(b_node->id()) ||
          uses[a_node->id()] != 1 || uses[b_node->id()] != 1 ||
          a_node->dtype() != bfloat16 || b_node->dtype() != bfloat16 ||
          a_node->primitive().stream() != stream ||
          b_node->primitive().stream() != stream) {
        continue;
      }
      auto [a_gs, a_bits, a_mode, a_tr] =
          static_cast<const QuantizedMatmul&>(a_node->primitive()).state();
      auto [b_gs, b_bits, b_mode, b_tr] =
          static_cast<const QuantizedMatmul&>(b_node->primitive()).state();
      if (a_mode != QuantizationMode::Affine || !a_tr || a_bits != 4 ||
          a_gs != 64 || b_mode != QuantizationMode::Affine || !b_tr ||
          b_bits != 4 || b_gs != 64) {
        continue;
      }
      const array& x = a_node->inputs()[0];
      if (b_node->inputs()[0].id() != x.id() || x.dtype() != bfloat16 ||
          x.ndim() < 2 || x.shape(-2) != 1 ||
          x.size() != static_cast<size_t>(x.shape(-1))) {
        continue;
      }
      const int64_t k = x.shape(-1);
      const array& w_a = a_node->inputs()[1];
      const array& s_a = a_node->inputs()[2];
      const array& bi_a = a_node->inputs()[3];
      const array& w_b = b_node->inputs()[1];
      const array& s_b = b_node->inputs()[2];
      const array& bi_b = b_node->inputs()[3];
      if (lookup(w_a) || lookup(s_a) || lookup(bi_a) || lookup(w_b) ||
          lookup(s_b) || lookup(bi_b) ||
          w_a.ndim() != 2 || w_b.ndim() != 2 ||
          s_a.ndim() != 2 || bi_a.ndim() != 2 ||
          s_b.ndim() != 2 || bi_b.ndim() != 2 ||
          w_a.dtype() != uint32 || w_b.dtype() != uint32) {
        continue;
      }
      const int64_t hv = w_a.shape(0);
      if (k <= 0 || (k % 512) != 0 || hv <= 0 || (hv % 8) != 0 ||
          hv != gdn_hv || w_b.shape(0) != hv || w_a.shape(1) != k / 8 ||
          w_b.shape(1) != k / 8 || s_a.shape(0) != hv ||
          s_a.shape(1) != k / 64 || bi_a.shape(0) != hv ||
          bi_a.shape(1) != k / 64 || s_b.shape(0) != hv ||
          s_b.shape(1) != k / 64 || bi_b.shape(0) != hv ||
          bi_b.shape(1) != k / 64 || s_a.dtype() != bfloat16 ||
          bi_a.dtype() != bfloat16 || s_b.dtype() != bfloat16 ||
          bi_b.dtype() != bfloat16 ||
          a_node->size() != static_cast<size_t>(hv) ||
          b_node->size() != static_cast<size_t>(hv)) {
        continue;
      }
      const size_t idx = state->gdn_ba_folds.size();
      state->gdn_ba_skip_roles.put_if_absent(a_node->id(), idx);
      state->gdn_ba_skip_roles.put_if_absent(b_node->id(), idx);
      state->gdn_ba_roles.put_if_absent(gdn.id(), idx);
      for (const auto& sib : gdn.siblings()) {
        state->gdn_ba_roles.put_if_absent(sib.id(), idx);
      }
      claimed.put_if_absent(a_node->id(), 1);
      claimed.put_if_absent(b_node->id(), 1);
      state->gdn_ba_folds.push_back(GdnInProjBaFold{
          GdnInProjBaPlan{
              x, w_a, s_a, bi_a, w_b, s_b, bi_b,
              static_cast<uint32_t>(k), static_cast<uint32_t>(hv)},
          *a_node, a_node->primitive_ptr(), a_node->inputs(),
          *b_node, b_node->primitive_ptr(), b_node->inputs(),
          GdnInProjBaFold::State::pending});
    }
  }

  // Decode GEMV groups (per node: MLX_OMARCHY_FUSED_GEMV for the device
  // that executes the node; the trio plan below does not depend on it). Candidates: affine transposed 4-bit/group-64
  // QuantizedMatmul nodes whose x is a single row and whose weight,
  // scale, and bias inputs were evaluated before this eval began (the
  // group dispatches when its FIRST member evaluates, so a later
  // member's inputs must already be readable). Nodes sharing one x form
  // groups of up to kQmmVecMultiWeights in tape order. A member's Add
  // epilogue is the Add that is the node's only consumer, same dtype
  // and size; its other operand may be any array that is ready when the
  // group dispatches (checked then, refused otherwise). A group of one
  // member without an epilogue gains nothing and is not planned.
  static thread_local TapeIndex<std::vector<const array*>> by_x;
  by_x.prepare(tape.size());
  std::vector<std::uintptr_t> x_order;
  for (const auto& node : tape) {
    if (!is_op(&node, typeid(QuantizedMatmul)) ||
        node.inputs().size() != 4 || claimed.count(node.id()) ||
        !(node_gemv_enabled(node) || node_epilogue_gemv_enabled(node))) {
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
        x.shape(-2) > 8 ||
        x.size() !=
            static_cast<size_t>(x.shape(-2)) *
                static_cast<size_t>(x.shape(-1))) {
      continue;
    }
    if (x.shape(-2) > 1 && !gemv_token_multi_enabled(*stream_device(node))) {
      continue;
    }
    // The multi-token Add epilogue fold has one deterministic divergence
    // on the AGX subgroup column (k=896 n=512 tokens=16, one element);
    // until root-caused the token route plans WITHOUT epilogues and the
    // residual/bias adds ride their separate eager dispatches. The fold
    // stays reachable for bisection via MLX_OMARCHY_QMM_VEC_TOKEN_FOLD=1.
    auto& group_list = by_x[x.id()];
    if (group_list.empty()) {
      x_order.push_back(x.id());
    }
    group_list.push_back(&node);
  }
  for (auto x_id : x_order) {
    const auto& nodes = *by_x.find(x_id);
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
    // The multi-token Add epilogue fold has one deterministic divergence
    // on the AGX subgroup column (k=896 n=512 tokens=16, one element);
    // until root-caused the token route plans WITHOUT epilogues and the
    // residual/bias adds ride their separate eager dispatches. The fold
    // stays reachable for bisection via MLX_OMARCHY_QMM_VEC_TOKEN_FOLD=1.
    const char* token_fold_env =
        x_rows > 1 ? omarchy::env_str("MLX_OMARCHY_QMM_VEC_TOKEN_FOLD") : nullptr;
    const bool fold_ok =
        x_rows == 1 ||
        (token_fold_env != nullptr && std::strcmp(token_fold_env, "0") != 0);
    for (size_t start = 0; start < nodes.size();
         start += kQmmVecMultiWeights) {
      GemvGroup group;
      bool worth = x_has_plan && start == 0;
      for (size_t i = start; i < nodes.size() && i < start + kQmmVecMultiWeights;
           ++i) {
        const array& node = *nodes[i];
        GemvFusionMember member{node, std::nullopt, std::nullopt};
        if (auto* uses_p = uses.find(node.id());
            uses_p != nullptr && *uses_p == 1) {
          if (auto* consumer = consumer_of.find(node.id())) {
          const array* add = *consumer;
          if (fold_ok && is_op(add, typeid(Add)) &&
              add->inputs().size() == 2 &&
              add->dtype() == node.dtype() &&
              add->size() == node.size() &&
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
                other_node->inputs()[0].size() *
                        static_cast<size_t>(x_rows) ==
                    node.size() &&
                other_node->size() == node.size() &&
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
        }
        worth = worth || member.epilogue.has_value();
        group.members.push_back(std::move(member));
      }
      // Epilogue-only route (device.h gemv_epilogue_only_part): general
      // multi-member folds and epilogue-free singles keep their
      // per-node path; single-member Add-epilogue groups plan, plus
      // balanced bf16 groups on G14C unless MLX_OMARCHY_FUSED_GEMV is
      // explicitly set to 0: balanced 3-member QKV, and balanced
      // 2-member pairs (MLP gate/up, and the two pairs a 4-member GDN
      // in_proj window [6144, 2048, 16, 16] splits into).
      bool balanced_qkv = false;
      std::vector<GemvGroup> pair_groups;
      if (!node_gemv_enabled(*nodes[start]) && x_rows == 1 &&
          group.members.size() >= 2) {
        bool epilogue_free = true;
        bool all_bf16 = true;
        for (const auto& m : group.members) {
          epilogue_free = epilogue_free && !m.epilogue;
          all_bf16 = all_bf16 && m.node.dtype() == bfloat16;
        }
        const Device* d = stream_device(*nodes[start]);
        if (d != nullptr && epilogue_free && all_bf16 &&
            gemv_balanced_qkv_route_enabled(d->capabilities().device_name)) {
          std::vector<int64_t> widths;
          widths.reserve(group.members.size());
          for (const auto& m : group.members) {
            widths.push_back(m.node.inputs()[1].shape(0));
          }
          balanced_qkv = group.members.size() == 3 &&
              gemv_balanced_qkv_widths_ok(widths);
          if (!balanced_qkv) {
            // Split-pair admission: each consecutive member pair whose
            // widths sit within a 4:1 ratio plans as its own dispatch,
            // so an asymmetric window never folds as one group while
            // its balanced pairs still fuse (members outside a passing
            // pair stay on the per-node path).
            for (size_t p = 0; p + 1 < group.members.size(); p += 2) {
              if (!gemv_balanced_pair_widths_ok(widths[p], widths[p + 1])) {
                continue;
              }
              GemvGroup pair;
              pair.members.push_back(std::move(group.members[p]));
              pair.members.push_back(std::move(group.members[p + 1]));
              pair_groups.push_back(std::move(pair));
            }
          }
        }
      }
      if (!node_gemv_enabled(*nodes[start]) && !balanced_qkv &&
          pair_groups.empty() &&
          (group.members.size() != 1 || !group.members[0].epilogue)) {
        continue;
      }
      if (!pair_groups.empty()) {
        for (auto& pair : pair_groups) {
          const size_t pair_index = state->gemv_groups.size();
          for (const auto& member : pair.members) {
            state->gemv_roles.put_if_absent(member.node.id(), pair_index);
            claimed.put_if_absent(member.node.id(), 1);
          }
          state->gemv_groups.push_back(std::move(pair));
        }
        continue;
      }
      if (group.members.size() < 2 && !worth) {
        continue;
      }
      if (worth && start == 0 && x_has_plan) {
        group.outgate = std::move(plan_it->second);
      }
      size_t index = state->gemv_groups.size();
      for (const auto& member : group.members) {
        state->gemv_roles.put_if_absent(member.node.id(), index);
        claimed.put_if_absent(member.node.id(), 1);
        if (member.epilogue) {
          state->gemv_roles.put_if_absent(member.epilogue->id(), index);
          claimed.put_if_absent(member.epilogue->id(), 1);
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
    if (!streams_ok || uses[plan.gate.id()] != 1 ||
        uses[plan.out.id()] != 1) {
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
    // The group dispatch refuses every group while the Q4 word path is off
    // (MLX_OMARCHY_QMM_VEC_Q4_WORD=0); adopting the fold then would leave the
    // deleted product unmaterialized.
    if (plan.x.dtype() != bfloat16 || !subgroup_ready ||
        !qmm_vec_q4_word_enabled() ||
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
  if (std::getenv("MLX_OMARCHY_SWIGLU_TRACE") != nullptr) {
    std::fprintf(
        stderr,
        "[swiglu-attach] plans=%zu groups=%zu\n",
        swiglu_plans.size(),
        state->gemv_groups.size());
  }
  if (!swiglu_plans.empty()) {
    // O(1) reader verification using first_two_readers recorded during the
    // initial tape pass (eliminates the prior O(G * P * N) full-tape scans).
    auto only_one_reader = [&](std::uintptr_t leaf, std::uintptr_t want) {
      auto* count = uses.find(leaf);
      if (count == nullptr || *count != 1) {
        return false;
      }
      auto* readers = first_two_readers.find(leaf);
      return readers != nullptr && (*readers)[0] == want;
    };
    auto only_two_readers =
        [&](std::uintptr_t leaf, std::uintptr_t w0, std::uintptr_t w1) {
          auto* count = uses.find(leaf);
          if (count == nullptr || *count != 2) {
            return false;
          }
          auto* readers = first_two_readers.find(leaf);
          if (readers == nullptr) {
            return false;
          }
          const std::uintptr_t r0 = (*readers)[0];
          const std::uintptr_t r1 = (*readers)[1];
          return (r0 == w0 && r1 == w1) || (r0 == w1 && r1 == w0);
        };
    const bool swiglu_trace =
        std::getenv("MLX_OMARCHY_SWIGLU_TRACE") != nullptr;
    for (size_t gi = 0; gi < state->gemv_groups.size(); ++gi) {
      auto& group = state->gemv_groups[gi];
      if (swiglu_trace) {
        std::fprintf(
            stderr,
            "[swiglu-attach] group %zu members=%zu ep0=%d ep1=%d f32=%d\n",
            gi,
            group.members.size(),
            static_cast<int>(group.members[0].epilogue.has_value()),
            group.members.size() > 1
                ? static_cast<int>(group.members[1].epilogue.has_value())
                : -1,
            static_cast<int>(
                group.members[0].node.dtype() == float32));
      }
      if (group.members.size() != 2 || group.members[0].epilogue ||
          group.members[1].epilogue ||
          group.members[0].node.dtype() == float32) {
        continue;
      }
      const std::uintptr_t gate_id = group.members[0].node.id();
      const std::uintptr_t up_id = group.members[1].node.id();
      if (group.members[0].node.inputs()[1].shape(0) !=
          group.members[1].node.inputs()[1].shape(0)) {
        if (swiglu_trace) {
          std::fprintf(stderr, "[swiglu-attach] group %zu n-mismatch\n", gi);
        }
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
        if (!straight && !swapped) {
          continue;
        }
        bool gate_ok = only_two_readers(
            plan.gate_id, plan.sigmoid_id, plan.inner_id);
        bool up_ok = only_one_reader(plan.up_id, plan.out.id());
        if (swiglu_trace) {
          std::fprintf(
              stderr,
              "[swiglu-attach] group %zu id-match straight=%d swapped=%d "
              "gate_readers=%d up_readers=%d\n",
              gi,
              static_cast<int>(straight),
              static_cast<int>(swapped),
              static_cast<int>(gate_ok),
              static_cast<int>(up_ok));
        }
        if (!gate_ok || !up_ok) {
          continue;
        }
        if (swapped) {
          std::swap(group.members[0], group.members[1]);
        }
        group.swiglu_out = plan.out;
        for (std::uintptr_t id :
             {plan.sigmoid_id, plan.inner_id, plan.out.id()}) {
          state->roles.erase(id);
          state->gemv_roles.put_if_absent(id, gi);
        }
        break;
      }
    }
  }
  static thread_local TapeIndex<std::vector<const array*>> dense_by_x;
  dense_by_x.prepare(tape.size());
  std::vector<std::uintptr_t> dense_x_order;
  for (const auto& node : tape) {
    if (!is_op(&node, typeid(Matmul)) || node.inputs().size() != 2 ||
        node.dtype() != bfloat16 || claimed.count(node.id()) ||
        !node_gemv_enabled(node)) {
      continue;
    }
    const array& x = node.inputs()[0];
    if (x.ndim() < 2 || x.shape(-2) != 1 ||
        x.size() != static_cast<size_t>(x.shape(-1))) {
      continue;
    }
    const array& source = dense_gemv_source(x);
    auto& dense_list = dense_by_x[source.id()];
    if (dense_list.empty()) {
      dense_x_order.push_back(source.id());
    }
    dense_list.push_back(&node);
  }
  for (auto x_id : dense_x_order) {
    const auto& nodes = *dense_by_x.find(x_id);
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
        state->dense_gemv_roles.put_if_absent(node.id(), index);
        claimed.put_if_absent(node.id(), 1);
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
  // well. The values side is a GEMV group, so there is nothing to plan
  // without one.
  if (kv_direct_enabled() &&
      (!state->gemv_groups.empty() || !state->dense_gemv_groups.empty())) {
    for (size_t index = 0; index < state->slice_update_pairs.size();
         ++index) {
      auto& pair = state->slice_update_pairs[index];
      DirectPlan plans[2];
      bool classifiable = true;
      for (int side = 0; side < 2 && classifiable; ++side) {
        const array* update = lookup(pair.nodes[side].inputs()[1]);
        const size_t* use_p = update ? uses.find(update->id()) : nullptr;
        if (!update || use_p == nullptr || *use_p != 1) {
          classifiable = false;
          break;
        }
        plans[side] = plan_keys_window(pair.nodes[side], update);
        if (plans[side].kind == DirectKind::none) {
          plans[side] = plan_values_window(
              pair.nodes[side],
              update,
              uses,
              state->gemv_groups,
              state->dense_gemv_roles,
              state->dense_gemv_groups);
        }
      }
      if (!classifiable || plans[0].kind == DirectKind::none ||
          plans[1].kind == DirectKind::none ||
          plans[0].kind == plans[1].kind) {
        static int kv_trace = 0;
        if (omarchy::env_str("MLX_OMARCHY_KV_TRACE") != nullptr && kv_trace < 6) {
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
      state->rope_redirect_roles.put_if_absent(
          pair.nodes[rope_side].inputs()[1].id(), index);
      for (const auto& view : sum_plan.view_chain) {
        state->reshape_redirect_roles.put_if_absent(view.id(), index);
      }
      pair.direct = true;
    }
  }
  // Backward epilogue fold (MLX_OMARCHY_QMM_RESIDUAL_NORM=1): a single-member
  // decode Qmm + ResidualAdd group whose materialized sum feeds a single-row
  // bf16 fast::RMSNorm runs that norm in the last finishing workgroup without
  // an inter-dispatch barrier. Keeps Sum0 materialized so other consumers of
  // the residual sum (e.g., skip connections) read the identical bits.
  if (!state->gemv_groups.empty()) {
    std::unordered_map<std::uintptr_t, const array*> res_norms_by_in;
    for (const auto& node : tape) {
      if (!is_op(&node, typeid(fast::RMSNorm)) || node.inputs().size() != 2 ||
          claimed.count(node.id()) || node.dtype() != bfloat16 ||
          lookup(node.inputs()[1]) != nullptr) {
        continue;
      }
      res_norms_by_in.try_emplace(node.inputs()[0].id(), &node);
    }
    if (!res_norms_by_in.empty()) {
      for (size_t gi = 0; gi < state->gemv_groups.size(); ++gi) {
        auto& group = state->gemv_groups[gi];
        if (group.members.size() != 1 || group.swiglu_out ||
            !group.members[0].epilogue || !group.members[0].addend ||
            group.members[0].sum_window) {
          continue;
        }
        const array& qmm = group.members[0].node;
        const array& add = *group.members[0].epilogue;
        // Rows 1 folds on the single-row resnorm kernel; rows 2..4 on
        // the token4+resnorm variant (dispatch re-verifies the full
        // contract and falls back eager when it cannot run).
        if (qmm.dtype() != bfloat16 || qmm.inputs()[0].ndim() < 2 ||
            qmm.inputs()[0].shape(-2) > 4) {
          continue;
        }
        auto it = res_norms_by_in.find(add.id());
        if (it == res_norms_by_in.end()) {
          continue;
        }
        const array& norm = *it->second;
        const array& nw = norm.inputs()[1];
        const Device* d = stream_device(qmm);
        if (claimed.count(norm.id()) ||
            norm.primitive().stream() != qmm.primitive().stream() ||
            norm.ndim() < 1 || norm.size() != add.size() ||
            norm.shape(-1) != qmm.inputs()[1].shape(0) ||
            d == nullptr || !qmm_residual_norm_enabled(
                d->hardware_capabilities(), norm.shape(-1), group.outgate.has_value()) ||
            nw.dtype() != bfloat16 ||
            (nw.size() != 1 &&
             nw.size() != static_cast<size_t>(norm.shape(-1)))) {
          continue;
        }
        const float eps =
            static_cast<const fast::RMSNorm&>(norm.primitive()).state().second;
        auto* qmm_u = uses.find(qmm.id());
        auto* qmm_r = first_two_readers.find(qmm.id());
        const bool dead_qmm_out = (qmm_u != nullptr && *qmm_u == 1u &&
            qmm_r != nullptr && (*qmm_r)[0] == add.id());
        group.res_norm = ResNormPlan{norm, nw, eps, dead_qmm_out};
        state->res_norm_skip_roles.put_if_absent(norm.id(), gi);
        claimed.put_if_absent(norm.id(), 1);
      }
    }
  }
  // Fused RMSNorm+RoPE q/k epilogue fold (MLX_OMARCHY_ROPE_EPI): a
  // decode group's member outputs feed the two norm-fused fast::RoPE
  // nodes (query, keys) of one attention head set through
  // single-consumer view chains. The group's Token4 dispatch runs
  // both norms and rotations in its last finishing workgroup: the
  // query rows store into the rope node's own buffer, the keys rows
  // store through the slice pair's producer-direct window. The rope
  // nodes and their view chains are deleted, so every contract item
  // is decided here; a group of at most three members keeps the six
  // epilogue bindings inside the free weight slots.
  if (rope_epi_enabled() && !state->gemv_groups.empty()) {
    auto use_count = [&](const array& value) {
      auto* it = uses.find(value.id());
      return it == nullptr ? size_t{0} : *it;
    };
    // member node id -> (group index, member index); only groups that
    // can host the epilogue are indexed.
    std::unordered_map<std::uintptr_t, std::pair<size_t, size_t>>
        member_of;
    for (size_t gi = 0; gi < state->gemv_groups.size(); ++gi) {
      auto& group = state->gemv_groups[gi];
      if (group.members.size() > 3 || group.swiglu_out || group.outgate ||
          group.res_norm) {
        continue;
      }
      for (size_t mi = 0; mi < group.members.size(); ++mi) {
        member_of.try_emplace(
            group.members[mi].node.id(), std::make_pair(gi, mi));
      }
    }
    // group index -> rope candidates; each entry carries the node and
    // the resolved view chain + member index.
    struct RopeCand {
      const array* node;
      std::vector<array> chain;
      size_t member_index;
    };
    // Compose the chain's logical strides over the member storage. A
    // lazily-created view node's desc holds contiguous strides until
    // its own eval writes the real view strides via
    // copy_shared_buffer — and the skip-role'd chain never evals, so
    // the desc at plan/dispatch time cannot be trusted. Composing
    // Reshape (contiguous strides for the new shape) and Transpose
    // (strides permuted by axes) from the member's contiguous rows
    // yields the view geometry the epilogue must read.
    const auto compose_view = [](const array& member,
                                 const std::vector<array>& chain) {
      Shape shape(member.shape());
      Strides strides(shape.size(), 1);
      for (int d = static_cast<int>(shape.size()) - 2; d >= 0; --d) {
        strides[d] = strides[d + 1] * shape[d + 1];
      }
      for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        const auto& prim = it->primitive();
        if (typeid(prim) == typeid(Transpose)) {
          const auto& axes = static_cast<const Transpose&>(prim).state();
          Shape tshape(axes.size());
          Strides tstrides(axes.size());
          for (size_t i = 0; i < axes.size(); ++i) {
            tshape[i] = shape[axes[i]];
            tstrides[i] = strides[axes[i]];
          }
          shape = std::move(tshape);
          strides = std::move(tstrides);
        } else {
          shape = Shape(it->shape());
          strides = Strides(shape.size(), 1);
          for (int d = static_cast<int>(shape.size()) - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * shape[d + 1];
          }
        }
      }
      return std::make_pair(std::move(shape), std::move(strides));
    };
    std::unordered_map<size_t, std::vector<RopeCand>> cands;
    const bool epi_trace =
        std::getenv("MLX_OMARCHY_ROPE_EPI_TRACE") != nullptr;
    int tr_rope = 0, tr_shape = 0, tr_member = 0;
    for (const auto& node : tape) {
      if (!is_op(&node, typeid(fast::RoPE)) || claimed.count(node.id()) ||
          node.dtype() != bfloat16) {
        continue;
      }
      const auto& rp =
          static_cast<const fast::RoPE&>(node.primitive());
      const auto& st = rp.state();
      // Norm-fused ropes ride three inputs (x, offset, weight); the
      // fused-norm kernel route only covers forward non-traditional.
      if (!rp.has_norm() || node.inputs().size() != 3 ||
          !std::get<5>(st) || std::get<2>(st) ||
          node.inputs()[1].size() != 1 ||
          node.inputs()[2].dtype() != bfloat16) {
        if (epi_trace) {
          ++tr_rope;
        }
        continue;
      }
      std::vector<array> chain;
      const array* cur = &node.inputs()[0];
      bool ok = cur->dtype() == bfloat16;
      while (ok &&
             (is_op(cur, typeid(Reshape)) ||
              is_op(cur, typeid(Transpose)))) {
        if (use_count(*cur) != 1 || claimed.count(cur->id())) {
          ok = false;
          break;
        }
        chain.push_back(*cur);
        cur = &cur->inputs()[0];
      }
      if (!ok || cur->dtype() != bfloat16) {
        if (epi_trace) {
          ++tr_shape;
        }
        continue;
      }
      auto mit = member_of.find(cur->id());
      if (mit == member_of.end()) {
        if (epi_trace) {
          ++tr_member;
        }
        continue;
      }
      if (cur->primitive().stream() != node.primitive().stream()) {
        continue;
      }
      cands[mit->second.first].push_back(
          RopeCand{&node, std::move(chain), mit->second.second});
    }
    if (epi_trace) {
      std::fprintf(
          stderr,
          "[rope-epi] scan: ropes=%d shape_rej=%d member_rej=%d cand_groups=%zu\n",
          tr_rope,
          tr_shape,
          tr_member,
          cands.size());
      for (auto& e : cands) {
        std::fprintf(stderr, "[rope-epi]   group %zu sides=%zu\n", e.first, e.second.size());
      }
    }
    for (auto& entry : cands) {
      const size_t gi = entry.first;
      auto& sides = entry.second;
      if (sides.size() != 2) {
        continue;
      }
      // The keys side is the rope the slice pair redirect claimed;
      // the query side is the other. Exactly one of each.
      const RopeCand* qc = nullptr;
      const RopeCand* kc = nullptr;
      for (const auto& side : sides) {
        if (state->rope_redirect_roles.count(side.node->id())) {
          if (kc != nullptr) {
            qc = nullptr;
            break;
          }
          kc = &side;
        } else {
          qc = &side;
        }
      }
      if (qc == nullptr || kc == nullptr ||
          qc->member_index == kc->member_index ||
          // The keys rope's only consumer is the deleted slice update;
          // a second consumer would read an unmaterialized buffer.
          use_count(*kc->node) != 1) {
        if (epi_trace) {
          std::fprintf(
              stderr,
              "[rope-epi] group %zu: sides=%zu q=%d k=%d same_mi=%d same_off=%d k_use=%zu\n",
              gi,
              sides.size(),
              qc != nullptr,
              kc != nullptr,
              qc != nullptr && kc != nullptr &&
                  qc->member_index == kc->member_index,
              qc != nullptr && kc != nullptr &&
                  qc->node->inputs()[1].id() == kc->node->inputs()[1].id(),
              kc ? use_count(*kc->node) : 0);
        }
        continue;
      }
      // The keys window must already be planned producer-direct on
      // the pair that claimed the rope's output.
      const size_t pair_index =
          *state->rope_redirect_roles.find(kc->node->id());
      const auto& pair = state->slice_update_pairs[pair_index];
      if (!pair.direct || !pair.windows[0]) {
        if (epi_trace) {
          std::fprintf(
              stderr,
              "[rope-epi] group %zu: pair %zu direct=%d window=%d\n",
              gi,
              pair_index,
              pair.direct,
              pair.windows[0].has_value());
        }
        continue;
      }
      // The plan stores the query rope's state and the dispatch
      // applies it to both sides, so the keys rope must carry the
      // same parameters: dims, base, scale, and the norm epsilon.
      // Forward/non-traditional is already required per candidate.
      const auto& q_state =
          static_cast<const fast::RoPE&>(qc->node->primitive()).state();
      const auto& k_state =
          static_cast<const fast::RoPE&>(kc->node->primitive()).state();
      if (std::get<1>(q_state) != std::get<1>(k_state) ||
          std::get<3>(q_state) != std::get<3>(k_state) ||
          std::get<4>(q_state) != std::get<4>(k_state) ||
          static_cast<const fast::RoPE&>(qc->node->primitive())
                  .norm_eps() !=
              static_cast<const fast::RoPE&>(kc->node->primitive())
                  .norm_eps()) {
        if (epi_trace) {
          std::fprintf(
              stderr,
              "[rope-epi] group %zu: q/k rope state differs "
              "(dims %d/%d base %g/%g scale %g/%g)\n",
              gi,
              std::get<1>(q_state),
              std::get<1>(k_state),
              std::get<3>(q_state),
              std::get<3>(k_state),
              std::get<4>(q_state),
              std::get<4>(k_state));
        }
        continue;
      }
      const auto& st = q_state;
      auto& group = state->gemv_groups[gi];
      const auto q_layout = compose_view(
          group.members[qc->member_index].node, qc->chain);
      const auto k_layout = compose_view(
          group.members[kc->member_index].node, kc->chain);
      group.rope_epi = RopeEpiPlan{
          RopeEpiSide{
              *qc->node,
              qc->member_index,
              qc->chain,
              q_layout.first,
              q_layout.second,
              0,
              std::nullopt,
              pair_index},
          RopeEpiSide{
              *kc->node,
              kc->member_index,
              kc->chain,
              k_layout.first,
              k_layout.second,
              0,
              pair.windows[0],
              pair_index},
          std::get<1>(st),
          std::get<2>(st),
          std::get<3>(st),
          std::get<4>(st),
          static_cast<const fast::RoPE&>(qc->node->primitive())
              .norm_eps()};
      for (const auto* side : {qc, kc}) {
        state->roles.emplace(
            side->node->id(), EagerRole{gi, EagerStep::skip});
        claimed.put_if_absent(side->node->id(), 1);
        for (const auto& view : side->chain) {
          state->roles.emplace(view.id(), EagerRole{gi, EagerStep::skip});
          claimed.put_if_absent(view.id(), 1);
        }
      }
    }
  }
  // Decode SDPA out-gate epilogue fold (MLX_OMARCHY_SDPA_OUTGATE=1):
  // matches tail = Multiply(Reshape(Transpose(SDPA)), Sigmoid(gate)) for
  // single-token (B=1, q_len=1, HD=128|256) bf16 attention. When gate is
  // itself a post-SDPA Reshape of a pre-SDPA 4D (1,1,H,D) strided split view
  // (Qwen3.5/Qwen3.8 q_proj split), skips the gate CopyGeneral as well as
  // Sigmoid and Multiply, writing tail directly from SDPA.
  if (fused_chain_enabled()) {
    auto use_count = [&](const array& value) {
      auto* it = uses.find(value.id());
      return it == nullptr ? size_t{0} : *it;
    };
    auto contiguous_strides = [](const Shape& shape) {
      Strides strides(shape.size(), 0);
      strides.back() = 1;
      for (int ax = static_cast<int>(shape.size()) - 2; ax >= 0; --ax) {
        strides[ax] = strides[ax + 1] * shape[ax + 1];
      }
      return strides;
    };
    // Note: eval_impl pops from the back of tape (higher index runs earlier).
    std::unordered_map<std::uintptr_t, size_t> tape_pos;
    tape_pos.reserve(tape.size());
    for (size_t ti = 0; ti < tape.size(); ++ti) {
      tape_pos.emplace(tape[ti].id(), ti);
    }
    for (const auto& tail : tape) {
      if (!is_op(&tail, typeid(Multiply)) || tail.inputs().size() != 2 ||
          tail.dtype() != bfloat16 || claimed.count(tail.id()) ||
          tail.ndim() != 3 || tail.shape(0) != 1 || tail.shape(1) != 1) {
        continue;
      }
      const Device* tail_dev = stream_device(tail);
      if (tail_dev == nullptr ||
          !sdpa_outgate_enabled(tail_dev->capabilities())) {
        continue;
      }
      const array* left = lookup(tail.inputs()[0]);
      const array* right = lookup(tail.inputs()[1]);
      const array* sigmoid = is_op(left, typeid(Sigmoid)) ? left
          : (is_op(right, typeid(Sigmoid)) ? right : nullptr);
      if (sigmoid == nullptr || sigmoid->inputs().size() != 1 ||
          use_count(*sigmoid) != 1 || claimed.count(sigmoid->id()) ||
          sigmoid->dtype() != bfloat16 ||
          sigmoid->primitive().stream() != tail.primitive().stream()) {
        continue;
      }
      const array& out_in =
          sigmoid == left ? tail.inputs()[1] : tail.inputs()[0];
      const array& gate_flat = sigmoid->inputs()[0];
      if (out_in.id() == gate_flat.id() || out_in.dtype() != bfloat16 ||
          gate_flat.dtype() != bfloat16 || out_in.shape() != tail.shape() ||
          gate_flat.shape() != tail.shape()) {
        continue;
      }
      const array* out_view = lookup(out_in);
      if (out_view == nullptr || !is_op(out_view, typeid(Reshape)) ||
          use_count(*out_view) != 1 || claimed.count(out_view->id()) ||
          out_view->inputs().empty()) {
        continue;
      }
      std::vector<const array*> out_chain;
      const array* cur = &out_view->inputs()[0];
      bool ok = true;
      while (is_op(cur, typeid(Transpose)) || is_op(cur, typeid(Reshape))) {
        if (use_count(*cur) != 1 || claimed.count(cur->id())) {
          ok = false;
          break;
        }
        out_chain.push_back(cur);
        cur = &cur->inputs()[0];
      }
      if (!ok || out_chain.empty() ||
          !is_op(cur, typeid(fast::ScaledDotProductAttention)) ||
          cur->dtype() != bfloat16 || cur->ndim() != 4 ||
          cur->shape(0) != 1 || cur->shape(2) != 1 ||
          (cur->shape(3) != 128 && cur->shape(3) != 256) ||
          cur->inputs().size() != 3 || use_count(*cur) != 1 ||
          claimed.count(cur->id()) ||
          cur->primitive().stream() != tail.primitive().stream()) {
        continue;
      }
      const array& sdpa = *cur;
      const int heads = sdpa.shape(1);
      const int head_dim = sdpa.shape(3);
      if (tail.shape() != Shape{1, 1, heads * head_dim}) {
        continue;
      }
      // Verify the view chain from contiguous (1,H,1,D) maps head-major
      // (head * D + d) contiguously into (1,1,H*D).
      Shape map_shape(sdpa.shape());
      Strides map_strides = contiguous_strides(map_shape);
      bool composes = true;
      for (auto it = out_chain.rbegin(); it != out_chain.rend(); ++it) {
        const array& view = **it;
        if (is_op(&view, typeid(Transpose))) {
          const auto& axes =
              static_cast<const Transpose&>(view.primitive()).state();
          if (axes.size() != map_shape.size()) {
            composes = false;
            break;
          }
          Shape next_shape(map_shape.size());
          Strides next_strides(map_shape.size());
          for (size_t ax = 0; ax < axes.size(); ++ax) {
            if (axes[ax] < 0 ||
                static_cast<size_t>(axes[ax]) >= map_shape.size()) {
              composes = false;
              break;
            }
            next_shape[ax] = map_shape[axes[ax]];
            next_strides[ax] = map_strides[axes[ax]];
          }
          if (!composes) {
            break;
          }
          map_shape = std::move(next_shape);
          map_strides = std::move(next_strides);
        } else {
          if (map_strides != contiguous_strides(map_shape)) {
            composes = false;
            break;
          }
          map_shape = view.shape();
          map_strides = contiguous_strides(map_shape);
        }
      }
      if (!composes || map_shape != Shape{1, 1, heads, head_dim} ||
          map_strides[2] != static_cast<ptrdiff_t>(head_dim) ||
          map_strides[3] != 1) {
        continue;
      }
      // Higher tape_pos runs earlier: sdpa must run before sigmoid and tail.
      const size_t sdpa_pos = tape_pos[sdpa.id()];
      if (tape_pos[sigmoid->id()] >= sdpa_pos ||
          tape_pos[tail.id()] >= sdpa_pos) {
        continue;
      }
      const array* gate_flat_node = lookup(gate_flat);
      const array* skip_gate_reshape = nullptr;
      std::optional<array> gate_src;
      if (gate_flat_node != nullptr &&
          is_op(gate_flat_node, typeid(Reshape)) &&
          use_count(*gate_flat_node) == 1 &&
          !claimed.count(gate_flat_node->id()) &&
          gate_flat_node->inputs().size() == 1 &&
          gate_flat_node->primitive().stream() == tail.primitive().stream() &&
          tape_pos[gate_flat_node->id()] < sdpa_pos) {
        const array& gsrc = gate_flat_node->inputs()[0];
        auto gsrc_pos = tape_pos.find(gsrc.id());
        if (gsrc.dtype() == bfloat16 && gsrc.ndim() == 4 &&
            gsrc.shape() == Shape{1, 1, heads, head_dim} &&
            (gsrc_pos == tape_pos.end() || gsrc_pos->second > sdpa_pos)) {
          gate_src = gsrc;
          skip_gate_reshape = gate_flat_node;
        }
      }
      if (!gate_src.has_value()) {
        auto gf_pos = tape_pos.find(gate_flat.id());
        if (gf_pos == tape_pos.end() || gf_pos->second > sdpa_pos) {
          gate_src = gate_flat;
        } else {
          continue;
        }
      }
      const size_t index = state->sdpa_outgate_folds.size();
      state->sdpa_outgate_folds.push_back(
          SdpaOutgateFold{SdpaOutgatePlan{tail, *gate_src}, false});
      state->sdpa_outgate_plans.emplace(sdpa.id(), index);
      claimed.put_if_absent(sdpa.id(), 1);
      for (const array* view : out_chain) {
        state->sdpa_outgate_skip_roles.emplace(view->id(), index);
        claimed.put_if_absent(view->id(), 1);
      }
      state->sdpa_outgate_skip_roles.emplace(out_view->id(), index);
      claimed.put_if_absent(out_view->id(), 1);
      if (skip_gate_reshape != nullptr) {
        state->sdpa_outgate_skip_roles.emplace(skip_gate_reshape->id(), index);
        claimed.put_if_absent(skip_gate_reshape->id(), 1);
      }
      state->sdpa_outgate_skip_roles.emplace(sigmoid->id(), index);
      claimed.put_if_absent(sigmoid->id(), 1);
      state->sdpa_outgate_skip_roles.emplace(tail.id(), index);
      claimed.put_if_absent(tail.id(), 1);
    }
  }
  // Decode-rows SDPA out-transpose fold: a materializing Reshape whose
  // single-consumer view chain bottoms out at a bf16
  // ScaledDotProductAttention is exactly the (1,H,S,D) -> (1,S,H*D)
  // transpose-copy the verify route's consumers build. The rows kernel
  // can store the transposed layout directly, deleting the copy. The
  // dispatch re-verifies the shape contract and falls back eager when
  // it cannot run.
  {
    auto use_count = [&](const array& value) {
      auto* it = uses.find(value.id());
      return it == nullptr ? size_t{0} : *it;
    };
    auto contiguous_strides = [](const Shape& shape) {
      Strides strides(shape.size(), 0);
      strides.back() = 1;
      for (int ax = static_cast<int>(shape.size()) - 2; ax >= 0; --ax) {
        strides[ax] = strides[ax + 1] * shape[ax + 1];
      }
      return strides;
    };
    for (const auto& node : tape) {
      if (!is_op(&node, typeid(Reshape)) || node.dtype() != bfloat16 ||
          claimed.count(node.id()) || node.ndim() != 3 ||
          node.shape(0) != 1 || node.inputs().empty()) {
        continue;
      }
      std::vector<const array*> chain;
      const array* cur = &node.inputs()[0];
      bool ok = true;
      while (is_op(cur, typeid(Transpose)) || is_op(cur, typeid(Reshape))) {
        if (use_count(*cur) != 1) {
          ok = false;
          break;
        }
        chain.push_back(cur);
        cur = &cur->inputs()[0];
      }
      if (!ok || chain.empty() ||
          !is_op(cur, typeid(fast::ScaledDotProductAttention)) ||
          cur->dtype() != bfloat16 || cur->ndim() != 4 ||
          cur->shape(0) != 1 || use_count(*cur) != 1 ||
          claimed.count(cur->id())) {
        continue;
      }
      const array& sdpa = *cur;
      // Compose the view map from the contiguous (1,H,S,D) out through
      // the chain, sdpa-side first: Transpose always composes; a
      // Reshape composes only off a contiguous map.
      Shape map_shape(sdpa.shape());
      Strides map_strides = contiguous_strides(map_shape);
      bool composes = true;
      for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        const array& view = **it;
        if (is_op(&view, typeid(Transpose))) {
          const auto& axes =
              static_cast<const Transpose&>(view.primitive()).state();
          if (axes.size() != map_shape.size()) {
            composes = false;
            break;
          }
          Shape next_shape(map_shape.size());
          Strides next_strides(map_shape.size());
          bool axes_ok = true;
          for (size_t ax = 0; ax < axes.size(); ++ax) {
            if (axes[ax] < 0 ||
                static_cast<size_t>(axes[ax]) >= map_shape.size()) {
              axes_ok = false;
              break;
            }
            next_shape[ax] = map_shape[axes[ax]];
            next_strides[ax] = map_strides[axes[ax]];
          }
          if (!axes_ok) {
            composes = false;
            break;
          }
          map_shape = std::move(next_shape);
          map_strides = std::move(next_strides);
        } else {
          if (map_strides != contiguous_strides(map_shape)) {
            composes = false;
            break;
          }
          map_shape = view.shape();
          map_strides = contiguous_strides(map_shape);
        }
      }
      // The chain's result must be exactly the (1,S,H,D) transposed
      // view of the out: strides (S*H*D, D, S*D, 1). The Reshape then
      // merges H*D contiguously, so node.shape is (1,S,H*D).
      if (!composes || map_shape.size() != 4 || map_shape[0] != 1 ||
          map_strides[0] != static_cast<ptrdiff_t>(
                                map_shape[1] * map_shape[2] * map_shape[3]) ||
          map_strides[1] != static_cast<ptrdiff_t>(map_shape[3]) ||
          map_strides[2] != static_cast<ptrdiff_t>(
                                map_shape[1] * map_shape[3]) ||
          map_strides[3] != 1 ||
          node.shape() !=
              Shape{1, map_shape[1], map_shape[2] * map_shape[3]}) {
        continue;
      }
      const size_t index = state->sdpa_transpose_folds.size();
      state->sdpa_transpose_folds.push_back(SdpaTransposeFold{node, {}, false});
      auto& fold = state->sdpa_transpose_folds.back();
      for (const array* view : chain) {
        fold.chain.push_back(*view);
        state->sdpa_transpose_skip_roles.emplace(view->id(), index);
        claimed.put_if_absent(view->id(), 1);
      }
      state->sdpa_transpose_plans.emplace(sdpa.id(), index);
      state->sdpa_transpose_skip_roles.emplace(node.id(), index);
      claimed.put_if_absent(node.id(), 1);
    }
  }

  // Fused residual Add -> RMSNorm pairs (M2D-2): a bf16 same-shape
  // BinaryVecBF16(Add) whose reader is an unclaimed fast::RMSNorm on
  // the same stream folds into ONE FastAddRmsNormAppleBF16 dispatch at
  // the add's tape position. The add's rounded residual stays
  // multi-reader safe: the fused kernel writes the residual buffer the
  // composed add would have written. The apple device gate lives at
  // fire time (TU-local to primitives.cpp); a refused device un-plans
  // on the first fire and every node keeps today's route.
  // MLX_OMARCHY_ADD_NORM_FUSE=0 plans nothing (the off arm for the
  // bit-exactness test and bisection).
  if (omarchy::env_str("MLX_OMARCHY_ADD_NORM_FUSE") == nullptr ||
      omarchy::env_flag("MLX_OMARCHY_ADD_NORM_FUSE")) {
    std::unordered_map<std::uintptr_t, const array*> norm_by_in;
    for (const auto& node : tape) {
      if (!is_op(&node, typeid(fast::RMSNorm)) || node.dtype() != bfloat16 ||
          node.inputs().size() != 2 || claimed.count(node.id())) {
        continue;
      }
      const array& in = node.inputs()[0];
      if (in.dtype() != bfloat16 || in.shape() != node.shape() ||
          !is_op(&in, typeid(Add)) || in.inputs().size() != 2 ||
          in.inputs()[0].dtype() != bfloat16 ||
          in.inputs()[1].dtype() != bfloat16 ||
          in.inputs()[0].shape() != in.shape() ||
          in.inputs()[1].shape() != in.shape() ||
          claimed.count(in.id())) {
        continue;
      }
      norm_by_in.emplace(in.id(), &node);
    }
    for (const auto& node : tape) {
      auto it = norm_by_in.find(node.id());
      if (it == norm_by_in.end() || claimed.count(node.id())) {
        continue;
      }
      const array* norm = it->second;
      if (norm->ndim() == 0) {
        continue;
      }
      size_t row_length = norm->shape(-1);
      if (row_length == 0) {
        continue;
      }
      size_t rows = norm->size() / row_length;
      const array& w = norm->inputs()[1];
      if (rows == 0 || w.dtype() != bfloat16 ||
          (w.size() != 1 && w.size() != row_length) ||
          norm->primitive().stream() != node.primitive().stream() ||
          norm->inputs()[0].primitive().stream() != node.primitive().stream()) {
        continue;
      }
      // Layout and readiness gate entirely at fire time (#944 review):
      // plan-time nodes are unscheduled, so their flags are meaningless.
      // The apple_norm_enabled device gate lives at fire time (it is
      // TU-local to primitives.cpp); a refused device un-plans on the
      // first fire and both nodes take today's route.
      const size_t index = state->add_norm_pairs.size();
      state->add_norm_roles.put_if_absent(node.id(), index);
      state->add_norm_roles.put_if_absent(norm->id(), index);
      state->add_norm_pairs.push_back(AddNormPair{{node, *norm}});
      claimed.put_if_absent(norm->id(), 1);
    }
  }

  // RMSNorm -> RoPE fold: a plain two-input fast::RoPE whose input is a
  // single-use fast::RMSNorm output gets the norm computed inside the
  // rope dispatch (the FastRopeNorm build; bf16 only). The claimed norm
  // never schedules, so its dispatch and the normed tensor's round trip
  // disappear. KV-direct windows stay armed: the fold runs the fused
  // primitive on the tape rope node itself.
  if (rope_norm_fold_enabled()) {
    static int fold_trace = 0;
    const bool trace = omarchy::env_str("MLX_OMARCHY_ROPE_NORM_TRACE") != nullptr;
    size_t cand = 0;
    for (const array& node : tape) {
      if (!is_op(&node, typeid(fast::RoPE)) || node.inputs().size() != 2 ||
          node.dtype() != bfloat16 ||
          static_cast<const fast::RoPE&>(node.primitive()).has_norm()) {
        continue;
      }
      ++cand;
      const array& in = node.inputs()[0];
      // The rope input is the RMSNorm output itself or one view of it.
      const array* norm_p = nullptr;
      const array* view_p = nullptr;
      if (is_op(&in, typeid(fast::RMSNorm))) {
        norm_p = &in;
      } else if (
          is_op(&in, typeid(Transpose)) || is_op(&in, typeid(Reshape))) {
        view_p = &in;
        norm_p = in.inputs().size() == 1 ? lookup(in.inputs()[0]) : nullptr;
      }
      if (norm_p == nullptr || !is_op(norm_p, typeid(fast::RMSNorm)) ||
          norm_p->inputs().size() != 2 || uses[norm_p->id()] != 1 ||
          uses[in.id()] != 1 || claimed.count(norm_p->id()) ||
          claimed.count(in.id()) || claimed.count(node.id()) ||
          in.dtype() != bfloat16) {
        if (trace && fold_trace < 4) {
          ++fold_trace;
          std::fprintf(
              stderr,
              "[rope-norm] rej: norm=%d ninputs=%zu uses_in=%zu "
              "claim_in=%d claim_rope=%d indt=%d prim=%s\n",
              norm_p != nullptr && is_op(norm_p, typeid(fast::RMSNorm)),
              in.inputs().size(),
              uses.count(in.id()) ? uses[in.id()] : 0,
              static_cast<int>(claimed.count(in.id())),
              static_cast<int>(claimed.count(node.id())),
              static_cast<int>(in.dtype().val()),
              in.has_primitive() ? in.primitive().name() : "<none>");
        }
        continue;
      }
      const array& n = *norm_p;
      const array& x = n.inputs()[0];
      const array& w = n.inputs()[1];
      // A view between norm and rope reads a row-major norm output, so
      // its layout replays bit-exact only over a row-major norm input.
      // (x is lazy at plan time; data_size is not populated until eval.)
      if (view_p != nullptr &&
          (!x.flags().row_contiguous ||
           in.offset() % in.itemsize() != 0 ||
           in.shape(-1) != x.shape(-1))) {
        if (trace && fold_trace < 8) {
          ++fold_trace;
          std::fprintf(
              stderr,
              "[rope-norm] rej-view: rowc=%d off=%lld in_d=%d x_d=%d\n",
              static_cast<int>(x.flags().row_contiguous),
              static_cast<long long>(in.offset()),
              in.shape(-1),
              x.shape(-1));
        }
        continue;
      }
      const auto& st =
          static_cast<const fast::RoPE&>(node.primitive()).state();
      const array& src = view_p != nullptr ? in : x;
      const int d = x.ndim() > 0 ? x.shape(-1) : 0;
      if (x.dtype() != bfloat16 || w.dtype() != bfloat16 ||
          w.size() != static_cast<size_t>(d) || d <= 0 || d > 256 ||
          (d % 2) != 0 || src.strides().back() != 1 || std::get<2>(st) ||
          !std::get<5>(st) ||
          fast::RoPE::use_fallback(node.primitive().stream())) {
        if (trace && fold_trace < 4) {
          ++fold_trace;
          std::fprintf(
              stderr,
              "[rope-norm] rej2: xdt=%d wdt=%d wsize=%zu d=%d sback=%lld "
              "trad=%d fwd=%d\n",
              static_cast<int>(x.dtype().val()),
              static_cast<int>(w.dtype().val()),
              w.size(),
              d,
              static_cast<long long>(x.strides().back()),
              static_cast<int>(std::get<2>(st)),
              static_cast<int>(std::get<5>(st)));
        }
        continue;
      }
      const size_t index = state->rope_norms.size();
      state->rope_norm_skip_roles.put_if_absent(n.id(), index);
      state->rope_norm_skip_roles.put_if_absent(in.id(), index);
      state->rope_norm_roles.put_if_absent(node.id(), index);
      state->rope_norms.push_back(RopeNormFold{
          n,
          node,
          view_p != nullptr ? std::optional<array>(in) : std::nullopt,
          x,
          w,
          n.primitive_ptr(),
          view_p != nullptr ? view_p->primitive_ptr() : nullptr,
          static_cast<const fast::RMSNorm&>(n.primitive()).state().second});
      claimed.put_if_absent(n.id(), 1);
      claimed.put_if_absent(in.id(), 1);
      claimed.put_if_absent(node.id(), 1);
    }
    if (trace && fold_trace < 8) {
      ++fold_trace;
      std::fprintf(
          stderr,
          "[rope-norm] planned=%zu cand=%zu\n",
          state->rope_norms.size(),
          cand);
    }
  }
  // Fused Q/K RMSNorm+RoPE+KV-cache-write+online-softmax BF16 decode SDPA
  // (HD=128 or HD=256, batch 1, q_len 1; MLX_OMARCHY_SDPA_ONLINE_ROPE=1).
  // Collapses FastRopeNormBF16(Q) + FastRopeNormBF16(K -> k_cache) +
  // SdpaDecodeOnline (plus optional SdpaOutgate epilogue) into a single
  // 1024-thread dispatch per attention layer.
  if (fused_chain_enabled()) {
    for (const array& sdpa : tape) {
      if (!is_op(&sdpa, typeid(fast::ScaledDotProductAttention)) ||
          sdpa.dtype() != bfloat16 || sdpa.ndim() != 4 || sdpa.shape(0) != 1 ||
          sdpa.shape(2) != 1 ||
          (sdpa.shape(3) != 128 && sdpa.shape(3) != 256) ||
          sdpa.inputs().size() != 3 ||
          (claimed.count(sdpa.id()) &&
           !state->sdpa_outgate_plans.count(sdpa.id()))) {
        continue;
      }
      const int head_dim = sdpa.shape(3);
      const array* q_rope = lookup(sdpa.inputs()[0]);
      const array* k_in = lookup(sdpa.inputs()[1]);
      const array& v_in = sdpa.inputs()[2];
      if (q_rope == nullptr || k_in == nullptr ||
          q_rope->dtype() != bfloat16 || k_in->dtype() != bfloat16 ||
          v_in.dtype() != bfloat16 || k_in->ndim() != 4 || v_in.ndim() != 4 ||
          k_in->shape(0) != 1 || k_in->shape(3) != head_dim ||
          v_in.shape(0) != 1 || v_in.shape(3) != head_dim ||
          k_in->shape(2) <= 0 || k_in->shape(1) <= 0 ||
          sdpa.shape(1) % k_in->shape(1) != 0) {
        continue;
      }
      const Device* sdpa_dev = stream_device(sdpa);
      const bool has_outgate = state->sdpa_outgate_plans.count(sdpa.id()) != 0;
      if (sdpa_dev == nullptr ||
          !sdpa_online_rope_enabled(
              sdpa_dev->capabilities(),
              static_cast<uint32_t>(k_in->shape(2)),
              head_dim,
              has_outgate)) {
        continue;
      }
      if (!is_op(q_rope, typeid(fast::RoPE)) || uses[q_rope->id()] != 1 ||
          state->rope_redirect_roles.count(q_rope->id()) ||
          state->sdpa_online_rope_skip_roles.count(q_rope->id()) ||
          (claimed.count(q_rope->id()) &&
           !state->rope_norm_roles.count(q_rope->id()))) {
        continue;
      }
      const array* k_su = k_in;
      if (is_op(k_su, typeid(Slice)) && !k_su->inputs().empty()) {
        k_su = lookup(k_su->inputs()[0]);
      }
      if (k_su == nullptr || !is_op(k_su, typeid(SliceUpdate))) {
        continue;
      }
      auto su_it = state->slice_update_roles.find(k_su->id());
      if (su_it == nullptr) {
        continue;
      }
      const size_t pair_idx = *su_it;
      const auto& pair = state->slice_update_pairs[pair_idx];
      if (!pair.direct || !pair.windows[0].has_value() ||
          !pair.windows[1].has_value() ||
          pair.windows[0]->node.id() != k_su->id() ||
          state->sdpa_online_rope_su_roles.count(k_su->id())) {
        continue;
      }
      const array* k_rope = lookup(k_su->inputs()[1]);
      if (k_rope == nullptr || !is_op(k_rope, typeid(fast::RoPE)) ||
          uses[k_rope->id()] != 1 || k_rope->dtype() != bfloat16 ||
          state->sdpa_online_rope_skip_roles.count(k_rope->id()) ||
          (claimed.count(k_rope->id()) &&
           !state->rope_norm_roles.count(k_rope->id()))) {
        continue;
      }
      auto red_it = state->rope_redirect_roles.find(k_rope->id());
      if (red_it == nullptr || *red_it != pair_idx) {
        continue;
      }
      if (q_rope->primitive().stream() != sdpa.primitive().stream() ||
          k_rope->primitive().stream() != sdpa.primitive().stream()) {
        continue;
      }
      const auto& qr = static_cast<const fast::RoPE&>(q_rope->primitive());
      const auto& kr = static_cast<const fast::RoPE&>(k_rope->primitive());
      const auto& qst = qr.state();
      const auto& kst = kr.state();
      // Forward, non-traditional, matching rotary dims (128 for HD=128; any
      // even 2..256 for HD=256), matching base/scale, shared scalar offset.
      const int rope_dims = std::get<1>(qst);
      const bool dims_ok = rope_dims == std::get<1>(kst) &&
          (head_dim == 128
               ? rope_dims == 128
               : (rope_dims > 0 && (rope_dims % 2) == 0 &&
                  rope_dims <= head_dim));
      if (!std::get<5>(qst) || !std::get<5>(kst) || std::get<2>(qst) ||
          std::get<2>(kst) || !dims_ok ||
          std::get<3>(qst) != std::get<3>(kst) ||
          std::get<4>(qst) != std::get<4>(kst) ||
          q_rope->inputs().size() < 2 || k_rope->inputs().size() < 2 ||
          q_rope->inputs()[1].size() != 1 ||
          q_rope->inputs()[1].dtype() != int32 ||
          !rope_offset_shared(q_rope->inputs()[1], k_rope->inputs()[1])) {
        continue;
      }
      auto* q_fold_ptr = state->rope_norm_roles.find(q_rope->id());
      auto* k_fold_ptr = state->rope_norm_roles.find(k_rope->id());
      const bool q_inline_norm = qr.has_norm() && q_rope->inputs().size() == 3;
      const bool k_inline_norm = kr.has_norm() && k_rope->inputs().size() == 3;
      const bool q_folded_norm = q_fold_ptr != nullptr && !qr.has_norm();
      const bool k_folded_norm = k_fold_ptr != nullptr && !kr.has_norm();
      const bool q_plain = !qr.has_norm() && q_rope->inputs().size() == 2 &&
          q_fold_ptr == nullptr;
      const bool k_plain = !kr.has_norm() && k_rope->inputs().size() == 2 &&
          k_fold_ptr == nullptr;
      bool has_norm = false;
      float eps = 0.0f;
      int q_fold_idx = -1;
      int k_fold_idx = -1;
      array q_raw = q_rope->inputs()[0];
      array k_raw = k_rope->inputs()[0];
      std::optional<array> q_weight;
      std::optional<array> k_weight;
      std::optional<array> q_view;
      std::optional<array> k_view;
      std::shared_ptr<Primitive> q_view_prim;
      std::shared_ptr<Primitive> k_view_prim;
      if (q_inline_norm && k_inline_norm) {
        if (qr.norm_eps() != kr.norm_eps() ||
            q_rope->inputs()[2].dtype() != bfloat16 ||
            k_rope->inputs()[2].dtype() != bfloat16 ||
            q_rope->inputs()[2].size() != static_cast<size_t>(head_dim) ||
            k_rope->inputs()[2].size() != static_cast<size_t>(head_dim)) {
          continue;
        }
        has_norm = true;
        eps = qr.norm_eps();
        q_weight = q_rope->inputs()[2];
        k_weight = k_rope->inputs()[2];
      } else if (q_folded_norm && k_folded_norm) {
        const auto& qf = state->rope_norms[*q_fold_ptr];
        const auto& kf = state->rope_norms[*k_fold_ptr];
        if (qf.eps != kf.eps ||
            qf.w.size() != static_cast<size_t>(head_dim) ||
            kf.w.size() != static_cast<size_t>(head_dim)) {
          continue;
        }
        has_norm = true;
        eps = qf.eps;
        q_fold_idx = static_cast<int>(*q_fold_ptr);
        k_fold_idx = static_cast<int>(*k_fold_ptr);
        q_raw = qf.x;
        k_raw = kf.x;
        q_weight = qf.w;
        k_weight = kf.w;
        q_view = qf.view;
        k_view = kf.view;
        q_view_prim = qf.view_prim;
        k_view_prim = kf.view_prim;
      } else if (q_plain && k_plain) {
        has_norm = false;
      } else {
        continue;
      }
      if (q_raw.dtype() != bfloat16 || k_raw.dtype() != bfloat16) {
        continue;
      }
      const size_t plan_idx = state->sdpa_online_rope_plans.size();
      state->sdpa_online_rope_plans.push_back(SdpaOnlineRopePlan{
          *q_rope,
          *k_rope,
          sdpa,
          q_raw,
          k_raw,
          q_rope->inputs()[1],
          q_weight,
          k_weight,
          q_view,
          k_view,
          q_view_prim,
          k_view_prim,
          q_rope->primitive_ptr(),
          k_rope->primitive_ptr(),
          q_rope->inputs(),
          k_rope->inputs(),
          q_fold_idx,
          k_fold_idx,
          pair_idx,
          has_norm,
          eps,
          std::get<3>(qst),
          std::get<4>(qst),
          rope_dims,
          SdpaOnlineRopeCtx{},
          false,
          false,
          SdpaOnlineRopePlan::State::pending});
      state->sdpa_online_rope_skip_roles.emplace(q_rope->id(), plan_idx);
      state->sdpa_online_rope_skip_roles.emplace(k_rope->id(), plan_idx);
      state->sdpa_online_rope_su_roles.emplace(pair.nodes[0].id(), plan_idx);
      state->sdpa_online_rope_su_roles.emplace(pair.nodes[1].id(), plan_idx);
      state->sdpa_online_rope_roles.emplace(sdpa.id(), plan_idx);
      claimed.put_if_absent(q_rope->id(), 1);
      claimed.put_if_absent(k_rope->id(), 1);
      claimed.put_if_absent(sdpa.id(), 1);
    }
    if (omarchy::env_str("MLX_OMARCHY_SDPA_ROPE_TRACE") != nullptr &&
        !state->sdpa_online_rope_plans.empty()) {
      static int sr_trace = 0;
      if (sr_trace < 4) {
        ++sr_trace;
        std::fprintf(
            stderr,
            "[sdpa-rope] planned=%zu\n",
            state->sdpa_online_rope_plans.size());
      }
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
  // Per node: MLX_OMARCHY_FUSED_TRIO for the device that executes it,
  // independent of MLX_OMARCHY_FUSED_GEMV.
  {
    RopePairScanTrace trace;
    // The RoPE nodes of one decode step are NOT tape-adjacent: each
    // rope call materializes its scalar offset between them and the
    // cache write views interleave, so collect the rope subsequence in
    // tape order and pair consecutive members (query, key).
    if (omarchy::env_str("MLX_OMARCHY_TRIO_TRACE") != nullptr) {
      size_t rope_ops = 0;
      size_t trio_ok = 0;
      for (const auto& node : tape) {
        if (is_op(&node, typeid(fast::RoPE))) {
          rope_ops += 1;
          trio_ok += node_trio_enabled(node) ? 1 : 0;
        }
      }
      std::fprintf(
          stderr,
          "[trio-scan] tape=%zu rope_ops=%zu trio_ok=%zu\n",
          tape.size(),
          rope_ops,
          trio_ok);
    }
    std::vector<const array*> rope_nodes;
    // The bf16 norm pair is opt-in while the pair kernel trails the
    // solo dispatches it replaces (MLX_OMARCHY_ROPE_NORM_PAIR=1); off,
    // norm ropes keep their solo paths and the scan pairs f16 ropes.
    const bool norm_pair_on = rope_norm_pair_enabled();
    for (const auto& node : tape) {
      if (!is_op(&node, typeid(fast::RoPE)) || !node_trio_enabled(node)) {
        continue;
      }
      const auto& node_rope =
          static_cast<const fast::RoPE&>(node.primitive());
      // A bf16 rope the RMSNorm->RoPE fold claimed (or a 3-input
      // rope_rms_norm carrying its weight) pairs through the
      // FastRopeNormPair kernels; the pair absorbs the fold so its
      // claimed norm nodes stay skipped. Plain ropes keep the 2-input
      // f16 contract.
      const bool norm_rope =
          norm_pair_on &&
          (state->rope_norm_roles.count(node.id()) ||
           (node.inputs().size() == 3 && node_rope.has_norm()));
      if (node.inputs().size() != 2 && !norm_rope) {
        trace.rej_node_inputs += 1;
        continue;
      }
      if (node.inputs()[1].size() != 1) {
        trace.rej_node_offset += 1;
        continue;
      }
      if (node.dtype() != float16 &&
          !(norm_rope && node.dtype() == bfloat16)) {
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
      auto fold_index_of = [&](const array& n) -> int {
        auto it = state->rope_norm_roles.find(n.id());
        return it ? static_cast<int>(*it) : -1;
      };
      const int fold_first = fold_index_of(first);
      const int fold_second = fold_index_of(second);
      // Both sides take the same form: two plain ropes, two fold-claimed
      // ropes with one eps, or two 3-input rope_rms_norm ropes. A
      // fold/3-input mix is exotic enough to leave to the ordinary
      // paths.
      const bool first_3in =
          first.inputs().size() == 3 && rope_first.has_norm();
      const bool second_3in =
          second.inputs().size() == 3 && rope_second.has_norm();
      const bool fold_pair = fold_first >= 0 && fold_second >= 0;
      const bool prim_pair = first_3in && second_3in;
      if (rope_second.state() != rope_first.state() ||
          (fold_first >= 0) != (fold_second >= 0) ||
          first_3in != second_3in || (!fold_pair && !prim_pair &&
          (fold_first >= 0 || first_3in)) ||
          (fold_pair &&
           (state->rope_norms[fold_first].eps !=
            state->rope_norms[fold_second].eps)) ||
          (prim_pair && rope_first.norm_eps() != rope_second.norm_eps()) ||
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
      // A fold-claimed rope's only claim is the fold the pair absorbs
      // (the fold itself refuses already-claimed ropes), so its claim
      // does not block pairing; any other claim does.
      auto non_fold_claim = [&](const array& n) {
        return claimed.count(n.id()) &&
            !state->rope_norm_roles.count(n.id());
      };
      if (non_fold_claim(first) || non_fold_claim(second)) {
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
      state->rope_pair_roles.put_if_absent(first.id(), index);
      state->rope_pair_roles.put_if_absent(second.id(), index);
      RopePair new_pair{{first, second}};
      new_pair.norm_folds = {fold_first, fold_second};
      state->rope_pairs.push_back(std::move(new_pair));
      claimed.put_if_absent(first.id(), 1);
      claimed.put_if_absent(second.id(), 1);
      trace.pairs += 1;
      i += 2;
    }
    static int trio_trace_plans = 0;
    if (omarchy::env_str("MLX_OMARCHY_TRIO_TRACE") != nullptr &&
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

  // Quantized-embedding prologue fold: a dequantize-direction
  // fast::Quantize whose three inputs each trace through single-use
  // Squeeze/Reshape/ExpandDims views to a Gather taking one full row of
  // a table on axis 0 by the same index array gets one gather-dequant
  // dispatch; the gathers, views and dequantize never run on their
  // own. Word-pack affine dequantize only (bits 2/4/8); anything else
  // keeps the ordinary path. MLX_OMARCHY_PROLOGUE_FOLD=0 disables.
  const bool prologue_fold =
      omarchy::env_str("MLX_OMARCHY_PROLOGUE_FOLD") == nullptr ||
      env_flag("MLX_OMARCHY_PROLOGUE_FOLD");
  for (const auto& node : tape) {
    if (!prologue_fold) {
      break;
    }
    if (!is_op(&node, typeid(fast::Quantize)) ||
        node.inputs().size() != 3 || claimed.count(node.id())) {
      continue;
    }
    const auto qst =
        static_cast<const fast::Quantize&>(node.primitive()).state();
    const int group_size = std::get<1>(qst);
    const int bits = std::get<2>(qst);
    const auto mode = std::get<3>(qst);
    const bool dequantize = std::get<4>(qst);
    if (!dequantize || mode != QuantizationMode::Affine ||
        (bits != 2 && bits != 4 && bits != 8) || group_size <= 0 ||
        (group_size * bits) % 32 != 0) {
      continue;
    }
    std::vector<const array*> gathers(3, nullptr);
    std::vector<std::vector<const array*>> arm_views(3);
    const array* ids = nullptr;
    bool ok = true;
    for (int arm = 0; arm < 3 && ok; ++arm) {
      const array* cur = lookup(node.inputs()[arm]);
      while (ok && cur != nullptr &&
             (is_op(cur, typeid(Squeeze)) || is_op(cur, typeid(Reshape)) ||
              is_op(cur, typeid(ExpandDims)))) {
        if (uses[cur->id()] != 1 || cur->inputs().size() != 1 ||
            claimed.count(cur->id())) {
          ok = false;
          break;
        }
        arm_views[arm].push_back(cur);
        cur = lookup(cur->inputs()[0]);
      }
      if (!ok || cur == nullptr || !is_op(cur, typeid(Gather)) ||
          uses[cur->id()] != 1 || claimed.count(cur->id()) ||
          cur->inputs().size() != 2 ||
          cur->primitive().stream() != node.primitive().stream()) {
        break;
      }
      const auto gst =
          static_cast<const Gather&>(cur->primitive()).state();
      const array& table = cur->inputs()[0];
      // Single full-row take on axis 0: slice covers row 0..1 plus the
      // table's remaining extents in full.
      if (gst.first.size() != 1 || gst.first[0] != 0 ||
          gst.second.size() != static_cast<size_t>(table.ndim()) ||
          table.ndim() != 2 || gst.second[0] != 1 ||
          gst.second[1] != table.shape(1)) {
        break;
      }
      const array* idx = &cur->inputs()[1];
      if (ids == nullptr) {
        ids = idx;
      } else if (idx->id() != ids->id()) {
        break;
      }
      gathers[arm] = cur;
    }
    if (gathers[0] == nullptr || gathers[1] == nullptr ||
        gathers[2] == nullptr || ids == nullptr) {
      continue;
    }
    const array& w = gathers[0]->inputs()[0];
    const array& scales = gathers[1]->inputs()[0];
    const array& biases = gathers[2]->inputs()[0];
    const int64_t nrows = ids->size();
    const int64_t words_per_row = w.shape(1);
    const int64_t groups_per_row = scales.shape(1);
    const int64_t elems_per_row = words_per_row * (32 / bits);
    if (w.shape(0) != scales.shape(0) || w.shape(0) != biases.shape(0) ||
        scales.shape(1) != biases.shape(1) ||
        elems_per_row % group_size != 0 ||
        groups_per_row * group_size != elems_per_row ||
        node.size() != nrows * elems_per_row || nrows <= 0 ||
        (ids->dtype() != int32 && ids->dtype() != uint32) ||
        w.dtype() != uint32 || scales.dtype() != node.dtype() ||
        biases.dtype() != node.dtype() ||
        (node.dtype() != float32 && node.dtype() != float16 &&
         node.dtype() != bfloat16)) {
      continue;
    }
    std::vector<PrologueFold::ReplayStep> replay;
    for (int arm = 0; arm < 3; ++arm) {
      replay.push_back(
          PrologueFold::ReplayStep{
              *gathers[arm],
              gathers[arm]->primitive_ptr(),
              gathers[arm]->inputs()});
      for (auto it = arm_views[arm].rbegin(); it != arm_views[arm].rend();
           ++it) {
        replay.push_back(
            PrologueFold::ReplayStep{
                **it, (*it)->primitive_ptr(), (*it)->inputs()});
      }
    }
    const size_t index = state->prologue_folds.size();
    state->prologue_fold_roles.put_if_absent(node.id(), index);
    for (const auto& step : replay) {
      state->prologue_skip_roles.put_if_absent(step.node.id(), index);
      claimed.put_if_absent(step.node.id(), 1);
    }
    claimed.put_if_absent(node.id(), 1);
    state->prologue_folds.push_back(PrologueFold{
        node,
        *ids,
        w,
        scales,
        biases,
        std::move(replay),
        bits,
        group_size,
        words_per_row,
        groups_per_row,
        nrows,
        std::nullopt,
        std::nullopt,
        0.0f,
        false,
        PrologueFold::State::pending});
    // G14C-gated fused embedding GatherDequantBF16 + FastRmsNormAppleBF16
    // (MLX_OMARCHY_PROLOGUE_NORM): eliminates 1 serial CDM barrier + 1-WG gather.
    const Device* pn_dev = stream_device(node);
    if (pn_dev != nullptr &&
        omarchy::prologue_norm_default_part(
            pn_dev->hardware_capabilities().device_name) &&
        node.dtype() == bfloat16 && nrows == 1 && bits == 4 &&
        group_size == 64 &&
        (elems_per_row == 2048 || elems_per_row == 2560 ||
         elems_per_row == 4096)) {
      for (const auto& cand : tape) {
        if (!is_op(&cand, typeid(fast::RMSNorm)) || cand.inputs().size() != 2 ||
            claimed.count(cand.id()) || cand.dtype() != bfloat16 ||
            cand.primitive().stream() != node.primitive().stream() ||
            lookup(cand.inputs()[1]) != nullptr ||
            cand.size() != node.size()) {
          continue;
        }
        const array* cin = &cand.inputs()[0];
        bool match_in = (cin->id() == node.id());
        if (!match_in) {
          const array* v = lookup(*cin);
          if (v != nullptr &&
              (is_op(v, typeid(Reshape)) || is_op(v, typeid(Squeeze)) ||
               is_op(v, typeid(ExpandDims))) &&
              v->inputs().size() == 1 && v->inputs()[0].id() == node.id()) {
            match_in = true;
          }
        }
        if (!match_in) {
          continue;
        }
        const array& nw = cand.inputs()[1];
        if (nw.dtype() != bfloat16 ||
            (nw.size() != static_cast<size_t>(elems_per_row) &&
             nw.size() != 1u)) {
          continue;
        }
        const float eps =
            static_cast<const fast::RMSNorm&>(cand.primitive()).state().second;
        auto& pf = state->prologue_folds[index];
        pf.norm_out = cand;
        pf.norm_weight = nw;
        pf.norm_eps = eps;
        state->prologue_norm_skip_roles.put_if_absent(cand.id(), index);
        claimed.put_if_absent(cand.id(), 1);
        break;
      }
    }
  }
  if (use_cache) {
    if (auto tmpl = extract_fold_plan_template(*state, tape, key_id_map)) {
      fold_plan_cache_insert(plan_key, std::move(*tmpl));
    }
  }
  if (fold_timing) {
    print_fold_timing(tape.size(), plan_t0, *state, cache_status);
  }
}

EagerFusionScope::~EagerFusionScope() {
  static const bool fold_timing =
      std::getenv("MLX_OMARCHY_FOLD_TIMING") != nullptr;
  if (fold_timing) {
    double* ph = eval_phase_slots();
    std::fprintf(
        stderr,
        "[fold-time] span us=%.0f | dispatch us=%.0f n=%zu | eval us=%.0f "
        "n=%.0f dispatch-blk us=%.0f bookkeeping us=%.0f commit us=%.0f "
        "finalize us=%.0f\n",
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0_)
            .count(),
        eager_dispatch_us,
        eager_dispatch_calls,
        ph[0],
        ph[5],
        ph[1],
        ph[2],
        ph[3],
        ph[4]);
    for (int i = 0; i < 8; ++i) {
      ph[i] = 0.0;
    }
  }
  eager_dispatch_us = 0.0;
  eager_dispatch_calls = 0;
  if (eager_state != nullptr) {
    eager_state->reset();
    auto& pool = fusion_state_pool();
    if (pool.size() < 4) {
      pool.emplace_back(eager_state);
    } else {
      delete eager_state;
    }
  }
  eager_state = static_cast<EagerFusionState*>(previous_);
}

// Out-gate prologue fold (sigmoid*multiply recomputed per o_proj workgroup).
// Chip-keyed like the other decode-path defaults: every o_proj workgroup
// redoes the gate on G13 (G13G dropped ~1.7%, H147; G13C M1 Max gained
// +2.79% 5/5 at ctx=512 and +2.47% 3/3 at ctx=2048 on Qwen3.5-9B paired ABBA
// when disabled atop the 2x8w2 QMV path) and on G14C (M2 Max B1 paired ABBA
// across Qwen3.8-2B and Qwen3.8-9B gained +0.97% to +1.92% when disabled,
// 11/11 pair wins).
// MLX_OMARCHY_OUTGATE_FOLD=0/1 forces either arm.
bool outgate_fold_default_for_device(const std::string& device_name) {
  bool g13 = device_name.find("G13") != std::string::npos;
  bool g14c = device_name.find("G14C") != std::string::npos;
  return !g13 && !g14c;
}

bool outgate_fold_enabled() {
  const char* v = omarchy::env_str("MLX_OMARCHY_OUTGATE_FOLD");
  if (v != nullptr && *v != '\0') {
    return !(v[0] == '0' && v[1] == '\0');
  }
  return outgate_fold_default_for_device(device().capabilities().device_name);
}

bool qmm_residual_norm_enabled(
    const CapabilityReport& caps, int norm_cols, bool outgate) {
  // The plain and out-gate kernels use 16 and 8 32-lane slots respectively.
  const uint32_t threads = outgate ? 256u : 512u;
  if (!fused_chain_enabled() || caps.subgroup_size != 32 ||
      (caps.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) == 0 ||
      caps.max_compute_work_group_invocations < threads ||
      caps.max_compute_work_group_size[0] < threads) {
    return false;
  }
  const char* env = omarchy::env_str("MLX_OMARCHY_QMM_RESIDUAL_NORM");
  if (env != nullptr && env[0] != '\0') {
    return std::strcmp(env, "0") != 0;
  }
  // Width allowlist: {256, 2048, 4096} on every part that passed the
  // subgroup-32 probe above. 2560 stays a measured exception: G13C,
  // G14C, plus non-unified-memory devices. G13G keeps the exclusion:
  // the isolated 5-pair jwm1 ABBA the #947 review asked for
  // (MLX_OMARCHY_QMM_RESIDUAL_NORM =0 vs =1 on one wheel, default
  // 2x8w2 geometry, identical greedy IDs, DEC-31 receipt
  // 2026-11-28-g13g-resnorm2560-isolated) measures the fold at
  // -1.3% ctx512 (0/5 pairs) and -1.2% ctx2048 (1/5) on current
  // defaults, while the excluded arm decodes 4B at 24.23 tok/s.
  if (norm_cols == 2560) {
    return caps.device_name.find("G13C") != std::string::npos ||
        caps.device_name.find("G14C") != std::string::npos ||
        !caps.unified_memory;
  }
  return norm_cols == 256 || norm_cols == 2048 || norm_cols == 4096;
}

bool fused_route_enabled(const char* variable, const std::string& device_name) {
  if (!fused_chain_enabled()) {
    return false;
  }
  if (omarchy::env_str(variable) != nullptr) {
    return env_flag(variable);
  }
  if (std::strcmp(variable, "MLX_OMARCHY_FUSED_GEMV") == 0) {
    return !fused_gemv_off_default_part(device_name);
  }
  return true;
}

bool fused_gemv_enabled(const Device& device) {
  return fused_route_enabled(
      "MLX_OMARCHY_FUSED_GEMV", device.capabilities().device_name);
}

// Balanced multi-member GEMV groups (3-member QKV + balanced 2-member pairs)
// on the epilogue-only route. MLX_OMARCHY_GEMV_BALANCED_QKV=1 enables on any
// part (including alongside MLX_OMARCHY_FUSED_GEMV=0 +
// MLX_OMARCHY_QMM_EPILOGUE_GEMV=1); =0 disables everywhere; unset follows
// gemv_balanced_qkv_part (G14C) and respects explicit MLX_OMARCHY_FUSED_GEMV=0.
bool gemv_balanced_qkv_route_enabled(const std::string& device_name) {
  if (!fused_chain_enabled()) {
    return false;
  }
  const char* bq_env = omarchy::env_str("MLX_OMARCHY_GEMV_BALANCED_QKV");
  const bool bq_set = bq_env != nullptr && bq_env[0] != '\0';
  if (bq_set && std::strcmp(bq_env, "0") == 0) {
    return false;
  }
  const char* fg_env = omarchy::env_str("MLX_OMARCHY_FUSED_GEMV");
  const bool explicit_gemv_off =
      fg_env != nullptr && fg_env[0] != '\0' &&
      !env_flag("MLX_OMARCHY_FUSED_GEMV");
  if (explicit_gemv_off && !bq_set) {
    return false;
  }
  const char* ep_env = omarchy::env_str("MLX_OMARCHY_QMM_EPILOGUE_GEMV");
  const bool epilogue_route = (ep_env != nullptr && ep_env[0] != '\0')
      ? (std::strcmp(ep_env, "0") != 0)
      : gemv_epilogue_only_part(device_name);
  if (!epilogue_route) {
    return false;
  }
  return bq_set ? true : gemv_balanced_qkv_part(device_name);
}

// MLX_OMARCHY_FUSED_GEMV_SWIGLU=0 keeps the SwiGLU store epilogue off
// (explicit MLX_OMARCHY_FUSED_GEMV=0 without MLX_OMARCHY_GEMV_BALANCED_QKV=1
// and MLX_OMARCHY_FUSED_CHAIN=0 also cover it); on by default whenever
// fused_gemv_enabled is on OR gemv_balanced_qkv_route_enabled is on.
bool fused_gemv_swiglu_route_enabled(const std::string& device_name) {
  if (!fused_chain_enabled()) {
    return false;
  }
  const bool route_active =
      fused_route_enabled("MLX_OMARCHY_FUSED_GEMV", device_name) ||
      gemv_balanced_qkv_route_enabled(device_name);
  if (!route_active) {
    return false;
  }
  const char* sw_env = omarchy::env_str("MLX_OMARCHY_FUSED_GEMV_SWIGLU");
  return sw_env == nullptr || sw_env[0] == '\0' ||
      env_flag("MLX_OMARCHY_FUSED_GEMV_SWIGLU");
}

bool fused_gemv_swiglu_enabled(const Device& device) {
  return fused_gemv_swiglu_route_enabled(device.capabilities().device_name);
}

// MLX_OMARCHY_QMM_VEC_TOKEN_MULTI=1 opts the grouped GEMV planner into
// rows 2..16 (the token column; OFF by default on AGX pending the
// data-dependent divergence bisect).
bool gemv_token_multi_enabled(const Device& device) {
  const char* env = omarchy::env_str("MLX_OMARCHY_QMM_VEC_TOKEN_MULTI");
  return fused_gemv_enabled(device) && env != nullptr && env[0] != '\0' &&
      std::strcmp(env, "0") != 0;
}

bool fused_trio_enabled(const Device& device) {
  return fused_route_enabled(
      "MLX_OMARCHY_FUSED_TRIO", device.capabilities().device_name);
}

bool kv_direct_enabled() {
  return fused_chain_enabled() &&
      (omarchy::env_str("MLX_OMARCHY_KV_DIRECT") == nullptr ||
       env_flag("MLX_OMARCHY_KV_DIRECT"));
}

bool rope_epi_enabled() {
  return fused_chain_enabled() &&
      std::getenv("MLX_OMARCHY_ROPE_EPI") != nullptr &&
      env_flag("MLX_OMARCHY_ROPE_EPI");
}

bool rope_norm_fold_enabled() {
  return fused_chain_enabled() &&
      (omarchy::env_str("MLX_OMARCHY_ROPE_NORM_FOLD") == nullptr ||
       env_flag("MLX_OMARCHY_ROPE_NORM_FOLD"));
}

bool gdn_inproj_ba_fold_enabled() {
  return fused_chain_enabled() &&
      omarchy::env_str("MLX_OMARCHY_GDN_INPROJ_BA_FOLD") != nullptr &&
      env_flag("MLX_OMARCHY_GDN_INPROJ_BA_FOLD");
}

GdnInProjBaPlan* find_gdn_inproj_ba_fold(const array& gdn_out) {
  if (!eager_state) {
    return nullptr;
  }
  auto it = eager_state->gdn_ba_roles.find(gdn_out.id());
  if (it == nullptr) {
    return nullptr;
  }
  auto& fold = eager_state->gdn_ba_folds[*it];
  if (fold.state != GdnInProjBaFold::State::pending) {
    return nullptr;
  }
  return &fold.plan;
}

void commit_gdn_inproj_ba_fold(const array& gdn_out) {
  if (!eager_state) {
    return;
  }
  auto it = eager_state->gdn_ba_roles.find(gdn_out.id());
  if (it == nullptr) {
    return;
  }
  eager_state->gdn_ba_folds[*it].state = GdnInProjBaFold::State::done;
}

void replay_gdn_inproj_ba_fold(const array& gdn_out, const Stream& stream) {
  if (!eager_state) {
    return;
  }
  auto it = eager_state->gdn_ba_roles.find(gdn_out.id());
  if (it == nullptr) {
    return;
  }
  auto& fold = eager_state->gdn_ba_folds[*it];
  if (fold.state != GdnInProjBaFold::State::pending) {
    return;
  }
  fold.state = GdnInProjBaFold::State::failed;
  auto& encoder = get_command_encoder(stream);
  for (const auto& input : fold.a_inputs) {
    encoder.add_temporary(input);
  }
  std::vector<array> a_outs{fold.a_node};
  fold.a_prim->eval_gpu(fold.a_inputs, a_outs);
  for (const auto& input : fold.b_inputs) {
    encoder.add_temporary(input);
  }
  std::vector<array> b_outs{fold.b_node};
  fold.b_prim->eval_gpu(fold.b_inputs, b_outs);
}

bool rope_norm_pair_enabled() {
  const char* env = omarchy::env_str("MLX_OMARCHY_ROPE_NORM_PAIR");
  return fused_chain_enabled() && env != nullptr && env[0] != '\0' &&
      std::strcmp(env, "0") != 0;
}

bool sdpa_online_rope_enabled(
    const CapabilityReport& caps,
    uint32_t k_len,
    int head_dim,
    bool has_outgate) {
  if (!fused_chain_enabled() || (head_dim != 128 && head_dim != 256)) {
    return false;
  }
  const uint32_t min_shared = head_dim == 256 ? 17152u : 16896u;
  constexpr VkSubgroupFeatureFlags kOnlineRopeSubgroupFeatures =
      VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
      VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
      VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
  const bool device_ok = caps.subgroup_size == 32u &&
      caps.max_compute_work_group_invocations >= 1024u &&
      caps.max_compute_work_group_size[0] >= 1024u &&
      caps.max_compute_shared_memory_size >= min_shared &&
      (caps.subgroup_operations & kOnlineRopeSubgroupFeatures) ==
          kOnlineRopeSubgroupFeatures;
  const char* env = omarchy::env_str("MLX_OMARCHY_SDPA_ONLINE_ROPE");
  if (env != nullptr && env[0] != '\0') {
    return std::strcmp(env, "0") != 0 && device_ok;
  }
  const char* online_env = omarchy::env_str("MLX_OMARCHY_SDPA_ONLINE");
  if (online_env != nullptr && std::strcmp(online_env, "0") == 0) {
    return false;
  }
  if (!device_ok) {
    return false;
  }
  return sdpa_online_rope_default(caps.device_name, k_len, head_dim, has_outgate);
}

static bool resolve_sdpa_rope_head_stride(
    const array& raw,
    const std::optional<array>& view,
    const std::shared_ptr<Primitive>& view_prim,
    int expected_heads,
    int expected_head_dim,
    uint32_t& head_stride_out) {
  if (raw.data_shared_ptr() == nullptr || raw.dtype() != bfloat16 ||
      raw.offset() % raw.itemsize() != 0) {
    return false;
  }
  auto contiguous_strides = [](const Shape& shape) {
    Strides strides(shape.size(), 0);
    strides.back() = 1;
    for (int ax = static_cast<int>(shape.size()) - 2; ax >= 0; --ax) {
      strides[ax] = strides[ax + 1] * shape[ax + 1];
    }
    return strides;
  };
  Strides strides;
  Shape shape;
  if (view.has_value()) {
    if (view->offset() % view->itemsize() != 0 ||
        view->shape(-1) != raw.shape(-1)) {
      return false;
    }
    shape = view->shape();
    if (view_prim && typeid(*view_prim) == typeid(Transpose)) {
      const auto& axes = static_cast<const Transpose&>(*view_prim).state();
      if (axes.size() != static_cast<size_t>(view->ndim()) ||
          raw.ndim() != static_cast<int>(axes.size()) ||
          raw.strides().back() != 1 ||
          axes.back() != static_cast<int>(axes.size()) - 1) {
        return false;
      }
      const Strides& parent = raw.strides();
      strides = Strides(axes.size(), 0);
      for (size_t ax = 0; ax < axes.size(); ++ax) {
        if (axes[ax] < 0 || static_cast<size_t>(axes[ax]) >= parent.size()) {
          return false;
        }
        strides[ax] = parent[axes[ax]];
      }
    } else {
      if (!raw.flags().row_contiguous) {
        return false;
      }
      strides = contiguous_strides(view->shape());
    }
  } else {
    shape = raw.shape();
    strides = raw.strides();
  }
  if (shape.size() != 4 || shape[0] != 1 || shape[1] != expected_heads ||
      shape[2] != 1 || shape[3] != expected_head_dim || strides[3] != 1 ||
      strides[1] < 0) {
    return false;
  }
  head_stride_out = static_cast<uint32_t>(strides[1]);
  return true;
}

static void replay_sdpa_rope_side(
    array& rope_node,
    const std::shared_ptr<Primitive>& rope_prim,
    const std::vector<array>& rope_inputs,
    int norm_fold_idx,
    const Stream& stream) {
  if (rope_node.data_shared_ptr() != nullptr) {
    return;
  }
  auto& encoder = get_command_encoder(stream);
  if (norm_fold_idx >= 0 && eager_state != nullptr) {
    auto& fold = eager_state->rope_norms[norm_fold_idx];
    if (fold.state != RopeNormFold::State::done) {
      fold.state = RopeNormFold::State::failed;
      encoder.add_temporary(fold.x);
      encoder.add_temporary(fold.w);
      std::vector<array> norm_in{fold.x, fold.w};
      std::vector<array> norm_out{fold.norm};
      fold.norm_prim->eval_gpu(norm_in, norm_out);
      if (fold.view.has_value()) {
        encoder.add_temporary(fold.norm);
        std::vector<array> view_in{fold.norm};
        std::vector<array> view_out{*fold.view};
        fold.view_prim->eval_gpu(view_in, view_out);
      }
    }
  }
  for (const auto& in : rope_inputs) {
    encoder.add_temporary(in);
  }
  std::vector<array> outs{rope_node};
  rope_prim->eval_gpu(rope_inputs, outs);
}

const SdpaOnlineRopeCtx* find_sdpa_online_rope(
    const array& out,
    const Stream& stream) {
  if (!eager_state) {
    return nullptr;
  }
  auto it = eager_state->sdpa_online_rope_roles.find(out.id());
  if (it == eager_state->sdpa_online_rope_roles.end()) {
    return nullptr;
  }
  auto& plan = eager_state->sdpa_online_rope_plans[it->second];
  if (plan.state != SdpaOnlineRopePlan::State::k_bound) {
    return nullptr;
  }
  auto& pair = eager_state->slice_update_pairs[plan.pair_index];
  uint32_t q_hs = 0;
  uint32_t k_hs = 0;
  const bool weights_ok = !plan.has_norm ||
      (plan.q_weight.has_value() && plan.k_weight.has_value() &&
       plan.q_weight->data_shared_ptr() != nullptr &&
       plan.k_weight->data_shared_ptr() != nullptr &&
       plan.q_weight->offset() % plan.q_weight->itemsize() == 0 &&
       plan.k_weight->offset() % plan.k_weight->itemsize() == 0);
  const bool off_ok = plan.offset.data_shared_ptr() != nullptr &&
      plan.offset.offset() % plan.offset.itemsize() == 0;
  const int expected_hd = plan.sdpa.shape(3);
  if (!pair.windows[0].has_value() ||
      pair.windows[0]->node.data_shared_ptr() == nullptr || !weights_ok ||
      !off_ok ||
      !resolve_sdpa_rope_head_stride(
          plan.q_raw,
          plan.q_view,
          plan.q_view_prim,
          plan.sdpa.shape(1),
          expected_hd,
          q_hs) ||
      !resolve_sdpa_rope_head_stride(
          plan.k_raw,
          plan.k_view,
          plan.k_view_prim,
          plan.k_rope.shape(1),
          expected_hd,
          k_hs)) {
    abort_sdpa_online_rope(out, stream);
    return nullptr;
  }
  plan.ctx = SdpaOnlineRopeCtx{
      &plan.q_raw,
      &plan.k_raw,
      &plan.offset,
      plan.has_norm ? &*plan.q_weight : nullptr,
      plan.has_norm ? &*plan.k_weight : nullptr,
      &*pair.windows[0],
      q_hs,
      k_hs,
      plan.has_norm,
      plan.eps,
      plan.base,
      plan.scale,
      plan.dims};
  return &plan.ctx;
}

void commit_sdpa_online_rope(const array& out) {
  if (!eager_state) {
    return;
  }
  auto it = eager_state->sdpa_online_rope_roles.find(out.id());
  if (it == eager_state->sdpa_online_rope_roles.end()) {
    return;
  }
  auto& plan = eager_state->sdpa_online_rope_plans[it->second];
  plan.state = SdpaOnlineRopePlan::State::done;
  if (plan.q_norm_fold >= 0) {
    eager_state->rope_norms[plan.q_norm_fold].state = RopeNormFold::State::done;
  }
  if (plan.k_norm_fold >= 0) {
    eager_state->rope_norms[plan.k_norm_fold].state = RopeNormFold::State::done;
  }
  if (omarchy::env_str("MLX_OMARCHY_SDPA_ROPE_TRACE") != nullptr) {
    static int commit_trace = 0;
    if (commit_trace < 4) {
      ++commit_trace;
      std::fprintf(stderr, "[sdpa-rope] committed=%d\n", commit_trace);
    }
  }
}

void abort_sdpa_online_rope(const array& out, const Stream& stream) {
  if (!eager_state) {
    return;
  }
  auto it = eager_state->sdpa_online_rope_roles.find(out.id());
  if (it == eager_state->sdpa_online_rope_roles.end()) {
    return;
  }
  auto& plan = eager_state->sdpa_online_rope_plans[it->second];
  if (plan.state == SdpaOnlineRopePlan::State::done ||
      plan.state == SdpaOnlineRopePlan::State::failed) {
    return;
  }
  if (omarchy::env_str("MLX_OMARCHY_SDPA_ROPE_TRACE") != nullptr) {
    static int abort_trace = 0;
    if (abort_trace < 4) {
      ++abort_trace;
      std::fprintf(stderr, "[sdpa-rope] aborted=%d\n", abort_trace);
    }
  }
  plan.state = SdpaOnlineRopePlan::State::failed;
  auto& pair = eager_state->slice_update_pairs[plan.pair_index];
  // k_su already bound window.node to k_cache storage: temporarily re-arm
  // pair.state = pending so k_rope's replay writes straight into window.node.
  if (pair.direct && pair.windows[0].has_value() &&
      pair.windows[0]->node.data_shared_ptr() != nullptr) {
    pair.state = SliceUpdatePair::State::pending;
  }
  if (plan.q_skipped) {
    replay_sdpa_rope_side(
        plan.q_rope, plan.q_rope_prim, plan.q_rope_inputs, plan.q_norm_fold, stream);
  }
  if (plan.k_skipped) {
    replay_sdpa_rope_side(
        plan.k_rope, plan.k_rope_prim, plan.k_rope_inputs, plan.k_norm_fold, stream);
  }
  if (pair.direct) {
    pair.state = SliceUpdatePair::State::done;
  }
}

KvDirectWindow* find_rope_kv_redirect(const array& out) {
  if (!eager_state) {
    return nullptr;
  }
  auto it = eager_state->rope_redirect_roles.find(out.id());
  if (it == nullptr) {
    return nullptr;
  }
  auto& pair = eager_state->slice_update_pairs[*it];
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
  if (it == nullptr) {
    return;
  }
  auto& pair = eager_state->slice_update_pairs[*it];
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

const array* find_sdpa_transpose_out(const array& out) {
  if (!eager_state) {
    return nullptr;
  }
  auto it = eager_state->sdpa_transpose_plans.find(out.id());
  if (it == eager_state->sdpa_transpose_plans.end()) {
    return nullptr;
  }
  auto& fold = eager_state->sdpa_transpose_folds[it->second];
  return fold.fired ? nullptr : &fold.reshape;
}

void commit_sdpa_transpose_out(const array& out) {
  if (!eager_state) {
    return;
  }
  auto it = eager_state->sdpa_transpose_plans.find(out.id());
  if (it != eager_state->sdpa_transpose_plans.end()) {
    eager_state->sdpa_transpose_folds[it->second].fired = true;
  }
}

bool sdpa_outgate_enabled(const CapabilityReport& caps) {
  const char* v = omarchy::env_str("MLX_OMARCHY_SDPA_OUTGATE");
  if (v != nullptr && *v != '\0') {
    return !(v[0] == '0' && v[1] == '\0');
  }
  return sdpa_outgate_default(caps.device_name);
}

const SdpaOutgatePlan* find_sdpa_outgate(const array& out) {
  if (!eager_state) {
    return nullptr;
  }
  auto it = eager_state->sdpa_outgate_plans.find(out.id());
  if (it == eager_state->sdpa_outgate_plans.end()) {
    return nullptr;
  }
  auto& fold = eager_state->sdpa_outgate_folds[it->second];
  return fold.fired ? nullptr : &fold.plan;
}

void commit_sdpa_outgate(const array& out) {
  if (!eager_state) {
    return;
  }
  auto it = eager_state->sdpa_outgate_plans.find(out.id());
  if (it != eager_state->sdpa_outgate_plans.end()) {
    eager_state->sdpa_outgate_folds[it->second].fired = true;
  }
}

void bind_kv_window_storage(KvDirectWindow& window, const Stream& stream) {
  if (window.node.data_shared_ptr() != nullptr) {
    auto& encoder = get_command_encoder(stream);
    encoder.add_temporary(window.node);
    encoder.add_temporary(window.base);
    return;
  }
  if (window.in_place) {
    window.node.copy_shared_buffer(
        window.base,
        window.base.strides(),
        window.base.flags(),
        window.base.data_size());
  } else {
    window.node.set_data(allocator().malloc(window.node.nbytes()));
    copy_gpu(
        window.base,
        window.node,
        window.base.flags().contiguous ? CopyType::Vector : CopyType::General,
        stream);
  }
  auto& encoder = get_command_encoder(stream);
  encoder.add_temporary(window.node);
  encoder.add_temporary(window.base);
}

// Runs the fused FastRopeNorm eval on the tape rope node itself, so a
// claimed KV-direct window still binds inside eval_gpu. The norm's
// inputs carry the pre-norm tensor and weight; the rope node supplies
// the offset. Returns false when a runtime contract check fails; the
// caller then runs the norm and rope on their ordinary paths.
bool dispatch_rope_norm_fold(
    array& rope_node,
    const RopeNormFold& fold,
    const Stream& stream) {
  const auto& rp = static_cast<const fast::RoPE&>(rope_node.primitive());
  const array& x = fold.x;
  const array& w = fold.w;
  const array& off = rope_node.inputs()[1];
  const auto& view = fold.view;
  // Lazy view descs carry contiguous strides; the real strides only appear
  // at eval. Compose them the way the sum fold does: Transpose permutes the
  // parent's strides, and a Reshape over a contiguous parent is the
  // contiguous strides of the new shape.
  auto contiguous_strides = [](const Shape& shape) {
    Strides strides(shape.size(), 0);
    strides.back() = 1;
    for (int ax = static_cast<int>(shape.size()) - 2; ax >= 0; --ax) {
      strides[ax] = strides[ax + 1] * shape[ax + 1];
    }
    return strides;
  };
  Strides src_strides;
  if (view.has_value()) {
    Strides parent = contiguous_strides(x.shape());
    if (typeid(*fold.view_prim) == typeid(Transpose)) {
      const auto& axes =
          static_cast<const Transpose&>(*fold.view_prim).state();
      if (axes.size() != static_cast<size_t>(view->ndim())) {
        return false;
      }
      src_strides = Strides(axes.size(), 0);
      for (size_t ax = 0; ax < axes.size(); ++ax) {
        if (axes[ax] < 0 ||
            static_cast<size_t>(axes[ax]) >= parent.size()) {
          return false;
        }
        src_strides[ax] = parent[axes[ax]];
      }
    } else {
      src_strides = contiguous_strides(view->shape());
    }
  } else {
    src_strides = x.strides();
  }
  const int d = x.ndim() > 0 ? x.shape(-1) : 0;
  if (x.data_shared_ptr() == nullptr || w.data_shared_ptr() == nullptr ||
      off.data_shared_ptr() == nullptr || x.dtype() != bfloat16 ||
      w.dtype() != bfloat16 || rope_node.dtype() != bfloat16 ||
      w.size() != static_cast<size_t>(d) || d <= 0 || d > 256 ||
      (d % 2) != 0 || src_strides.back() != 1 ||
      (view.has_value() &&
       (!x.flags().row_contiguous || x.size() != x.data_size() ||
        view->offset() % view->itemsize() != 0 ||
        view->shape(-1) != x.shape(-1)))) {
    static int fold_run_trace = 0;
    if (omarchy::env_str("MLX_OMARCHY_ROPE_NORM_TRACE") != nullptr &&
        fold_run_trace < 4) {
      ++fold_run_trace;
      std::fprintf(
          stderr,
          "[rope-norm] dispatch refuse: xset=%d wset=%d offset=%d xdt=%d "
          "wdt=%d odt=%d d=%d sback=%lld\n",
          static_cast<int>(x.data_shared_ptr() != nullptr),
          static_cast<int>(w.data_shared_ptr() != nullptr),
          static_cast<int>(off.data_shared_ptr() != nullptr),
          static_cast<int>(x.dtype().val()),
          static_cast<int>(w.dtype().val()),
          static_cast<int>(rope_node.dtype().val()),
          d,
          static_cast<long long>(x.ndim() > 0 ? x.strides().back() : -1));
    }
    return false;
  }
  const auto& st = rp.state();
  // The view's layout replays over the norm's contiguous input buffer:
  // per-row RMSNorm and the view commute, so rope(view(norm(x))) ==
  // rope_rms_norm(view-of-x).
  const array* src = &x;
  std::optional<array> view_of_x;
  if (view.has_value()) {
    view_of_x = array(view->shape(), x.dtype(), nullptr, {});
    // The flags come from the same computation Transpose::eval runs over
    // the real strides, with the contiguous norm output as the parent.
    array::Flags vflags{};
    vflags.contiguous = true;
    vflags.row_contiguous = true;
    vflags.col_contiguous = true;
    int64_t f_stride = 1;
    int64_t b_stride = 1;
    for (int i = 0, ri = view->ndim() - 1; i < view->ndim(); ++i, --ri) {
      vflags.col_contiguous &=
          (src_strides[i] == f_stride || view->shape(i) == 1);
      f_stride *= view->shape(i);
      vflags.row_contiguous &=
          (src_strides[ri] == b_stride || view->shape(ri) == 1);
      b_stride *= view->shape(ri);
    }
    // The view node never evaluated, so its data_size stayed zero; the
    // span over x's row-major buffer is x's own (dense, same elements).
    view_of_x->copy_shared_buffer(
        x, src_strides, vflags, x.data_size(),
        view->offset() / view->itemsize());
    // input_ready refuses an unscheduled input; the buffer is resident.
    view_of_x->set_status(array::Status::available);
    src = &*view_of_x;
  }
  array fused = fast::rope_rms_norm(
      *src,
      std::get<1>(st),
      w,
      fold.eps,
      std::get<2>(st),
      std::get<3>(st),
      std::get<4>(st),
      off,
      stream);
  std::vector<array> inputs{*src, off, w};
  std::vector<array> outputs{rope_node};
  // The eval loop pins the tape rope's inputs (the skipped view and
  // offset); the fused dispatch actually reads x's buffer and the
  // weight, so pin them to this submission.
  auto& encoder = get_command_encoder(stream);
  encoder.add_temporary(*src);
  encoder.add_temporary(off);
  encoder.add_temporary(w);
  fused.primitive().eval_gpu(inputs, outputs);
  return true;
}

// One gather-dequant dispatch for a planned embedding prologue fold:
// binds the token-id buffer plus the three source tables and writes
// the dequantize output. ids is read on the GPU, so the index value
// never crosses the host. Returns false on a contract check; the
// caller then replays the ordinary gathers/views and lets the
// dequantize run normally.
bool dispatch_prologue_fold(
    array& node,
    PrologueFold& fold,
    const Stream& stream) {
  const array& ids = fold.ids;
  const array& w = fold.w;
  const array& scales = fold.scales;
  const array& biases = fold.biases;
  if (ids.data_shared_ptr() == nullptr || w.data_shared_ptr() == nullptr ||
      scales.data_shared_ptr() == nullptr ||
      biases.data_shared_ptr() == nullptr ||
      !ids.flags().row_contiguous || !w.flags().row_contiguous ||
      !scales.flags().row_contiguous || !biases.flags().row_contiguous ||
      ids.offset() % ids.itemsize() != 0 ||
      w.offset() % w.itemsize() != 0 ||
      scales.offset() % scales.itemsize() != 0 ||
      biases.offset() % biases.itemsize() != 0) {
    return false;
  }
  const uint64_t total = fold.nrows * fold.words_per_row;
  const uint64_t max32 = std::numeric_limits<uint32_t>::max();
  if (total > max32 || fold.bits > max32 || w.size() > max32 ||
      scales.size() > max32 || fold.group_size > max32 ||
      node.size() > max32 || w.offset() / w.itemsize() > max32 ||
      scales.offset() / scales.itemsize() > max32 ||
      biases.size() > max32 || biases.offset() / biases.itemsize() > max32 ||
      w.shape(0) > max32 || fold.words_per_row > max32 ||
      fold.groups_per_row > max32 ||
      ids.offset() / ids.itemsize() > max32) {
    return false;
  }
  node.set_data(allocator().malloc(node.nbytes()));
  if (node.size() == 0) {
    return true;
  }
  auto& encoder = get_command_encoder(stream);
  encoder.add_temporary(ids);
  encoder.add_temporary(w);
  encoder.add_temporary(scales);
  encoder.add_temporary(biases);
  // Fused GatherDequantBF16 + FastRmsNormAppleBF16 path (1 WG, 256 threads).
  if (fold.norm_out && fold.norm_weight &&
      fold.norm_weight->data_shared_ptr() != nullptr &&
      fold.norm_weight->flags().row_contiguous &&
      fold.norm_weight->offset() % fold.norm_weight->itemsize() == 0 &&
      fold.norm_out->size() == node.size()) {
    const auto& caps = encoder.device().capabilities();
    constexpr VkSubgroupFeatureFlags kPnSgOps =
        VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    if (caps.subgroup_size == 32u &&
        (caps.subgroup_operations & kPnSgOps) == kPnSgOps &&
        caps.max_compute_work_group_invocations >= 256u &&
        caps.max_compute_work_group_size[0] >= 256u) {
      array& norm_out = *fold.norm_out;
      const array& nw = *fold.norm_weight;
      norm_out.set_data(allocator().malloc(norm_out.nbytes()));
      encoder.add_temporary(nw);
      encoder.add_temporary(norm_out);
      ComputeParams params;
      params.count = static_cast<uint32_t>(fold.words_per_row);
      params.operation = static_cast<uint32_t>(fold.bits);
      params.lhs_size = static_cast<uint32_t>(nw.size());
      params.rhs_size = static_cast<uint32_t>(nw.offset() / nw.itemsize());
      params.reduce_size = static_cast<uint32_t>(node.size());
      params.output_size = static_cast<uint32_t>(fold.group_size);
      params.lhs_offset = static_cast<uint32_t>(w.offset() / w.itemsize());
      params.rhs_offset =
          static_cast<uint32_t>(scales.offset() / scales.itemsize());
      params.output_offset = 0u;
      params.aux_size = 0u;
      params.aux_offset =
          static_cast<uint32_t>(biases.offset() / biases.itemsize());
      params.matrix_m = static_cast<uint32_t>(w.shape(0));
      params.matrix_n = static_cast<uint32_t>(fold.words_per_row);
      params.matrix_k = static_cast<uint32_t>(fold.groups_per_row);
      params.flags = ids.dtype() == uint32 ? 1u : 0u;
      params.alpha = fold.norm_eps;
      params.dims = static_cast<uint32_t>(ids.offset() / ids.itemsize());
      std::array<ComputeBinding, 7> bindings{
          binding(ids),
          binding(w),
          binding(scales),
          binding(biases),
          binding(node),
          binding(nw),
          binding(norm_out)};
      encoder.dispatch_compute(
          ComputeKernel::GatherDequantNormBF16, bindings, params, 1u, 1u, 1u);
      fold.norm_done = true;
      return true;
    }
  }
  ComputeParams params;
  params.count = static_cast<uint32_t>(fold.nrows * fold.words_per_row);
  params.operation = static_cast<uint32_t>(fold.bits);
  params.lhs_size = static_cast<uint32_t>(w.size());
  params.rhs_size = static_cast<uint32_t>(scales.size());
  params.reduce_size = static_cast<uint32_t>(fold.group_size);
  params.output_size = static_cast<uint32_t>(node.size());
  params.lhs_offset =
      static_cast<uint32_t>(w.offset() / w.itemsize());
  params.rhs_offset =
      static_cast<uint32_t>(scales.offset() / scales.itemsize());
  params.aux_size = static_cast<uint32_t>(biases.size());
  params.aux_offset =
      static_cast<uint32_t>(biases.offset() / biases.itemsize());
  params.output_offset = 0;
  params.matrix_m = static_cast<uint32_t>(w.shape(0));
  params.matrix_n = static_cast<uint32_t>(fold.words_per_row);
  params.matrix_k = static_cast<uint32_t>(fold.groups_per_row);
  params.flags = ids.dtype() == uint32 ? 1u : 0u;
  params.dims =
      static_cast<uint32_t>(ids.offset() / ids.itemsize());
  std::array<ComputeBinding, 5> bindings{
      binding(ids), binding(w), binding(scales), binding(biases),
      binding(node)};
  const auto kernel = node.dtype() == float16
      ? ComputeKernel::GatherDequantF16
      : node.dtype() == bfloat16 ? ComputeKernel::GatherDequantBF16
                                 : ComputeKernel::GatherDequantF32;
  encoder.dispatch_compute(
      kernel, bindings, params, compute_dispatch_group_count(params.count));
  return true;
}

bool try_eval_eager_fusion(array& node, const Stream& stream) {
  static const bool fold_timing =
      std::getenv("MLX_OMARCHY_FOLD_TIMING") != nullptr;
  struct DispatchTimer {
    bool on;
    std::chrono::steady_clock::time_point t0;
    explicit DispatchTimer(bool enabled)
        : on(enabled),
          t0(enabled ? std::chrono::steady_clock::now()
                     : std::chrono::steady_clock::time_point{}) {}
    ~DispatchTimer() {
      if (on) {
        eager_dispatch_us +=
            std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0)
                .count();
        ++eager_dispatch_calls;
      }
    }
  } dispatch_timer(fold_timing);
  if (!eager_state) {
    return false;
  }
  if (auto skip = eager_state->res_norm_skip_roles.find(node.id());
      skip != nullptr) {
    return eager_state->gemv_groups[*skip].res_norm_done;
  }
  if (auto skip = eager_state->gdn_ba_skip_roles.find(node.id());
      skip != nullptr) {
    return eager_state->gdn_ba_folds[*skip].state !=
        GdnInProjBaFold::State::failed;
  }
  if (auto skip = eager_state->sdpa_transpose_skip_roles.find(node.id());
      skip != eager_state->sdpa_transpose_skip_roles.end()) {
    return eager_state->sdpa_transpose_folds[skip->second].fired;
  }
  if (auto skip = eager_state->sdpa_outgate_skip_roles.find(node.id());
      skip != eager_state->sdpa_outgate_skip_roles.end()) {
    return eager_state->sdpa_outgate_folds[skip->second].fired;
  }
  if (auto skip = eager_state->prologue_norm_skip_roles.find(node.id());
      skip != nullptr) {
    return eager_state->prologue_folds[*skip].norm_done;
  }
  if (auto skip = eager_state->prologue_skip_roles.find(node.id());
      skip != nullptr) {
    // The fold's dispatch owns this node while pending or done; on a
    // failed fold the fire arm replays every skipped node before the
    // dequantize runs, so nothing ever falls through unwritten.
    return eager_state->prologue_folds[*skip].state !=
        PrologueFold::State::failed;
  }
  if (auto fit = eager_state->prologue_fold_roles.find(node.id());
      fit != nullptr) {
    auto& fold = eager_state->prologue_folds[*fit];
    if (fold.state == PrologueFold::State::done) {
      return true;
    }
    if (fold.state == PrologueFold::State::failed) {
      return false;
    }
    if (fold.quant.status() == array::Status::evaluated &&
        fold.quant.data_shared_ptr() != nullptr) {
      // A nested settle scope already fired this fold; re-dispatching
      // would only rewrite the output.
      fold.state = PrologueFold::State::done;
      return true;
    }
    fold.state = dispatch_prologue_fold(node, fold, stream)
        ? PrologueFold::State::done
        : PrologueFold::State::failed;
    if (fold.state == PrologueFold::State::failed) {
      // Replay the skipped gathers and views in dependency order so
      // the ordinary dequantize path below reads settled inputs.
      auto& encoder = get_command_encoder(stream);
      for (auto& step : fold.replay) {
        for (const auto& input : step.inputs) {
          encoder.add_temporary(input);
        }
        std::vector<array> outs{step.node};
        step.prim->eval_gpu(step.inputs, outs);
      }
      return false;
    }
    return true;
  }
  if (auto skip = eager_state->rope_norm_skip_roles.find(node.id());
      skip != nullptr) {
    // The rope-side fold owns this norm's dispatch; a failed fold runs
    // it in place from the rope arm below, so skip only while the fold
    // can still fire.
    return eager_state->rope_norms[*skip].state !=
        RopeNormFold::State::failed;
  }
  if (auto sr_skip = eager_state->sdpa_online_rope_skip_roles.find(node.id());
      sr_skip != eager_state->sdpa_online_rope_skip_roles.end()) {
    auto& plan = eager_state->sdpa_online_rope_plans[sr_skip->second];
    if (plan.state != SdpaOnlineRopePlan::State::failed ||
        node.data_shared_ptr() != nullptr) {
      if (node.id() == plan.q_rope.id()) {
        plan.q_skipped = true;
      } else if (node.id() == plan.k_rope.id()) {
        plan.k_skipped = true;
      }
      return true;
    }
  }
  if (auto sr_su = eager_state->sdpa_online_rope_su_roles.find(node.id());
      sr_su != eager_state->sdpa_online_rope_su_roles.end()) {
    auto& plan = eager_state->sdpa_online_rope_plans[sr_su->second];
    if (plan.state == SdpaOnlineRopePlan::State::k_bound ||
        plan.state == SdpaOnlineRopePlan::State::done) {
      return true;
    }
    if (plan.state == SdpaOnlineRopePlan::State::pending) {
      auto& pair = eager_state->slice_update_pairs[plan.pair_index];
      // Only check KV-window readiness here; q_raw/k_raw may live on sibling
      // tape branches not yet evaluated when v_su or k_su is visited. Their
      // buffers and head strides are validated at SDPA time in
      // find_sdpa_online_rope().
      if (pair.direct && pair.values_committed && pair.windows[0].has_value() &&
          pair.windows[1].has_value() &&
          pair.windows[1]->node.data_shared_ptr() != nullptr &&
          pair.windows[0]->base.data_shared_ptr() != nullptr) {
        auto& win = *pair.windows[0];
        bind_kv_window_storage(win, stream);
        win.prebound = true;
        pair.state = SliceUpdatePair::State::done;
        plan.state = SdpaOnlineRopePlan::State::k_bound;
        return true;
      }
      // Runtime refusal before binding k_su: only replay ropes that were
      // already skipped (so their inputs are guaranteed evaluated); unvisited
      // ropes will execute normally when reached on the tape.
      plan.state = SdpaOnlineRopePlan::State::failed;
      pair.direct = false;
      pair.state = SliceUpdatePair::State::pending;
      if (plan.q_skipped) {
        replay_sdpa_rope_side(
            plan.q_rope, plan.q_rope_prim, plan.q_rope_inputs, plan.q_norm_fold, stream);
      }
      if (plan.k_skipped) {
        replay_sdpa_rope_side(
            plan.k_rope, plan.k_rope_prim, plan.k_rope_inputs, plan.k_norm_fold, stream);
      }
    }
  }
  if (auto rope_pair = eager_state->rope_pair_roles.find(node.id());
      rope_pair != nullptr) {
    auto& pair = eager_state->rope_pairs[*rope_pair];
    if (pair.state == RopePair::State::done) {
      return true;
    }
    if (pair.state == RopePair::State::pending) {
      if (pair.nodes[0].status() == array::Status::evaluated &&
          pair.nodes[1].status() == array::Status::evaluated) {
        // A settle pass already evaluated both members: the pair can
        // pop inside a nested settle scope before the outer tape
        // reaches it, and each scope plans the same pairs. One
        // dispatch wrote both outputs; re-firing only rewrites them.
        pair.state = RopePair::State::done;
        return true;
      }
      // A fold-carrying pair passes each side's stashed pre-norm input,
      // weight and eps; a successful dispatch owns the fold's norm
      // math, so the folds are consumed (their claimed norms stay
      // skipped).
      std::optional<RopePairNorm> pair_norms[2];
      std::array<const RopePairNorm*, 2> norm_ptrs{nullptr, nullptr};
      for (int s = 0; s < 2; ++s) {
        if (pair.norm_folds[s] >= 0) {
          const auto& f = eager_state->rope_norms[pair.norm_folds[s]];
          pair_norms[s].emplace(
              RopePairNorm{f.x, f.w, f.eps, f.view, f.view_prim});
          norm_ptrs[s] = &*pair_norms[s];
        }
      }
      pair.state = RopePair::State::dispatching;
      bool fired = dispatch_rope_pair(pair.nodes, norm_ptrs, stream);
      pair.state = fired ? RopePair::State::done : RopePair::State::failed;
      if (fired) {
        for (int s = 0; s < 2; ++s) {
          if (pair.norm_folds[s] >= 0) {
            eager_state->rope_norms[pair.norm_folds[s]].state =
                RopeNormFold::State::done;
          }
        }
        return true;
      }
      static int trio_run_trace = 0;
      if (omarchy::env_str("MLX_OMARCHY_TRIO_TRACE") != nullptr &&
          trio_run_trace < 3) {
        ++trio_run_trace;
        std::fprintf(
            stderr,
            "[trio-run] pair un-planned: runtime contract refused\n");
      }
    }
    // The pair is failed (refused now or on the other member's eval) or
    // dispatching (a nested eval reached a member mid-dispatch). A
    // fold-claimed rope still has its fold below: it replays the
    // claimed norm and view so the ordinary rope eval reads settled
    // inputs. A plain rope has no second coverage, so refuse it.
    if (eager_state->rope_norm_roles.find(node.id()) == nullptr) {
      return false;
    }
  }
  if (auto fold_it = eager_state->rope_norm_roles.find(node.id());
      fold_it != nullptr) {
    auto& fold = eager_state->rope_norms[*fold_it];
    if (fold.state == RopeNormFold::State::done) {
      return true;
    }
    if (fold.state == RopeNormFold::State::failed) {
      return false;
    }
    fold.state = dispatch_rope_norm_fold(node, fold, stream)
        ? RopeNormFold::State::done
        : RopeNormFold::State::failed;
    if (fold.state == RopeNormFold::State::failed) {
      // The skip arm never ran the norm or the view; dispatch both now
      // so the ordinary rope path below reads a settled input. Their
      // tape nodes are already detached, so use the stashed primitives
      // and inputs.
      auto& encoder = get_command_encoder(stream);
      encoder.add_temporary(fold.x);
      encoder.add_temporary(fold.w);
      std::vector<array> norm_in{fold.x, fold.w};
      std::vector<array> norm_out{fold.norm};
      fold.norm_prim->eval_gpu(norm_in, norm_out);
      if (fold.view.has_value()) {
        encoder.add_temporary(fold.norm);
        std::vector<array> view_in{fold.norm};
        std::vector<array> view_out{*fold.view};
        fold.view_prim->eval_gpu(view_in, view_out);
      }
      return false;
    }
    return true;
  }
  if (auto add_norm = eager_state->add_norm_roles.find(node.id());
      add_norm != nullptr) {
    auto& pair = eager_state->add_norm_pairs[*add_norm];
    if (pair.state == AddNormPair::State::done) {
      return true;
    }
    if (pair.state == AddNormPair::State::failed) {
      return false;
    }
    if (pair.nodes[0].status() == array::Status::evaluated &&
        pair.nodes[1].status() == array::Status::evaluated) {
      // A settle pass already evaluated both members; one dispatch
      // wrote both outputs, re-firing would only rewrite them.
      pair.state = AddNormPair::State::done;
      return true;
    }
    bool fired = dispatch_add_norm_pair(pair.nodes, stream);
    pair.state = fired ? AddNormPair::State::done : AddNormPair::State::failed;
    return pair.state == AddNormPair::State::done;
  }
  if (auto rope_pair = eager_state->rope_pair_roles.find(node.id());
      rope_pair != nullptr) {
    auto& pair = eager_state->rope_pairs[*rope_pair];
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
    if (!fired && omarchy::env_str("MLX_OMARCHY_TRIO_TRACE") != nullptr &&
        trio_run_trace < 3) {
      ++trio_run_trace;
      std::fprintf(
          stderr, "[trio-run] pair un-planned: runtime contract refused\n");
    }
    return pair.state == RopePair::State::done;
  }
  if (auto update = eager_state->slice_update_roles.find(node.id());
      update != nullptr) {
    auto& pair = eager_state->slice_update_pairs[*update];
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
      reshape != nullptr) {
    auto& pair = eager_state->slice_update_pairs[*reshape];
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
      dense != nullptr) {
    auto& group = eager_state->dense_gemv_groups[*dense];
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
            w.base.offset() % w.base.itemsize() != 0) {
          abort_kv_direct();
          group.sum_windows.clear();
          break;
        }
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
      if (group.state == DenseGemvGroup::State::failed &&
          !group.sum_windows.empty()) {
        abort_kv_direct();
        group.sum_windows.clear();
      }
    }
    return group.state == DenseGemvGroup::State::done;
  }
  if (auto gemv = eager_state->gemv_roles.find(node.id());
      gemv != nullptr) {
    auto& group = eager_state->gemv_groups[*gemv];
    if (group.state == GemvGroup::State::pending) {
      group.state = dispatch_quantized_gemv_group(
                        group.members,
                        group.swiglu_out ? &*group.swiglu_out : nullptr,
                        stream,
                        group.outgate ? &*group.outgate : nullptr,
                        group.res_norm ? &*group.res_norm : nullptr,
                        &group.res_norm_done,
                        group.rope_epi ? &*group.rope_epi : nullptr,
                        &group.rope_epi_done)
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
