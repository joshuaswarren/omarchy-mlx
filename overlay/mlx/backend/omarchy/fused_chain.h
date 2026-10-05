// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::omarchy {

// Fused elementwise chain: collapses a run of same-shape float
// unary/binary Compiled-tape nodes into ONE compute dispatch. The chain
// interpreter shader (shaders/fused_chain.comp) executes a packed
// instruction program with all intermediates in registers, so a
// k-op chain costs 1 dispatch instead of k. Honeykrisp register
// pressure stays at 8 scalars per invocation.
//
// Any node the chain cannot carry (unsupported op class, shape change,
// exotic broadcast, too many leaves, no tail dependency) makes the
// whole run fall back to the per-node tape path: loud refusal
// semantics are unchanged. bf16 nodes fuse through the chain's own
// bf16 kernels, which round every intermediate to the storage dtype
// exactly like per-node dispatch. The eager SwiGLU planner below keeps
// its own bf16 support with bf16 intermediate rounding.
//
// Fusion defaults on after exact-ID parity and paired performance validation on
// M1 hardware. MLX_OMARCHY_FUSED_CHAIN=0 restores the per-node path.

struct FusedChainImpl;
bool fused_chain_enabled();

class FusedChain {
 public:
  // gate_enabled carries the fused-chain decision, read once per tape
  // evaluation; false makes every try_add refuse and preserves the per-node path.
  explicit FusedChain(bool gate_enabled);
  ~FusedChain();

  FusedChain(const FusedChain&) = delete;
  FusedChain& operator=(const FusedChain&) = delete;
  FusedChain(FusedChain&&);
  FusedChain& operator=(FusedChain&&);

  // Pure op/dtype check (the gate lives in the constructor argument):
  // a float32/float16/bfloat16 fusable unary or binary elementwise primitive.
  static bool can_start(const array& node);

  // Attempts to append `node` with resolved `inputs`. Returns false if
  // the node cannot be carried; the caller then closes the chain before
  // it. `is_tape_output` members may only be the chain tail. Two hard
  // contracts:
  // - an EXTENSION (chain already open) must consume the tail's
  //   register; a no-dependency same-shape sibling is refused so the
  //   interpreter closes the chain and the sibling opens a fresh one
  //   (a non-tail interior member can be consumed from outside the
  //   chain and would never be materialized);
  // - leaves pushed for a rejected member are rolled back, so a
  //   refused add leaves no orphan slots behind.
  // Every refusal that would make a carried chain undispatchable (f16
  // capabilities, leaf bounds, broadcast form) is decided HERE, before
  // a node is accepted.
  bool try_add(
      const array& node,
      const std::vector<array>& inputs,
      bool is_tape_output);

  // Dispatches the accumulated chain (1 or more nodes) as one fused
  // kernel. Returns the fused output as an evaluated, graph-free array
  // (no primitive, no inputs). Returns nullopt only for an empty chain:
  // refusals happen in try_add before acceptance.
  std::optional<array> evaluate(const Stream& stream);

  // Dispatch into the current tail's graph array. Used by eager fusion,
  // where the scheduler owns that array and no replacement descriptor may
  // be substituted for it.
  void evaluate_tail(const Stream& stream);

  // Number of tape nodes currently carried.
  size_t size() const;

  // Tracing-graph id of the chain's last member (size() > 0).
  std::uintptr_t tail_id() const;

  // True when `id` is one of the carried members (interior members have
  // no materialized output; consumers of one must close the chain).
  bool carries(std::uintptr_t id) const;

  // True when the CURRENT tail is a tape output: the caller must close
  // the chain before adding anything else (interior tape outputs would
  // lose their materialized results).
  bool tail_is_tape_output() const;

 private:
  std::unique_ptr<FusedChainImpl> impl_;
};


// One eval_impl graph window. The scope preflights exact eager SwiGLU
// shapes (gate * sigmoid(gate) * up) whose two interior values have one
// consumer, then try_eval_eager_fusion defers those interior dispatches
// and records the three operations as one FusedChain dispatch. With the
// gate off this is an empty, allocation-free plan.
class EagerFusionScope {
 public:
  explicit EagerFusionScope(const std::deque<array>& tape);
  ~EagerFusionScope();

  EagerFusionScope(const EagerFusionScope&) = delete;
  EagerFusionScope& operator=(const EagerFusionScope&) = delete;

 private:
  void* previous_;
};

// Returns true when the primitive was recorded (or deliberately deferred)
// by the active eager-fusion scope; false keeps the ordinary eval_gpu path.
bool try_eval_eager_fusion(array& node, const Stream& stream);

// Producer-direct KV cache write window. When a SliceUpdate pair's new
// rows can be stored by the row's PRODUCER (the RoPE kernel for keys,
// the fused GEMV Add epilogue for values), the merged pair dispatch is
// deleted: the producer writes the updated cache copy's window directly
// and the copy of the bytes never happens. The window is the paste
// region of the SliceUpdate output |node| (a fresh copy of |base|, the
// cache), addressed in elements relative to the node buffer start.
struct KvDirectWindow {
  array node;
  array base;
  uint32_t offset{0};
  uint32_t strides[4]{};
  uint32_t ndim{0};
  // GEMV sum-store column map only: column c of the epilogue sum lands
  // at offset + (c / head_dim) * row_gap + (c % head_dim). Single-batch
  // decode only.
  uint32_t row_gap{0};
  uint32_t head_dim{0};
  // True when the planner proved |base| has no other consumer in the
  // tape: the producer then writes the new rows straight into |base|'s
  // buffer (|node| shares that storage) and the fresh cache copy
  // disappears entirely.
  bool in_place{false};
};

// DecodeFusion: one fused decode GEMV group. Up to kQmmVecMultiWeights
// affine transposed 4-bit/group-64 QuantizedMatmul nodes that read one
// single-row x, each optionally followed by the Add that is its only
// consumer (a bias or residual add), recorded as ONE QmmVecQ4Multi
// dispatch when the first member evaluates. Every member output and
// every Add output is materialized, so retained references stay valid.
struct GemvFusionMember {
  array node;
  std::optional<array> epilogue;
  std::optional<array> addend;
  // Planned producer-direct write of the epilogue sum into the values
  // cache copy; empty keeps the sum in its own buffer.
  std::optional<KvDirectWindow> sum_window;
};

// Out-gate prologue fold: the group's shared x is the output of
// x = Multiply(Sigmoid(gate), |out|) whose only consumer is the group's
// first member. |gate| and |out| are the prologue vectors (element 0,
// whole, dense), |x| the deleted Multiply's output — the kernel
// recomputes it per workgroup with the elementwise arithmetic and
// rounds, and materializes it so retained references stay valid.
struct OutgatePlan {
  array gate;
  array out;
  array x;
  // The tape id of the deleted Sigmoid node: the gate's only allowed
  // reader when the plan attaches.
  std::uintptr_t sigmoid_id;
};


// Validates the group against the kernel contract, allocates every
// output, and records the dispatch. |swiglu_out| plans the SwiGLU
// store epilogue: exactly two epilogue-free members of equal length N
// whose chain is silu(gate) * up; the dispatch computes both dots per
// workgroup and stores only the product into |swiglu_out|, and both
// member outputs alias that buffer (their only readers were the
// deleted swiglu dispatch). |outgate| plans the out-gate prologue:
// the shared x is the deleted Multiply(Sigmoid(gate), out) output, and
// the one dispatch reads gate and out directly, reconstructing x per
// workgroup with the elementwise arithmetic and rounding. Returns false
// having allocated nothing when any member falls outside the contract;
// the caller then lets every node take its ordinary eval_gpu path.
// Defined in primitives.cpp beside QuantizedMatmul::eval_gpu.
bool dispatch_quantized_gemv_group(
    std::vector<GemvFusionMember>& members,
    array* swiglu_out,
    const Stream& stream,
    OutgatePlan* outgate = nullptr);

bool dispatch_dense_gemv_group(
    std::vector<array>& nodes,
    const array& input,
    const std::optional<KvDirectWindow>* sum_windows,
    const Stream& stream);

enum class SliceUpdatePairDispatch : uint8_t { done, not_ready, unsupported };
SliceUpdatePairDispatch dispatch_slice_update_pair(
    std::array<array, 2>& nodes,
    const Stream& stream);

// Producer-direct KV write hooks for the planner's two producers.
// find_rope_kv_redirect returns the planned window for this RoPE output
// while its pair is still pending; commit marks the pair done (both
// producers wrote their windows, the merged pair dispatch is skipped);
// abort un-plans the direct write so the pair falls back to the ordinary
// merged SliceUpdatePair dispatch.
KvDirectWindow* find_rope_kv_redirect(const array& out);
void commit_rope_kv_redirect(const array& out);
void abort_kv_direct();
// Marks the values side of the direct write as stored: the GEMV group
// wrote the sum into the planned window. The keys side requires this
// before it will fire, and the reshaped sum view aliases the window
// once it is set.
void commit_values_kv_write(const array& sum_node);

// MLX_OMARCHY_KV_DIRECT=0 keeps both cache writes on the merged pair
// dispatch (the MLX_OMARCHY_FUSED_CHAIN gate also covers it).
bool kv_direct_enabled();

// MLX_OMARCHY_FUSED_GEMV=0 keeps every QuantizedMatmul, dense BF16
// decode GEMV group, and Add on the per-node path (the
// MLX_OMARCHY_FUSED_CHAIN gate also covers it).
bool fused_gemv_enabled();

// MLX_OMARCHY_OUTGATE_FOLD=0/1 overrides; default off on G13 parts other
// than G13C (see fused_chain.cpp).
bool outgate_fold_enabled();

// MLX_OMARCHY_FUSED_GEMV_SWIGLU=0 keeps the SwiGLU store epilogue
// (the gate/up GEMV group that stores silu(gate) * up directly) off
// (the MLX_OMARCHY_FUSED_GEMV gate also covers it); on by default.
bool fused_gemv_swiglu_enabled();

// Decode trio: MLX_OMARCHY_FUSED_TRIO=0 keeps f16 RMSNorm rows, the
// fused SwiGLU chain dispatch, and RoPE pairs on their standalone
// kernels and paths (the MLX_OMARCHY_FUSED_CHAIN gate also covers it).
bool fused_trio_enabled();

// Validates a planned (query, key) RoPE pair against the FastTrioRopePairF16
// rope-pair contract, allocates the outputs, prepares the keys-side
// producer-direct KV window when one is planned, and records the ONE
// dispatch that rotates both tensors with the near-copied fast_rope
// arithmetic (fast_trio.comp mode 1). Returns false having allocated
// nothing when anything refuses; both nodes then take the ordinary
// RoPE path unchanged. Defined in primitives.cpp beside
// dispatch_quantized_gemv_group.
bool dispatch_rope_pair(std::array<array, 2>& nodes, const Stream& stream);

// Persistent decode MLP tail (H290 -> H294): ONE PersistentTailBF16
// dispatch replacing the decode tail's three dispatches (fast_norm;
// qmm_vec multi fold on [gate, up]; qmm_vec multi add on down). The
// kernel runs the three stages over G tile-strided workgroups with two
// in-kernel grid barriers; G comes from a startup probe of the real
// pipeline's register pressure (HkTurnover formula), cross-checked with
// the measured-good ceilings, and any timeout flag falls back sticky.
// |x| is the norm input and the down Add's addend (residual); |norm_w|
// the RMSNorm weight with |eps|; |norm_out| the x_norm buffer and
// |swiglu_out| the x_mid buffer, both allocated and written by the
// dispatch; |gu_members| is exactly [gate, up] without epilogues or
// windows and |dn_member| the down projection with its Add epilogue
// (addend == *x) and no window. gu_group/dn_group/state are planner
// bookkeeping. Returns false having allocated nothing when any
// contract check, the probe, or the sticky gate refuses; the caller
// then takes the shipped three-dispatch path. Defined in
// primitives.cpp beside dispatch_quantized_gemv_group.
struct PersistentTailPlan {
  PersistentTailPlan(
      array x_,
      array norm_w_,
      float eps_,
      array norm_out_,
      array swiglu_out_,
      std::vector<GemvFusionMember> gu_members_,
      GemvFusionMember dn_member_,
      size_t gu_group_,
      size_t dn_group_)
      : x(std::move(x_)),
        norm_w(std::move(norm_w_)),
        eps(eps_),
        norm_out(std::move(norm_out_)),
        swiglu_out(std::move(swiglu_out_)),
        gu_members(std::move(gu_members_)),
        dn_member(std::move(dn_member_)),
        gu_group(gu_group_),
        dn_group(dn_group_) {}
  array x;
  array norm_w;
  float eps{0.0f};
  array norm_out;
  array swiglu_out;
  std::vector<GemvFusionMember> gu_members;
  GemvFusionMember dn_member;
  size_t gu_group{0};
  size_t dn_group{0};
  enum class State : uint8_t { pending, done, failed } state{State::pending};
};
bool dispatch_persistent_mlp_tail(
    PersistentTailPlan& plan,
    const Stream& stream);

// MLX_OMARCHY_PERSISTENT_MLP: route the decode MLP tail through the one
// persistent dispatch. Default off (shipped three-dispatch path); =0 is
// the kill switch.
bool persistent_mlp_enabled();
} // namespace mlx::core::omarchy