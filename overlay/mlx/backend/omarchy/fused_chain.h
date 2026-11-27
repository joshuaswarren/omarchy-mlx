// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
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
  std::chrono::steady_clock::time_point t0_;
};

// Returns true when the primitive was recorded (or deliberately deferred)
// by the active eager-fusion scope; false keeps the ordinary eval_gpu path.
bool try_eval_eager_fusion(array& node, const Stream& stream);

// MLX_OMARCHY_FOLD_TIMING: thread-local phase accumulators shared with
// eval.cpp's per-node timers; the scope dtor prints and resets them.
// Slots: 0=eval() total, 1=dispatch block, 2=needs_commit bookkeeping,
// 3=encoder.commit in eval, 4=finalize total, 5=eval() node count.
double* eval_phase_slots();

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
  // True when |base| was donatable at plan time (the SliceUpdate held the
  // last handle to it and its storage): the producer then writes the new
  // rows straight into |base|'s buffer (|node| shares that storage) and
  // the fresh cache copy disappears entirely.
  bool in_place{false};
  // True when |node|'s storage was already bound ahead of producer replay
  // (e.g. a fused SDPA-RoPE plan that fell back after binding k_su).
  bool prebound{false};
};

// DecodeFusion: one fused decode GEMV group. Up to kQmmVecMultiWeights
// affine transposed 4-bit/group-64 QuantizedMatmul nodes that read one
// single-row x, each optionally followed by the Add that is its only
// consumer (a bias or residual add), recorded as ONE QmmVecQ4Multi
// dispatch when the first member evaluates. Every member output and
// every Add output is materialized, so retained references stay valid.
// Dual-write strided member output (MLX_OMARCHY_QMM_DUAL_OUT): the
// member's output lives at |offset| inside the whole dense |parent|
// buffer; the dispatch aliases the member node into the parent (the
// swiglu-fold copy_shared_buffer precedent) and stores through
// out_strides[i] + column. Empty keeps today's whole-buffer output.
struct GemvOutView {
  array parent;
  uint32_t offset;
};

struct GemvFusionMember {
  array node;
  std::optional<array> epilogue;
  std::optional<array> addend;
  // Planned producer-direct write of the epilogue sum into the values
  // cache copy; empty keeps the sum in its own buffer.
  std::optional<KvDirectWindow> sum_window;
  // Planned dual-write strided output; empty keeps the whole buffer.
  std::optional<GemvOutView> out_view;
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

// Backward epilogue fold: a 1-member bf16 decode QMM + ResidualAdd group whose
// sum output is consumed by a single-row fast::RMSNorm. The dispatch runs
// the norm inside the last finishing workgroup without an extra launch or
// CDM barrier while keeping the sum materialized for other consumers.
struct ResNormPlan {
  array out;
  array weight;
  float eps;
};

// Forward epilogue fold: a decode group's member outputs feed the two
// fused RMSNorm+RoPE nodes (q, k) of one attention head set. The last
// finishing workgroup of the group's dispatch runs both norms and
// rotations inline: the query rows store into the rope node's own
// fresh buffer, the keys rows store through the slice pair's planned
// producer-direct KV window. Both RoPE nodes and their input view
// chains are deleted, so every contract item is decided at plan time;
// a dispatch-time refusal leaves the ropes unmaterialized and is a
// plan error, never a fallback.
struct RopeEpiSide {
  // The fast::RoPE node (3 inputs: view of member out, offset, norm
  // weight); its eval is deleted. |member_index| picks the group's
  // member whose raw output row is the norm+rope source. |view_chain|
  // holds the Reshape/Transpose nodes between the member output and
  // the rope input, also deleted.
  array rope;
  size_t member_index;
  std::vector<array> view_chain;
  // The rope input view's LOGICAL layout over the member storage,
  // composed at plan time: a lazy view node's desc holds contiguous
  // strides until its own eval writes the real ones via
  // copy_shared_buffer, and the skip-role'd chain never evals, so the
  // desc never carries the view geometry. |view_offset| is the
  // element offset into the member storage (0 for the pure
  // reshape/transpose chains accepted here).
  Shape view_shape;
  Strides view_strides;
  int64_t view_offset = 0;
  // Keys side only: the pair's planned producer-direct window; the
  // epilogue stores the rotated keys rows through it. The query side
  // leaves |window| empty and writes |rope|'s own buffer.
  std::optional<KvDirectWindow> window;
  size_t pair_index;
};

struct RopeEpiPlan {
  RopeEpiSide q;
  RopeEpiSide k;
  // Rope scalar contract shared by both sides, copied out of the
  // primitive state at plan time.
  int dims;
  bool traditional;
  float base;
  float scale;
  float norm_eps;
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
// workgroup with the elementwise arithmetic and rounding. |res_norm|
// plans the trailing fast::RMSNorm epilogue over the single member's
// Add sum; when it runs, |*res_norm_done| is set true. |rope_epi|
// plans the fused RMSNorm+RoPE q/k epilogue over two member outputs;
// when it runs, |*rope_epi_done| is set true. Returns false having
// allocated nothing when any member falls outside the contract; the
// caller then lets every node take its ordinary eval_gpu path.
// Defined in primitives.cpp beside QuantizedMatmul::eval_gpu.
bool dispatch_quantized_gemv_group(
    std::vector<GemvFusionMember>& members,
    array* swiglu_out,
    const Stream& stream,
    OutgatePlan* outgate = nullptr,
    ResNormPlan* res_norm = nullptr,
    bool* res_norm_done = nullptr,
    RopeEpiPlan* rope_epi = nullptr,
    bool* rope_epi_done = nullptr);

bool dispatch_dense_gemv_group(
    std::vector<array>& nodes,
    const array& input,
    std::optional<KvDirectWindow>* sum_windows,
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
// Gives the window's SliceUpdate output its storage before its
// producer's row stores: |base|'s own buffer when the window is in place,
// otherwise a fresh copy of |base| enqueued ahead of the stores. Pins
// both buffers to the open batch.
void bind_kv_window_storage(KvDirectWindow& window, const Stream& stream);

// MLX_OMARCHY_KV_DIRECT=0 keeps both cache writes on the merged pair
// dispatch (the MLX_OMARCHY_FUSED_CHAIN gate also covers it).
bool kv_direct_enabled();

// MLX_OMARCHY_ROPE_NORM_FOLD=0 keeps the RMSNorm -> RoPE fold off (the
// MLX_OMARCHY_FUSED_CHAIN gate also covers it); on by default.
bool rope_norm_fold_enabled();
// MLX_OMARCHY_ROPE_EPI=1 opts the fused RMSNorm+RoPE q/k epilogue into
// the group's Token4 dispatch (off by default until measured).
bool rope_epi_enabled();

// MLX_OMARCHY_ROPE_NORM_PAIR=1 opts the bf16 norm-rope pair scan in
// (off by default: the default-on route measured -0.81% decode
// against the solo FastRopeNorm dispatches it replaces). The f16
// trio pairs are unaffected; the MLX_OMARCHY_FUSED_CHAIN and
// MLX_OMARCHY_FUSED_TRIO gates still cover the whole scan.
bool rope_norm_pair_enabled();

// MLX_OMARCHY_SDPA_ONLINE_ROPE=1 opts the fused Q/K RMSNorm+RoPE+KV-write+
// online-softmax BF16 decode SDPA (HD=128 or HD=256) scan in.
struct CapabilityReport;
bool sdpa_online_rope_enabled(
    const CapabilityReport& caps,
    uint32_t k_len,
    int head_dim = 128);

struct SdpaOnlineRopeCtx {
  const array* q_raw{nullptr};
  const array* k_raw{nullptr};
  const array* offset{nullptr};
  const array* q_weight{nullptr};
  const array* k_weight{nullptr};
  KvDirectWindow* k_window{nullptr};
  uint32_t q_head_stride{0};
  uint32_t k_head_stride{0};
  bool has_norm{false};
  float norm_eps{0.0f};
  float rope_base{0.0f};
  float rope_scale{1.0f};
  int rope_dims{128};
};

// Returns the planned fused Q/K RopeNorm+KV-write+SDPA context when the SDPA
// node has a bound pending plan, or nullptr otherwise.
const SdpaOnlineRopeCtx* find_sdpa_online_rope(
    const array& out,
    const Stream& stream);
void commit_sdpa_online_rope(const array& out);
void abort_sdpa_online_rope(const array& out, const Stream& stream);

// find_sdpa_transpose_out returns the materializing Reshape node a
// decode-rows SDPA should store into directly (the planned
// (1,H,S,D) -> (1,S,H*D) out-transpose fold), or nullptr when the
// output has no planned fold or the fold already fired. The caller
// re-verifies the reshape's shape/stride contract before adopting it.
// commit marks the fold done so the view chain and reshape evals skip.
const array* find_sdpa_transpose_out(const array& out);
void commit_sdpa_transpose_out(const array& out);

// Fused SDPA output-gate epilogue (tail = Reshape(Transpose(SDPA)) *
// Sigmoid(gate), with optional strided gate Reshape bypass): SDPA writes the
// gated product directly into tail, skipping the gate copy, sigmoid, and
// multiply dispatches. Bit-identical to the unfused sequence.
struct SdpaOutgatePlan {
  array tail;
  array gate_src;
};
bool sdpa_outgate_enabled();
const SdpaOutgatePlan* find_sdpa_outgate(const array& out);
void commit_sdpa_outgate(const array& out);

// Fused in_proj_a + in_proj_b quantized projections folded into idle subgroups
// of the coalesced GdnConvDeltaUpdate kernel during Phase 1 (conv+SiLU+norm).
// Bit-exact to standalone QmmVecQ4WordSubgroupBF16R2S8W2 for K % 512 == 0,
// Hv % 8 == 0, affine 4-bit group-64 transposed weights sharing one bf16 row x.
struct GdnInprojAbCtx {
  const array* x{nullptr};
  const array* wa{nullptr};
  const array* sa{nullptr};
  const array* ba{nullptr};
  const array* wb{nullptr};
  const array* sb{nullptr};
  const array* bb{nullptr};
  int k{0};
  int hv{0};
};
bool gdn_inproj_ab_enabled();
const GdnInprojAbCtx* find_gdn_inproj_ab(const array& out, const Stream& stream);
void commit_gdn_inproj_ab(const array& out);
void abort_gdn_inproj_ab(const array& out, const Stream& stream);

class Device;
struct CapabilityReport;

// One fused decode route variable (MLX_OMARCHY_FUSED_GEMV or
// MLX_OMARCHY_FUSED_TRIO) for a device name: the MLX_OMARCHY_FUSED_CHAIN
// master gate must be on; then a set variable wins (any value other than
// "0" is on). Unset, MLX_OMARCHY_FUSED_GEMV is on except on the parts of
// fused_gemv_off_default_part (device.h), and MLX_OMARCHY_FUSED_TRIO is on
// on every part. Reads the environment on every call and caches nothing,
// so each caller decides for its own device.
bool fused_route_enabled(const char* variable, const std::string& device_name);

// MLX_OMARCHY_FUSED_GEMV=0 keeps every QuantizedMatmul, dense BF16
// decode GEMV group, and Add on the per-node path (the
// MLX_OMARCHY_FUSED_CHAIN gate also covers it). Unset, it follows
// fused_route_enabled for the device that executes the work.
bool fused_gemv_enabled(const Device& device);

// The active decode Qmm column geometry ("2x4", "4x8" or "2x16"): the
// chip-gated default plus the MLX_OMARCHY_QMM_GEOM override. For tests and
// docs; defined in primitives.cpp beside the geometry selection.
const char* qmm_geometry_name();
// MLX_OMARCHY_QMM_VEC_Q4_WORD (unset or 1: on). dispatch_quantized_gemv_group refuses every
// group when it is off, so the out-gate fold planner must consult it before adoption.
// Defined in primitives.cpp.
bool qmm_vec_q4_word_enabled();

// Returns the chip-keyed default for the out-gate prologue fold: off on G13
// and G14C, on elsewhere.
bool outgate_fold_default_for_device(const std::string& device_name);

// MLX_OMARCHY_OUTGATE_FOLD=0/1 overrides; unset falls back to
// outgate_fold_default_for_device(device().capabilities().device_name).
bool outgate_fold_enabled();

// Residual-norm decode fold: the default depends on width and on the
// executing device's subgroup and workgroup limits. The override can
// bypass the width policy, not hardware limits or the fused-chain switch.
bool qmm_residual_norm_enabled(
    const CapabilityReport& caps, int norm_cols, bool outgate);

// MLX_OMARCHY_FUSED_GEMV_SWIGLU=0 keeps the SwiGLU store epilogue
// (the gate/up GEMV group that stores silu(gate) * up directly) off
// (explicit MLX_OMARCHY_FUSED_GEMV=0 and MLX_OMARCHY_FUSED_CHAIN=0 also
// cover it); on by default where fused_gemv_enabled is on or on parts
// whose balanced 2-member MLP pairs run on the epilogue-only route (G14C).
bool fused_gemv_swiglu_route_enabled(const std::string& device_name);
bool fused_gemv_swiglu_enabled(const Device& device);

// MLX_OMARCHY_QMM_VEC_TOKEN_MULTI=1 opts the grouped GEMV planner into
// rows 2..16 (the token column; OFF by default on AGX pending the
// data-dependent divergence bisect).
bool gemv_token_multi_enabled(const Device& device);

// Trio: MLX_OMARCHY_FUSED_TRIO=0 keeps f16 RMSNorm rows, the fused
// SwiGLU chain dispatch, and RoPE pairs on their standalone kernels and
// paths (the MLX_OMARCHY_FUSED_CHAIN gate also covers it). The RMSNorm
// and SwiGLU legs have no row-count guard, so the gate reaches f16
// prefill as well as decode. Unset, it is on (fused_route_enabled, asked
// for the device that executes the work); it is independent of
// MLX_OMARCHY_FUSED_GEMV.
bool fused_trio_enabled(const Device& device);

// Validates a planned (query, key) RoPE pair against the FastTrioRopePairF16
// rope-pair contract, allocates the outputs, prepares the keys-side
// producer-direct KV window when one is planned, and records the ONE
// dispatch that rotates both tensors with the near-copied fast_rope
// arithmetic (fast_trio.comp mode 1). Returns false having allocated
// nothing when anything refuses; both nodes then take the ordinary
// RoPE path unchanged. Defined in primitives.cpp beside
// dispatch_quantized_gemv_group.

// Norm payload for one side of a norm-carrying rope pair: the pre-norm
// input and the RMSNorm weight the FastRopeNormPair kernels fold in,
// plus the norm's eps. A null entry means the plain 2-input pair form.
// The optional view (Transpose/Reshape between the norm output and the
// rope) replays over x's contiguous buffer the way
// dispatch_rope_norm_fold composes it.
struct RopePairNorm {
  array x;
  array w;
  float eps;
  std::optional<array> view;
  std::shared_ptr<Primitive> view_prim;
};

bool dispatch_rope_pair(
    std::array<array, 2>& nodes,
    const std::array<const RopePairNorm*, 2>& norms,
    const Stream& stream);
bool dispatch_rope_pair(std::array<array, 2>& nodes, const Stream& stream);

// Fused residual Add + RMSNorm pair (M2D-2): one FastAddRmsNormAppleBF16
// dispatch replaces BinaryVecBF16(Add) -> FastRmsNormAppleBF16 for a
// same-shape bf16 edge, writing the rounded residual (nodes[0]) and the
// normed output (nodes[1]) in the composed path's bit-exact order.
// Returns false having allocated nothing when anything refuses; both
// nodes then take their ordinary paths unchanged. Defined in
// primitives.cpp beside dispatch_rope_pair.
bool dispatch_add_norm_pair(std::array<array, 2>& nodes, const Stream& stream);
} // namespace mlx::core::omarchy