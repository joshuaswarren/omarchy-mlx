// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/compiled.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>

#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/fused_chain.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::omarchy {
namespace {

// Upstream compile_fuse (mlx/compile.cpp is_fusable) only ever places
// unary, binary, ternary (Select), and Broadcast primitives inside a
// Compiled tape, and every class in that set has an Omarchy eval_gpu
// implementation that refuses its own unsupported dtypes and layouts by
// name. The interpreter therefore carries no op allowlist of its own.
[[noreturn]] void unresolved_tape_input(const Primitive& primitive) {
  throw std::runtime_error(
      "[omarchy] Compiled tape op " + std::string(primitive.name()) +
      " reads an input the tape never produced; the interpreter cannot"
      " evaluate this tape. No silent CPU fallback occurs.");
}

} // namespace

void eval_compiled_tape(
    const std::vector<array>& tape,
    const std::vector<array>& tape_inputs,
    const std::vector<array>& tape_outputs,
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const Stream& stream) {
  // Tape arrays refer to the tracing graph. Substitute the eval-time inputs
  // positionally and key every tape node to its computed temporary.
  std::unordered_map<std::uintptr_t, array> resolved;
  resolved.reserve(tape_inputs.size() + tape.size());
  for (size_t i = 0; i < tape_inputs.size(); ++i) {
    resolved.emplace(tape_inputs[i].id(), inputs[i]);
  }
  std::unordered_map<std::uintptr_t, array> broadcast_alias;
  std::unordered_set<std::uintptr_t> output_ids;
  output_ids.reserve(tape_outputs.size());
  for (const auto& out : tape_outputs) {
    output_ids.insert(out.id());
  }
  // FuseDecodeChains gate: read ONCE per tape evaluation. With it off,
  // no chain structures are built and the interpreter below runs the
  // pre-chain per-node loop byte for byte. Per-node submission is a
  // batching bisector and owns the evaluation when set.
  const bool per_node_submit = env_flag("MLX_OMARCHY_TAPE_PER_NODE_SUBMIT");
  const bool fusion_enabled = fused_chain_enabled() && !per_node_submit;
  // Nodes consumed by MORE than one tape consumer (or surfaced as tape
  // outputs) must keep a materialized value, so a fused chain closes
  // after them; only single-consumer nodes may stay interior. This is
  // what makes interior chain members unreachable to non-chain ops by
  // construction.
  std::unordered_set<std::uintptr_t> must_materialize;
  if (fusion_enabled) {
    std::unordered_map<std::uintptr_t, Shape> eval_shapes;
    eval_shapes.reserve(tape_inputs.size() + tape.size());
    for (size_t i = 0; i < tape_inputs.size(); ++i) {
      eval_shapes.emplace(tape_inputs[i].id(), inputs[i].shape());
    }
    for (const auto& node : tape) {
      if (!node.has_primitive()) {
        eval_shapes.emplace(node.id(), node.shape());
        continue;
      }
      bool known = true;
      std::vector<Shape> in_shapes;
      in_shapes.reserve(node.inputs().size());
      for (const auto& in : node.inputs()) {
        auto it = eval_shapes.find(in.id());
        if (it == eval_shapes.end()) {
          known = false;
          break;
        }
        in_shapes.push_back(it->second);
      }
      if (!known) {
        continue;
      }
      Shape out_shape;
      bool is_broadcast = typeid(node.primitive()) == typeid(Broadcast);
      if (is_broadcast) {
        const auto& bcast = static_cast<const Broadcast&>(node.primitive());
        try {
          if (in_shapes.size() >= 2) {
            out_shape = in_shapes[0];
            for (size_t k = 1; k < in_shapes.size(); ++k) {
              out_shape = broadcast_shapes(out_shape, in_shapes[k]);
            }
          } else if (
              !in_shapes.empty() &&
              broadcast_shapes(in_shapes[0], bcast.state()) == bcast.state()) {
            out_shape = bcast.state();
          } else {
            continue;
          }
        } catch (const std::invalid_argument&) {
          continue;
        }
      } else {
        int nd = 0;
        for (const auto& s : in_shapes) {
          nd = std::max(nd, static_cast<int>(s.size()));
        }
        out_shape.resize(nd, 0);
        for (const auto& s : in_shapes) {
          auto dd = nd - static_cast<int>(s.size());
          for (int i = dd; i < nd; ++i) {
            out_shape[i] = std::max(out_shape[i], s[i - dd]);
          }
        }
      }
      auto inserted = eval_shapes.emplace(node.id(), std::move(out_shape));
      if (is_broadcast && !node.inputs().empty()) {
        auto src_it = eval_shapes.find(node.inputs()[0].id());
        if (src_it != eval_shapes.end() &&
            inserted.first->second == src_it->second) {
          auto alias_it = broadcast_alias.find(node.inputs()[0].id());
          broadcast_alias.emplace(
              node.id(),
              alias_it != broadcast_alias.end() ? alias_it->second
                                                : node.inputs()[0]);
        }
      }
    }
    std::vector<std::uintptr_t> alias_outputs;
    for (const auto& id : output_ids) {
      auto alias_it = broadcast_alias.find(id);
      if (alias_it != broadcast_alias.end()) {
        alias_outputs.push_back(alias_it->second.id());
      }
    }
    for (const auto& id : alias_outputs) {
      output_ids.insert(id);
    }
    for (auto it = output_ids.begin(); it != output_ids.end();) {
      if (broadcast_alias.count(*it) > 0) {
        it = output_ids.erase(it);
      } else {
        ++it;
      }
    }
    // Nodes consumed by MORE than one canonical tape consumer (or surfaced
    // as tape outputs) must keep a materialized value, so a fused chain
    // closes after them; only single-consumer nodes may stay interior. This
    // is what makes interior chain members unreachable to non-chain ops by
    // construction. Edges through identity broadcasts count toward the
    // canonical source, and the broadcast nodes themselves are not
    // consumers - they read no buffer.
    std::unordered_map<std::uintptr_t, int> consumer_counts;
    for (const auto& node : tape) {
      if (!node.has_primitive() || broadcast_alias.count(node.id()) > 0) {
        continue;
      }
      for (const auto& in : node.inputs()) {
        auto alias_it = broadcast_alias.find(in.id());
        ++consumer_counts[alias_it != broadcast_alias.end()
                              ? alias_it->second.id()
                              : in.id()];
      }
    }
    must_materialize = output_ids;
    for (const auto& [id, count] : consumer_counts) {
      if (count > 1) {
        must_materialize.insert(id);
      }
    }
  }
  // MLX_OMARCHY_TAPE_CENSUS=<path>: append one line per tape evaluation
  // describing the fragment - node count, per-node op name, dtype,
  // tape-wide use count (references by other nodes plus the output
  // flag), and whether the node is a tape output. Measurement
  // instrument for the elementwise-chain census; off by default.
  if (const char* census_path = std::getenv("MLX_OMARCHY_TAPE_CENSUS")) {
    static FILE* census_out = std::fopen(census_path, "a");
    if (census_out) {
      std::unordered_map<std::uintptr_t, int> uses;
      std::unordered_map<std::uintptr_t, int> index;
      for (size_t n = 0; n < tape.size(); ++n) {
        index[tape[n].id()] = static_cast<int>(n);
      }
      for (const auto& node : tape) {
        for (const auto& in : node.inputs()) {
          uses[in.id()]++;
        }
      }
      std::ostringstream line;
      line << "CENSUS nodes=" << tape.size();
      for (size_t n = 0; n < tape.size(); ++n) {
        const auto& node = tape[n];
        int u = uses.count(node.id()) ? uses.at(node.id()) : 0;
        if (output_ids.count(node.id())) {
          u++;
        }
        char kind = '?';
        switch (kindof(node.dtype())) {
          case Dtype::Kind::b: kind = 'b'; break;
          case Dtype::Kind::i: kind = 'i'; break;
          case Dtype::Kind::u: kind = 'u'; break;
          case Dtype::Kind::V: kind = 'V'; break;
          case Dtype::Kind::f: kind = 'f'; break;
          case Dtype::Kind::c: kind = 'c'; break;
        }
        line << " | n" << n << "="
             << (node.has_primitive() ? node.primitive().name() : "Const")
             << " " << kind << " uses=" << u << " in=[";
        size_t ti = 0;
        for (const auto& in : node.inputs()) {
          auto it = index.find(in.id());
          line << (ti ? "," : "")
               << (it != index.end() ? std::to_string(it->second)
                                     : std::string("x"));
          ti++;
        }
        line << "]" << (output_ids.count(node.id()) ? " OUT" : "");
      }
      std::fprintf(census_out, "%s\n", line.str().c_str());
      std::fflush(census_out);
    }
  }

  // Compiled-tape debug switches (docs/install-omarchy.md). All default
  // off; with none set this function runs exactly the batching path it
  // ran before they existed. The scope publishes the barrier and reuse
  // switches to the encoder and the allocator for this recording;
  // per-node submission and the per-tape drain are applied inline below.
  // Recording happens inside this scope, so every recorded command
  // already carries the switched shape; the eventual submission needs no
  // switch state.
  TapeDebugScope debug_scope(
      env_flag("MLX_OMARCHY_TAPE_FULL_BARRIERS"),
      env_flag("MLX_OMARCHY_TAPE_NO_REUSE"));
  const bool sync_every = env_flag("MLX_OMARCHY_TAPE_SYNC_EVERY");
  static std::atomic<bool> announced{false};
  if (!announced.load(std::memory_order_relaxed) &&
      (per_node_submit || sync_every || tape_full_barriers() ||
       tape_no_reuse()) &&
      !announced.exchange(true, std::memory_order_relaxed)) {
    std::fprintf(
        stderr,
        "[omarchy] compiled-tape debug switches active (diagnostics only,"
        " not product configuration; docs/install-omarchy.md):%s%s%s%s\n",
        per_node_submit ? " MLX_OMARCHY_TAPE_PER_NODE_SUBMIT" : "",
        sync_every ? " MLX_OMARCHY_TAPE_SYNC_EVERY" : "",
        tape_full_barriers() ? " MLX_OMARCHY_TAPE_FULL_BARRIERS" : "",
        tape_no_reuse() ? " MLX_OMARCHY_TAPE_NO_REUSE" : "");
  }


  auto& encoder = get_command_encoder(stream);
  // Fused-chain accumulation (FuseDecodeChains, DEFAULT OFF via
  // MLX_OMARCHY_FUSED_CHAIN, read once above): consecutive same-shape
  // float unary/binary tape nodes collapse into ONE dispatch when the
  // gate is on, and every EXTENSION must consume the open chain's tail
  // so interior members are never consumable from outside. A node the
  // chain cannot carry closes it; the closed chain either fuses (one
  // dispatch) or falls back to the per-node path below, so refusal
  // semantics are unchanged. bf16 nodes fuse through the chain's own
  // bf16 kernels (ROUND_INTERMEDIATE rounds each instruction to storage
  // dtype, matching per-node dispatch); the stale-shape defect behind
  // the 2026-09-02 bf16 gate and the DivLast leaf misindexing behind
  // the in-model fused-bf16 corruption are both fixed (26c67149,
  // da43969e).
  std::optional<FusedChain> chain;
  if (fusion_enabled) {
    chain.emplace(true);
  }
  // A node that consumes the OPEN chain's tail passes the tail's
  // tracing array itself; that reference is only valid while the chain
  // stays open, so any fallback below re-resolves after closing.
  auto close_chain = [&]() -> void {
    if (!chain || chain->size() == 0) {
      return;
    }
    trace::counters().compiled_tape_node_evaluations += chain->size();
    uint64_t dispatches_before =
        trace::counters().vk_compute_dispatches.load(std::memory_order_relaxed);
    encoder.in_tape_recording = true;
    auto fused = chain->evaluate(stream);
    encoder.in_tape_recording = false;
    trace::counters().compiled_tape_dispatches.fetch_add(
        trace::counters().vk_compute_dispatches.load(std::memory_order_relaxed) -
            dispatches_before,
        std::memory_order_relaxed);
    const auto tail = chain->tail_id();
    chain.emplace(true);
    if (fused) {
      resolved.emplace(tail, *fused);
      if (output_ids.find(tail) == output_ids.end()) {
        encoder.add_temporary(*fused);
      }
    }
  };
  auto resolve_inputs = [&](const array& node, std::vector<array>& out_inputs)
      -> bool {
    for (const auto& in : node.inputs()) {
      auto alias_it = broadcast_alias.find(in.id());
      const array& src =
          alias_it != broadcast_alias.end() ? alias_it->second : in;
      auto it = resolved.find(src.id());
      if (it != resolved.end()) {
        out_inputs.push_back(it->second);
      } else if (chain && chain->size() > 0 && chain->tail_id() == src.id()) {
        // Continuation of the open chain.
        out_inputs.push_back(src);
      } else if (!src.has_primitive()) {
        // Scalar constants baked into the tape hold their data already.
        out_inputs.push_back(src);
      } else {
        return false;
      }
    }
    return true;
  };
  for (const auto& node : tape) {
    if (!node.has_primitive()) {
      // A constant the tracer stored in the tape carries its own data.
      resolved.emplace(node.id(), node);
      continue;
    }
    Primitive& primitive = node.primitive();

    std::vector<array> node_inputs;
    node_inputs.reserve(node.inputs().size());
    if (fusion_enabled) {
      if (broadcast_alias.count(node.id()) > 0) {
        continue;
      }
      if (!resolve_inputs(node, node_inputs)) {
        // The node consumes the OPEN chain's tail as an INTERIOR
        // operand of a node that will not join: fuse what the chain
        // carries, then resolve against its materialized tail value.
        close_chain();
        node_inputs.clear();
        if (!resolve_inputs(node, node_inputs)) {
          unresolved_tape_input(primitive);
        }
      }

      // Fast path: the node joins the open fused chain. A tape output
      // may only sit at the tail, so the chain closes right after one.
      if (chain->try_add(
              node,
              node_inputs,
              must_materialize.find(node.id()) !=
                  must_materialize.end())) {
        if (chain->tail_is_tape_output()) {
          close_chain();
        }
        continue;
      }

      // The node cannot join (unsupported op class, shape change, no
      // tail dependency, ...). Close the chain - it may itself fuse -
      // and fall back to the per-node path. If any input referenced
      // the closed chain's tail, re-resolve it against the fused
      // output.
      bool needs_reresolve = false;
      for (const auto& in : node.inputs()) {
        auto alias_it = broadcast_alias.find(in.id());
        const std::uintptr_t target =
            alias_it != broadcast_alias.end() ? alias_it->second.id() : in.id();
        if (chain->size() > 0 && chain->tail_id() == target) {
          needs_reresolve = true;
        }
      }
      close_chain();
      if (needs_reresolve) {
        node_inputs.clear();
        if (!resolve_inputs(node, node_inputs)) {
          unresolved_tape_input(primitive);
        }
      }
    } else {
      // Gate off: the pre-chain resolution, byte for byte - no chain
      // objects, no chain scans, no chain structures.
      for (const auto& in : node.inputs()) {
        auto it = resolved.find(in.id());
        if (it != resolved.end()) {
          node_inputs.push_back(it->second);
        } else if (!in.has_primitive()) {
          // Scalar constants baked into the tape hold their data
          // already.
          node_inputs.push_back(in);
        } else {
          unresolved_tape_input(primitive);
        }
      }
    }

    // One output per tape node. The node's primitive stays attached
    // during the dispatch so eval_gpu sees the compiled stream; after it,
    // the value is cut off its graph like an eager node the evaluator has
    // run (evaluated, then detached). A consumer's eval_gpu may settle()
    // a graph over its inputs (the Sin/Cos argument gate); a value left
    // unscheduled with live inputs would be re-dispatched there with its
    // whole upstream sub-tape.
    //
    // The output shape is derived from the eval-time inputs, not the
    // trace. A shapeless fragment serves every input shape from one
    // trace - the compile cache matches ndim and dtype only - so a
    // decode call at [1,1,...] legally reuses the prefill-traced tape,
    // and node.shape() would be the stale prefill shape. Elementwise
    // outputs are the trailing broadcast of their inputs (the same
    // contract upstream applies to fused fragments); Broadcast keeps its
    // own rule. At the trace shape this equals node.shape(), so
    // exact-shape tapes are byte-for-byte unchanged.
    Shape out_shape;
    if (typeid(primitive) == typeid(Broadcast)) {
      out_shape = primitive.output_shapes(node_inputs).front();
    } else {
      int nd = 0;
      for (const auto& in : node_inputs) {
        nd = std::max(nd, static_cast<int>(in.ndim()));
      }
      out_shape.resize(nd, 0);
      for (const auto& in : node_inputs) {
        auto dd = nd - static_cast<int>(in.ndim());
        for (int i = dd; i < nd; ++i) {
          out_shape[i] = std::max(out_shape[i], in.shape()[i - dd]);
        }
      }
    }
    std::vector<array> outs;
    outs.push_back(
        array(std::move(out_shape), node.dtype(), node.primitive_ptr(), node_inputs));
    trace::counters().compiled_tape_node_evaluations++;
    uint64_t dispatches_before =
        trace::counters().vk_compute_dispatches.load(std::memory_order_relaxed);
    encoder.in_tape_recording = true;
    primitive.eval_gpu(node_inputs, outs);
    encoder.in_tape_recording = false;
    trace::counters().compiled_tape_dispatches.fetch_add(
        trace::counters().vk_compute_dispatches.load(std::memory_order_relaxed) -
        dispatches_before,
        std::memory_order_relaxed);
    outs[0].set_status(array::Status::evaluated);
    outs[0].detach();
    resolved.emplace(node.id(), outs[0]);
    if (output_ids.find(node.id()) == output_ids.end()) {
      // The evaluator's temporaries cover only the Compiled inputs and
      // outputs; keep intermediate buffers alive until the submission
      // completes. Output buffers live with the graph instead.
      encoder.add_temporary(outs[0]);
    }
    if (per_node_submit) {
      // MLX_OMARCHY_TAPE_PER_NODE_SUBMIT (diagnostic): submit after every
      // node, matching eager's op-granular command-buffer shape while
      // running the tape's own code path. Each submission waits on this
      // stream's previous completion, so the tape serializes exactly as
      // an eager chain would. If the Honeykrisp corruption disappears
      // under this switch, the defect lives in many-dispatches-per-
      // command-buffer; if it persists, the command buffer is innocent
      // and the tape's resource handling stays suspect.
      encoder.commit();
    }
  }
  close_chain();

  for (size_t j = 0; j < tape_outputs.size(); ++j) {
    auto alias_it = broadcast_alias.find(tape_outputs[j].id());
    const array& source =
        alias_it != broadcast_alias.end() ? alias_it->second : tape_outputs[j];
    outputs[j].copy_shared_buffer(resolved.at(source.id()));
  }
  if (sync_every) {
    // MLX_OMARCHY_TAPE_SYNC_EVERY (diagnostic): drain this stream
    // before returning, so no submission of the tape - and nothing
    // queued behind it - executes while the host runs ahead. Tests
    // whether host run-ahead is load-bearing for the Honeykrisp
    // corruption; the M1 verdict on the first three switches made this
    // the next named suspect (receipts/2026-09-03-tape-layer-isolation-
    // switches.md, MEASURED OUTCOME).
    encoder.synchronize();
  }
}

} // namespace mlx::core::omarchy
