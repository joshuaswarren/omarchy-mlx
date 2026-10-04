// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/encoder.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <unistd.h>
#include <sys/syscall.h>

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/backend/omarchy/gpu_profiler.h"
#include "mlx/backend/omarchy/vulkan.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core::omarchy {

namespace {

// Saturating byte-range end for dependency tracking. VK_WHOLE_SIZE and
// overflowing sizes clamp to the address space so a tracked range never
// under-covers the accesses it stands for.
inline VkDeviceSize tracked_range_end(VkDeviceSize offset, VkDeviceSize size) {
  if (size == VK_WHOLE_SIZE) {
    return UINT64_MAX;
  }
  VkDeviceSize end = offset + size;
  return end < offset ? UINT64_MAX : end;
}

// MLX_OMARCHY_DEBUG_REPEAT_<CLASS>=k / MLX_OMARCHY_DEBUG_EMPTY=k
// (diagnostic instrument, default off — see
// receipts/2026-10-04-prefill-sensitivity). REPEAT re-issues every
// dispatch of one kernel class k-1 extra times back-to-back with the
// same bound pipeline, descriptor set, push constants, and grid, so the
// kernel recomputes identical values from unchanged inputs (all
// repeatable classes are idempotent; generation digests must stay
// bit-identical). The added GPU time per class is therefore
// (k-1) x that class's busy time plus per-dispatch turnaround, and the
// marginal wall response d(wall)/d(extra) reads directly as
// on-critical-path (~1) vs overlapped (~0). EMPTY records k extra
// 1-workgroup binding-free null-kernel dispatches after every dispatch
// to price pure per-dispatch turnover inside a real graph.
// Repeats/empties are not MLX work: no work-budget, hazard-tracker,
// profiler, or dispatch-counter updates, and they refuse wave-sched
// mode (which buffers nodes instead of recording, so the injection
// would silently no-op).
enum class DebugClass : uint8_t {
  Qmm = 0,
  Sdpa,
  Gdn,
  Rms,
  Rope,
  Cast,
  Copy,
  Ew,
  Softmax,
  Matmul,
  Other,
  Count
};

DebugClass debug_class_of(ComputeKernel kernel) {
  switch (kernel) {
    case ComputeKernel::QmmF32:
    case ComputeKernel::QmmF16:
    case ComputeKernel::QmmBF16:
    case ComputeKernel::QmmVecF32:
    case ComputeKernel::QmmVecF16:
    case ComputeKernel::QmmVecBF16:
    case ComputeKernel::QmmVecSubgroupF32:
    case ComputeKernel::QmmVecSubgroupF16:
    case ComputeKernel::QmmVecSubgroupBF16:
    case ComputeKernel::QmmTileF32:
    case ComputeKernel::QmmTileF16:
    case ComputeKernel::QmmTileBF16:
    case ComputeKernel::QmmFpF32:
    case ComputeKernel::QmmFpF16:
    case ComputeKernel::QmmFpBF16:
    case ComputeKernel::QmmVecFpF32:
    case ComputeKernel::QmmVecFpF16:
    case ComputeKernel::QmmVecFpBF16:
    case ComputeKernel::QmmVecSubgroupFpF32:
    case ComputeKernel::QmmVecSubgroupFpF16:
    case ComputeKernel::QmmVecSubgroupFpBF16:
    case ComputeKernel::QmmTileFpF32:
    case ComputeKernel::QmmTileFpF16:
    case ComputeKernel::QmmTileFpBF16:
    case ComputeKernel::GatherQmmF32:
    case ComputeKernel::GatherQmmF16:
    case ComputeKernel::GatherQmmBF16:
    case ComputeKernel::GatherQmmNbF32:
    case ComputeKernel::GatherQmmNbF16:
    case ComputeKernel::GatherQmmNbBF16:
    case ComputeKernel::GatherQmmNbFpF32:
    case ComputeKernel::GatherQmmNbFpF16:
    case ComputeKernel::GatherQmmNbFpBF16:
    case ComputeKernel::GatherQmmNbFpHgsF32:
    case ComputeKernel::GatherQmmNbFpHgsF16:
    case ComputeKernel::GatherQmmNbFpHgsBF16:
    case ComputeKernel::QmmVecQ4WordF32:
    case ComputeKernel::QmmVecQ4WordF16:
    case ComputeKernel::QmmVecQ4WordBF16:
    case ComputeKernel::QmmVecQ4WordSubgroupF32:
    case ComputeKernel::QmmVecQ4WordSubgroupF16:
    case ComputeKernel::QmmVecQ4WordSubgroupBF16:
    case ComputeKernel::QmmTileRbF16:
    case ComputeKernel::QmmPrefillCoopmatF16:
    case ComputeKernel::QmmVecQ4MultiF32:
    case ComputeKernel::QmmVecQ4MultiF16:
    case ComputeKernel::QmmVecQ4MultiBF16:
    case ComputeKernel::QmmVecQ4MultiSubgroupF32:
    case ComputeKernel::QmmVecQ4MultiSubgroupF16:
    case ComputeKernel::QmmVecQ4MultiSubgroupBF16:
    case ComputeKernel::QmmTileRbPreciseF16:
    case ComputeKernel::QmmPrefillFmaF16:
    case ComputeKernel::QmmPrefillFmaL16C4F16:
    case ComputeKernel::QmmPrefillFmaL8C4F16:
    case ComputeKernel::QmmPrefillFmaPreciseF16:
    case ComputeKernel::MatmulVecMultiBF16:
    case ComputeKernel::QmmPrefillCoopmatM16F16:
    case ComputeKernel::QmmPrefillCoopmatBF16:
    case ComputeKernel::QmmPrefillCoopmatM16BF16:
    case ComputeKernel::QmmPrefillCoopmatBF16X32:
    case ComputeKernel::QmmPrefillCoopmatM16BF16X32:
    case ComputeKernel::QmmPrefillCoopmatBF16X32FullN:
    case ComputeKernel::QmmPrefillCoopmatM16BF16X32FullN:
    case ComputeKernel::QmmVecGreedyBF16:
    case ComputeKernel::QmmVecQ4MultiOutgateBF16:
    case ComputeKernel::GatherMmF32:
    case ComputeKernel::GatherMmF16:
    case ComputeKernel::GatherMmBF16:
    case ComputeKernel::QmmPrefillCoopmatBF16X32FullNRasterSwap:
    case ComputeKernel::QmmPrefillCoopmatBF16X32FullNRasterG2:
    case ComputeKernel::QmmPrefillCoopmatBF16X32FullNRasterG4:
    case ComputeKernel::QmmPrefillCoopmatBF16X32FullNRasterG8:
    case ComputeKernel::QmmPrefillCoopmatBF16X32FullNTwoN:
      return DebugClass::Qmm;
    case ComputeKernel::SdpaDecodeNativeF16:
    case ComputeKernel::SdpaDecodeNativeBF16:
    case ComputeKernel::SdpaDecodeNativeTwoPassP1F16:
    case ComputeKernel::SdpaDecodeNativeTwoPassP2F16:
    case ComputeKernel::SdpaDecodeNativeBF16Hd256:
    case ComputeKernel::SdpaDecodeNativeBF16Hd32:
    case ComputeKernel::SdpaDecodeNativeBF16Hd96:
    case ComputeKernel::SdpaDecodeNativeBF16Hd128:
    case ComputeKernel::SdpaDecodeNativeBF16Hd160:
    case ComputeKernel::SdpaDecodeNativeBF16Hd192:
    case ComputeKernel::SdpaDecodeNativeBF16Hd224:
    case ComputeKernel::SdpaDecodeNativeF16Hd128:
    case ComputeKernel::SdpaDecodeNativeTwoPassP1F16Hd128:
    case ComputeKernel::SdpaDecodeNativeTwoPassP2F16Hd128:
    case ComputeKernel::PartitionSmallKF32:
    case ComputeKernel::PartitionSmallKF16:
    case ComputeKernel::PartitionSmallKBF16:
    case ComputeKernel::SdpaVjpOdoF32:
    case ComputeKernel::SdpaVjpOdoF16:
    case ComputeKernel::SdpaVjpOdoBF16:
    case ComputeKernel::SdpaVjpDsF32:
    case ComputeKernel::SdpaVjpLseF32:
    case ComputeKernel::SdpaVjpReduceF32:
    case ComputeKernel::SdpaVjpReduceF16:
    case ComputeKernel::SdpaVjpReduceBF16:
    case ComputeKernel::MatmulF32CoopmatQkBF16:
    case ComputeKernel::MatmulF32CoopmatPvBF16:
      return DebugClass::Sdpa;
    case ComputeKernel::GatedDeltaDecodeBF16:
    case ComputeKernel::GatedDeltaPrefillBF16:
    case ComputeKernel::GatedDeltaPrefillCoopmatBF16:
    case ComputeKernel::GatedDeltaPrefillCoopmatBatchBF16:
    case ComputeKernel::GdnConvDecodeBF16:
    case ComputeKernel::GatedDeltaDecodeBF16Untiled:
    case ComputeKernel::GatedDeltaDecodeBF16Pf:
    case ComputeKernel::GdnVjpSaveBF16:
    case ComputeKernel::GdnVjpBF16:
    case ComputeKernel::GdnConvDecodeAppleBF16:
    case ComputeKernel::ConvDw1dF32:
    case ComputeKernel::ConvDw1dF16:
    case ComputeKernel::ConvDw1dBF16:
      return DebugClass::Gdn;
    case ComputeKernel::FastRmsNormF32:
    case ComputeKernel::FastRmsNormF16:
    case ComputeKernel::FastRmsNormBF16:
    case ComputeKernel::FastLayerNormF32:
    case ComputeKernel::FastLayerNormF16:
    case ComputeKernel::FastLayerNormBF16:
    case ComputeKernel::FastRmsNormVjpDxF32:
    case ComputeKernel::FastRmsNormVjpDxF16:
    case ComputeKernel::FastRmsNormVjpDxBF16:
    case ComputeKernel::FastLayerNormVjpDxF32:
    case ComputeKernel::FastLayerNormVjpDxF16:
    case ComputeKernel::FastLayerNormVjpDxBF16:
    case ComputeKernel::FastRmsNormVjpDwF32:
    case ComputeKernel::FastRmsNormVjpDwF16:
    case ComputeKernel::FastRmsNormVjpDwBF16:
    case ComputeKernel::FastLayerNormVjpDwF32:
    case ComputeKernel::FastLayerNormVjpDwF16:
    case ComputeKernel::FastLayerNormVjpDwBF16:
    case ComputeKernel::FastRmsNormVjpDwReduceF32:
    case ComputeKernel::FastRmsNormVjpDwReduceF16:
    case ComputeKernel::FastRmsNormVjpDwReduceBF16:
    case ComputeKernel::FastNormGatedBF16:
    case ComputeKernel::FastNormGatedOnlyBF16:
    case ComputeKernel::FastTrioNormF16:
    case ComputeKernel::FastRmsNormAppleBF16:
    case ComputeKernel::FastNormGatedAppleBF16:
    case ComputeKernel::FastNormGatedOnlyAppleBF16:
      return DebugClass::Rms;
    case ComputeKernel::FastRopeF32:
    case ComputeKernel::FastRopeF16:
    case ComputeKernel::FastRopeBF16:
    case ComputeKernel::FastRopeFreqsF32:
    case ComputeKernel::FastRopeFreqsF16:
    case ComputeKernel::FastRopeFreqsBF16:
    case ComputeKernel::FastRopeNormBF16:
    case ComputeKernel::FastTrioRopePairF16:
    case ComputeKernel::FastRopeNormAppleBF16:
      return DebugClass::Rope;
    case ComputeKernel::CastF16F32:
    case ComputeKernel::CastBoolF32:
    case ComputeKernel::CastBoolI32:
    case ComputeKernel::CastBoolF16:
    case ComputeKernel::CastBoolBF16:
    case ComputeKernel::CastF32F16:
    case ComputeKernel::CastBF16F32:
    case ComputeKernel::CastF32BF16:
    case ComputeKernel::CastBF16F16:
    case ComputeKernel::CastF16BF16:
    case ComputeKernel::CastI32F32:
    case ComputeKernel::CastU32F32:
    case ComputeKernel::CastF32I32:
    case ComputeKernel::CastI32F16:
    case ComputeKernel::CastF16I32:
    case ComputeKernel::CastI32BF16:
    case ComputeKernel::CastBF16I32:
    case ComputeKernel::CastF32Complex64:
    case ComputeKernel::CastI32Complex64:
    case ComputeKernel::CastU32Complex64:
    case ComputeKernel::CastBoolComplex64:
    case ComputeKernel::CastF16Complex64:
    case ComputeKernel::CastBF16Complex64:
    case ComputeKernel::CastComplex64F32:
    case ComputeKernel::CastIntW1W1:
    case ComputeKernel::CastIntW1W2:
    case ComputeKernel::CastIntW1W4:
    case ComputeKernel::CastIntW1W8:
    case ComputeKernel::CastIntW2W1:
    case ComputeKernel::CastIntW2W2:
    case ComputeKernel::CastIntW2W4:
    case ComputeKernel::CastIntW2W8:
    case ComputeKernel::CastIntW4W1:
    case ComputeKernel::CastIntW4W2:
    case ComputeKernel::CastIntW4W4:
    case ComputeKernel::CastIntW4W8:
    case ComputeKernel::CastIntW8W1:
    case ComputeKernel::CastIntW8W2:
    case ComputeKernel::CastIntW8W4:
    case ComputeKernel::CastIntW8W8:
    case ComputeKernel::QuantizeF32:
    case ComputeKernel::QuantizeF16:
    case ComputeKernel::QuantizeBF16:
    case ComputeKernel::DequantF32:
    case ComputeKernel::DequantF16:
    case ComputeKernel::DequantBF16:
    case ComputeKernel::QuantizeFpF32:
    case ComputeKernel::QuantizeFpF16:
    case ComputeKernel::QuantizeFpBF16:
    case ComputeKernel::DequantFpF32:
    case ComputeKernel::DequantFpF16:
    case ComputeKernel::DequantFpBF16:
    case ComputeKernel::Fp8ToF32:
    case ComputeKernel::Fp8ToF16:
    case ComputeKernel::Fp8ToBF16:
    case ComputeKernel::Fp8FromF32:
    case ComputeKernel::Fp8FromF16:
    case ComputeKernel::Fp8FromBF16:
      return DebugClass::Cast;
    case ComputeKernel::CopyGeneralF32:
    case ComputeKernel::CopyGeneralF16:
    case ComputeKernel::CopyGeneralBF16:
    case ComputeKernel::CopyGeneralU32:
    case ComputeKernel::CopyGeneralBool:
    case ComputeKernel::CopyGeneralU8:
    case ComputeKernel::CopyGeneralU16:
    case ComputeKernel::CopyGeneralU64:
    case ComputeKernel::CopyGeneralComplex64:
      return DebugClass::Copy;
    case ComputeKernel::ElementwiseF32:
    case ComputeKernel::ElementwiseF16:
    case ComputeKernel::ElementwiseBF16:
    case ComputeKernel::ElementwiseI32:
    case ComputeKernel::ElementwiseU32:
    case ComputeKernel::ElementwiseI8:
    case ComputeKernel::ElementwiseU8:
    case ComputeKernel::ElementwiseI16:
    case ComputeKernel::ElementwiseU16:
    case ComputeKernel::ElementwiseI64:
    case ComputeKernel::ElementwiseU64:
    case ComputeKernel::ElementwiseLiteF32:
    case ComputeKernel::ElementwiseLiteF16:
    case ComputeKernel::ElementwiseLiteBF16:
    case ComputeKernel::BinaryVecF16:
    case ComputeKernel::BinaryVecBF16:
    case ComputeKernel::UnaryVecF16:
    case ComputeKernel::UnaryVecBF16:
    case ComputeKernel::SiluF16:
    case ComputeKernel::SiluBF16:
    case ComputeKernel::SwigluF16:
    case ComputeKernel::SwigluBF16:
    case ComputeKernel::FastTrioSwigluF16:
    case ComputeKernel::FusedChainF32:
    case ComputeKernel::FusedChainBF16:
    case ComputeKernel::HadamardF32:
    case ComputeKernel::HadamardF16:
    case ComputeKernel::HadamardBF16:
    case ComputeKernel::ComplexElementwise:
    case ComputeKernel::ComplexReal:
    case ComputeKernel::ComplexImag:
    case ComputeKernel::ComplexAbs:
    case ComputeKernel::ComplexAbsAsComplex:
      return DebugClass::Ew;
    case ComputeKernel::SoftmaxF32:
    case ComputeKernel::SoftmaxF16:
    case ComputeKernel::SoftmaxBF16:
    case ComputeKernel::LogSumExpF32:
    case ComputeKernel::LogSumExpF16:
    case ComputeKernel::LogSumExpBF16:
      return DebugClass::Softmax;
    case ComputeKernel::MatmulF32:
    case ComputeKernel::MatmulF32Coopmat:
    case ComputeKernel::MatmulF16:
    case ComputeKernel::MatmulBF16:
    case ComputeKernel::MatmulVecF32:
    case ComputeKernel::MatmulVecF16:
    case ComputeKernel::MatmulVecBF16:
    case ComputeKernel::MatmulComplex64:
    case ComputeKernel::MatmulBF16Coopmat:
    case ComputeKernel::MatmulRbF16:
    case ComputeKernel::MatmulBf16Fma:
    case ComputeKernel::MatmulBf16FmaL16C4:
    case ComputeKernel::SegmentedMmF32:
    case ComputeKernel::SegmentedMmF16:
    case ComputeKernel::SegmentedMmBF16:
      return DebugClass::Matmul;
    default:
      return DebugClass::Other;
  }
}

uint32_t debug_env_count(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr) {
    return 0;
  }
  long n = std::strtol(v, nullptr, 10);
  if (n < 0) {
    return 0;
  }
  return static_cast<uint32_t>(std::min<long>(n, 4096));
}

struct DebugRepeatConfig {
  uint32_t repeat[static_cast<size_t>(DebugClass::Count) - 1]{};
  uint32_t empty{0};

  bool any() const {
    for (uint32_t r : repeat) {
      if (r > 0) {
        return true;
      }
    }
    return empty > 0;
  }
};

const DebugRepeatConfig& debug_repeat_config() {
  static const DebugRepeatConfig cfg = []() {
    DebugRepeatConfig c;
    c.repeat[static_cast<size_t>(DebugClass::Qmm)] =
        debug_env_count("MLX_OMARCHY_DEBUG_REPEAT_QMM");
    c.repeat[static_cast<size_t>(DebugClass::Sdpa)] =
        debug_env_count("MLX_OMARCHY_DEBUG_REPEAT_SDPA");
    c.repeat[static_cast<size_t>(DebugClass::Gdn)] =
        debug_env_count("MLX_OMARCHY_DEBUG_REPEAT_GDN");
    c.repeat[static_cast<size_t>(DebugClass::Rms)] =
        debug_env_count("MLX_OMARCHY_DEBUG_REPEAT_RMS");
    c.repeat[static_cast<size_t>(DebugClass::Rope)] =
        debug_env_count("MLX_OMARCHY_DEBUG_REPEAT_ROPE");
    c.repeat[static_cast<size_t>(DebugClass::Cast)] =
        debug_env_count("MLX_OMARCHY_DEBUG_REPEAT_CAST");
    c.repeat[static_cast<size_t>(DebugClass::Copy)] =
        debug_env_count("MLX_OMARCHY_DEBUG_REPEAT_COPY");
    c.repeat[static_cast<size_t>(DebugClass::Ew)] =
        debug_env_count("MLX_OMARCHY_DEBUG_REPEAT_EW");
    c.repeat[static_cast<size_t>(DebugClass::Softmax)] =
        debug_env_count("MLX_OMARCHY_DEBUG_REPEAT_SOFTMAX");
    c.repeat[static_cast<size_t>(DebugClass::Matmul)] =
        debug_env_count("MLX_OMARCHY_DEBUG_REPEAT_MATMUL");
    c.empty = debug_env_count("MLX_OMARCHY_DEBUG_EMPTY");
    return c;
  }();
  return cfg;
}

} // namespace

std::vector<uint32_t> wave_levels(std::span<const WaveNode> nodes) {
  const size_t n = nodes.size();
  std::vector<uint32_t> level(n, 0);
  // Earlier nodes touching each buffer, so a node only range-checks
  // against nodes that could possibly hazard it.
  std::unordered_map<VkBuffer, std::vector<uint32_t>> touchers;
  auto overlaps = [](const TrackedRange& a, const TrackedRange& b) {
    return a.buffer == b.buffer && a.offset < b.end && b.offset < a.end;
  };
  std::vector<uint32_t> candidates;
  for (size_t i = 0; i < n; ++i) {
    candidates.clear();
    for (const auto& r : nodes[i].reads) {
      if (auto it = touchers.find(r.buffer); it != touchers.end()) {
        candidates.insert(candidates.end(), it->second.begin(), it->second.end());
      }
    }
    for (const auto& r : nodes[i].writes) {
      if (auto it = touchers.find(r.buffer); it != touchers.end()) {
        candidates.insert(candidates.end(), it->second.begin(), it->second.end());
      }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    uint32_t wave = 0;
    for (uint32_t j : candidates) {
      const WaveNode& earlier = nodes[j];
      bool hazard = false;
      for (const auto& r : nodes[i].reads) {
        for (const auto& w : earlier.writes) {
          if (overlaps(r, w)) {
            hazard = true;
            break;
          }
        }
      }
      for (const auto& w : nodes[i].writes) {
        if (hazard) {
          break;
        }
        for (const auto& tw : earlier.writes) {
          if (overlaps(w, tw)) {
            hazard = true;
            break;
          }
        }
        if (hazard) {
          break;
        }
        for (const auto& tr : earlier.reads) {
          if (overlaps(w, tr)) {
            hazard = true;
            break;
          }
        }
      }
      if (hazard) {
        wave = std::max(wave, level[j] + 1);
      }
    }
    level[i] = wave;
    for (const auto& r : nodes[i].reads) {
      touchers[r.buffer].push_back(static_cast<uint32_t>(i));
    }
    for (const auto& r : nodes[i].writes) {
      touchers[r.buffer].push_back(static_cast<uint32_t>(i));
    }
  }
  return level;
}

bool CommandEncoder::wave_sched() {
  // MLX_OMARCHY_WAVE_SCHED (docs/install-omarchy.md): default off. When on
  // (and only with gated barriers — wave mode rides the same SPIR-V access
  // reflection and tracker), dispatch/copy/fill nodes buffer at arrival
  // and submit() records the batch in greedy earliest-wave order with one
  // full dependency barrier between waves. Reordering is numerics-inert:
  // a node's inputs are final once its hazard sources complete, and every
  // hazard source of a node sits in a strictly earlier wave.
  static const bool on = []() {
    const char* e = std::getenv("MLX_OMARCHY_WAVE_SCHED");
    return e != nullptr && env_flag("MLX_OMARCHY_WAVE_SCHED") &&
        gated_barriers();
  }();
  return on;
}

bool CommandEncoder::wave_diag() {
  // MLX_OMARCHY_WAVE_DIAG (census diagnostic): per flush, classify the
  // edge that determined each node's wave (RAW/WAW/WAR against the
  // max-level hazard source) and print a compact histogram. Counts only;
  // inert for recording.
  static const bool on = []() {
    const char* e = std::getenv("MLX_OMARCHY_WAVE_DIAG");
    return e != nullptr && env_flag("MLX_OMARCHY_WAVE_DIAG");
  }();
  return on;
}

bool CommandEncoder::export_dep_masks() {
  // MLX_OMARCHY_EXPORT_DEP_MASKS (design 22b395d phase 1): default ON
  // for the phase-1 branch — the computation reuses tracker state and
  // is provably inert (records are stored, never consumed).
  static const bool on = !env_flag("MLX_OMARCHY_NO_EXPORT_DEP_MASKS");
  return on;
}
bool CommandEncoder::dep_dump() {
  static const int n = []() {
    const char *v = getenv("MLX_OMARCHY_DEP_DUMP");
    return v ? atoi(v) : 0;
  }();
  return n > 0;
}
int CommandEncoder::dep_dump_n() {
  static const int n = []() {
    const char *v = getenv("MLX_OMARCHY_DEP_DUMP");
    return v ? atoi(v) : 0;
  }();
  return n;
}
bool CommandEncoder::gated_barriers() {
  // MLX_OMARCHY_GATED_BARRIERS (docs/install-omarchy.md): default ON since
  // 2026-09-29 (set 0 to restore the unconditional pre+post barriers).
  // Dispatch/copy/fill nodes record a barrier only when their buffer
  // ranges overlap unsynced work of the open batch. Read once: the gate
  // shapes recorded commands, so flipping it mid-batch would desync the
  // tracker from the command buffer.
  static const bool on = []() {
    return std::getenv("MLX_OMARCHY_GATED_BARRIERS") == nullptr ||
        env_flag("MLX_OMARCHY_GATED_BARRIERS");
  }();
  return on;
}

bool CommandEncoder::batch_needs_barrier(
    std::span<const TrackedRange> reads,
    std::span<const TrackedRange> writes) const {
  auto overlaps = [](const TrackedRange& a, const TrackedRange& b) {
    return a.buffer == b.buffer && a.offset < b.end && b.offset < a.end;
  };
  // Read after write and write after write.
  for (const auto& r : reads) {
    for (const auto& w : tracked_writes_) {
      if (overlaps(r, w)) {
        return true;
      }
    }
  }
  // Write after write and write after read: compute-to-compute in one
  // queue has no execution dependency without a barrier, so a node
  // writing a range any earlier node read must also wait.
  for (const auto& w : writes) {
    for (const auto& tw : tracked_writes_) {
      if (overlaps(w, tw)) {
        return true;
      }
    }
    for (const auto& tr : tracked_reads_) {
      if (overlaps(w, tr)) {
        return true;
      }
    }
  }
  return false;
}

void CommandEncoder::record_dependency_barrier() {
  // The heaviest correct dependency: all commands, all memory access,
  // both directions. The tracker restarts after it because the barrier
  // orders everything recorded before it.
  VkMemoryBarrier full{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  full.srcAccessMask =
      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  full.dstAccessMask =
      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  vk::device_table().CmdPipelineBarrier(
      cmd_,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      0,
      1,
      &full,
      0,
      nullptr,
      0,
      nullptr);
  tracked_reads_.clear();
  tracked_writes_.clear();
  head_synced_ = true;
}

void CommandEncoder::reset_dependency_tracking() {
  tracked_reads_.clear();
  tracked_writes_.clear();
  head_synced_ = false;
}

CommandEncoder::CommandEncoder(Device& device) : device_(device) {
  auto& dt = vk::device_table();
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = device_.queue_family();
  VKX_CHECK(dt.CreateCommandPool(device_.handle(), &pci, nullptr, &pool_));

  std::array<VkCommandBuffer, kInFlightCommandBuffers> buffers{};
  VkCommandBufferAllocateInfo ai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool_;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = kInFlightCommandBuffers;
  VKX_CHECK(dt.AllocateCommandBuffers(device_.handle(), &ai, buffers.data()));
  VkEventCreateInfo event_info{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
  for (int i = 0; i < kInFlightCommandBuffers; ++i) {
    slots_[i].cmd = buffers[i];
    VKX_CHECK(dt.CreateEvent(
        device_.handle(), &event_info, nullptr, &slots_[i].started));
  }
  prof::get().attach(this, device_);
}

CommandEncoder::~CommandEncoder() {
  // Drain pending work (and its temporaries) while the device is alive.
  // A wedged queue throws here; swallow so destruction can continue, the
  // bounded error already surfaced through synchronize().
  //
  // The command and descriptor pools are deliberately NOT destroyed
  // here. After a watchdog throw the newest submission is still in
  // flight, and even after a successful join Mesa signals a submission's
  // semaphores before its submit-final cleanup retires the command
  // buffer (see drain_through), so a pool destroy in this destructor can
  // free state the driver's queue thread still walks - observed as
  // teardown SIGSEGV, pure-virtual aborts, and Khronos-validation-layer
  // crashes after a watchdog throw. The pools are children of the
  // VkDevice; vkDestroyDevice releases them with it.
  try {
    synchronize();
  } catch (const std::exception&) {
  }
}

// Join this encoder's newest in-flight submission, if any, and refresh
// host mappings. After this, every command buffer this encoder submitted
// has left the pending state (the completion timeline is strictly
// ordered), so they can legally be begun again, and host reads see the
// submissions' final bytes.
void CommandEncoder::join_last_completion(const char* reason) {
  if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr) {
    fprintf(stderr, "[rtmod] JOIN tid=%lu reason=%s last=%lu\n", (unsigned long)syscall(SYS_gettid), reason,
            (unsigned long)last_completion_);
  }
  if (last_completion_ == 0) {
    return;
  }
  uint64_t value = last_completion_;
  uint64_t join_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  device_.completions().wait(value);
  uint64_t wait_t1 = prof::get().profiling() ? prof::host_ns() : 0;
  last_completion_ = 0;
  omarchy::allocator().invalidate_noncoherent(device_.handle());
  uint64_t inval_t2 = prof::get().profiling() ? prof::host_ns() : 0;
  prof::get().on_join(this, value, join_t0, wait_t1, inval_t2, reason);
}

void CommandEncoder::ensure_recording() {
  if (recording_) {
    return;
  }
  uint64_t begin_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  // Acquire a ring slot whose submission has completed. Newer submissions
  // keep executing on the device while this batch records; only when all
  // slots are in flight does the host join the oldest. The profiler's
  // begin cost includes this wait: it is the residual host-side stall the
  // 2026-09-02 profile attributed to per-record joins.
  auto& completions = device_.completions();
  int chosen = -1;
  uint64_t oldest_value = UINT64_MAX;
  int oldest = -1;
  for (int i = 0; i < kInFlightCommandBuffers; ++i) {
    uint64_t value = slots_[i].in_flight;
    if (value == 0 || completions.drained_value() >= value) {
      slots_[i].in_flight = 0;
      chosen = i;
      break;
    }
    if (value < oldest_value) {
      oldest_value = value;
      oldest = i;
    }
  }
  if (chosen < 0) {
    completions.wait(oldest_value);
    slots_[oldest].in_flight = 0;
    chosen = oldest;
  }
  current_slot_ = chosen;
  cmd_ = slots_[chosen].cmd;
  completions.reset_progress_event(slots_[chosen].started);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VKX_CHECK(vk::device_table().BeginCommandBuffer(cmd_, &bi));
  vk::device_table().CmdSetEvent(
      cmd_, slots_[chosen].started, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
  prof::get().on_begin(
      this, chosen, cmd_, begin_t0 != 0 ? prof::host_ns() - begin_t0 : 0);
  recording_ = true;
}

void CommandEncoder::copy_buffer(
    VkBuffer src,
    VkBuffer dst,
    VkDeviceSize size,
    VkDeviceSize src_offset,
    VkDeviceSize dst_offset) {
  if (wave_sched()) {
    PendingNode& node = pending_.emplace_back();
    node.kind = PendingNode::Kind::Copy;
    node.prim = trace::current_prim();
    node.src = src;
    node.dst = dst;
    node.size = size;
    node.src_offset = src_offset;
    node.dst_offset = dst_offset;
    node.rd[node.nr++] = {src, src_offset, tracked_range_end(src_offset, size)};
    node.wr[node.nw++] = {dst, dst_offset, tracked_range_end(dst_offset, size)};
    node_count_++;
    trace::counters().vk_buffer_copies++;
    return;
  }
  ensure_recording();
  // The tape-full diagnostic forces the heaviest dependency around the
  // copy and restarts tracking either way.
  if (tape_full_barriers()) {
    record_dependency_barrier();
  }
  if (gated_barriers()) {
    TrackedRange read{src, src_offset, tracked_range_end(src_offset, size)};
    TrackedRange write{dst, dst_offset, tracked_range_end(dst_offset, size)};
    if (!head_synced_ || batch_needs_barrier({&read, 1}, {&write, 1})) {
      record_dependency_barrier();
      trace::counters().barriers_emitted++;
      prof::get().on_barrier(true);
    } else {
      trace::counters().barriers_skipped++;
      prof::get().on_barrier(false);
    }
    tracked_reads_.push_back(read);
    tracked_writes_.push_back(write);
  }
  VkBufferCopy region{};
  region.srcOffset = src_offset;
  region.dstOffset = dst_offset;
  region.size = size;
  vk::device_table().CmdCopyBuffer(cmd_, src, dst, 1, &region);
  node_count_++;
  trace::counters().vk_buffer_copies++;
}

void CommandEncoder::fill_buffer(
    VkBuffer dst,
    uint32_t value,
    VkDeviceSize size,
    VkDeviceSize offset) {
  if (wave_sched()) {
    PendingNode& node = pending_.emplace_back();
    node.kind = PendingNode::Kind::Fill;
    node.prim = trace::current_prim();
    node.dst = dst;
    node.size = size;
    node.write_offset = offset;
    node.value = value;
    node.wr[node.nw++] = {dst, offset, tracked_range_end(offset, size)};
    node_count_++;
    trace::counters().vk_buffer_fills++;
    return;
  }
  ensure_recording();
  if (tape_full_barriers()) {
    record_dependency_barrier();
  }
  if (gated_barriers()) {
    TrackedRange write{dst, offset, tracked_range_end(offset, size)};
    if (!head_synced_ || batch_needs_barrier({}, {&write, 1})) {
      record_dependency_barrier();
      trace::counters().barriers_emitted++;
      prof::get().on_barrier(true);
    } else {
      trace::counters().barriers_skipped++;
      prof::get().on_barrier(false);
    }
    tracked_writes_.push_back(write);
  }
  vk::device_table().CmdFillBuffer(cmd_, dst, offset, size, value);
  node_count_++;
  trace::counters().vk_buffer_fills++;
}

// Allocate one descriptor set from the cached pool. A pool serves up to
// kDescriptorSetsPerPool dispatches; on exhaustion it is retired into the
// currently-recording submission's temporaries, so it is destroyed only
// after that submission completes — which is strictly after every earlier
// submission whose batches allocated sets from it (completion timeline
// values increase along the queue).
VkDescriptorSet CommandEncoder::acquire_descriptor_set(
    ComputeRuntime& compute) {
  auto& dt = vk::device_table();
  // MLX_OMARCHY_TAPE_NO_REUSE (diagnostic, docs/install-omarchy.md):
  // every dispatch gets its own descriptor pool with exactly one set, so
  // no pool - and therefore no set - is shared with any other dispatch.
  // The pool retires into the current submission's temporaries with the
  // same lifetime rule as a retired cached pool below.
  if (tape_no_reuse()) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = compute.binding_limit();
    VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VkDescriptorPool pool{VK_NULL_HANDLE};
    VKX_CHECK(dt.CreateDescriptorPool(
        device_.handle(), &pool_info, nullptr, &pool));
    retired_pools_.push_back(std::shared_ptr<VkDescriptorPool>(
        new VkDescriptorPool(pool),
        [device = device_.handle()](VkDescriptorPool* owned) {
          vk::device_table().DestroyDescriptorPool(device, *owned, nullptr);
          delete owned;
        }));
    VkDescriptorSetLayout descriptor_layout = compute.descriptor_layout();
    VkDescriptorSetAllocateInfo allocate_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate_info.descriptorPool = pool;
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &descriptor_layout;
    VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
    VKX_CHECK(dt.AllocateDescriptorSets(
        device_.handle(), &allocate_info, &descriptor_set));
    return descriptor_set;
  }
  if (desc_pool_ == VK_NULL_HANDLE || desc_pool_remaining_ == 0) {
    if (desc_pool_ != VK_NULL_HANDLE) {
      retired_pools_.push_back(std::shared_ptr<VkDescriptorPool>(
          new VkDescriptorPool(desc_pool_),
          [device = device_.handle()](VkDescriptorPool* owned) {
            vk::device_table().DestroyDescriptorPool(device, *owned, nullptr);
            delete owned;
          }));
      desc_pool_ = VK_NULL_HANDLE;
    }
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount =
        kDescriptorSetsPerPool * compute.binding_limit();
    VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = kDescriptorSetsPerPool;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VKX_CHECK(dt.CreateDescriptorPool(
        device_.handle(), &pool_info, nullptr, &desc_pool_));
    desc_pool_remaining_ = kDescriptorSetsPerPool;
  }
  VkDescriptorSetLayout descriptor_layout = compute.descriptor_layout();
  VkDescriptorSetAllocateInfo allocate_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocate_info.descriptorPool = desc_pool_;
  allocate_info.descriptorSetCount = 1;
  allocate_info.pSetLayouts = &descriptor_layout;
  VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
  VKX_CHECK(dt.AllocateDescriptorSets(
      device_.handle(), &allocate_info, &descriptor_set));
  desc_pool_remaining_--;
  return descriptor_set;
}

void CommandEncoder::dispatch_compute(
    ComputeKernel kernel,
    std::span<const ComputeBinding> bindings,
    const ComputeParams& params,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z) {
  if (group_count_x == 0 || group_count_y == 0 || group_count_z == 0) {
    return;
  }
  auto& compute = device_.compute();
  dispatch_compute_pipeline(
      compute.pipeline(kernel),
      kernel,
      bindings,
      params,
      group_count_x,
      group_count_y,
      group_count_z);
}

void CommandEncoder::dispatch_compute(
    const std::string& cache_key,
    std::span<const uint32_t> spirv,
    std::span<const ComputeBinding> bindings,
    const ComputeParams& params,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z) {
  if (group_count_x == 0 || group_count_y == 0 || group_count_z == 0) {
    return;
  }
  auto& compute = device_.compute();
  dispatch_compute_pipeline(
      compute.pipeline(cache_key, spirv),
      ComputeKernel::Custom,
      bindings,
      params,
      group_count_x,
      group_count_y,
      group_count_z);
}

void CommandEncoder::dispatch_compute_pipeline(
    VkPipeline pipeline,
    ComputeKernel profile_kernel,
    std::span<const ComputeBinding> bindings,
    const ComputeParams& params,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z) {
  if (wave_sched() && debug_repeat_config().any()) {
    throw std::invalid_argument(
        "[omarchy] MLX_OMARCHY_DEBUG_REPEAT_* / MLX_OMARCHY_DEBUG_EMPTY "
        "inject recorded dispatches and require MLX_OMARCHY_WAVE_SCHED=0 "
        "(wave-sched buffers nodes, so the injection would silently "
        "no-op).");
  }
  if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr) {
    fprintf(stderr,
            "[rtmod] DISPATCH kernel=%d count=%u gx=%u gy=%u gz=%u\n",
            static_cast<int>(profile_kernel), params.count, group_count_x,
            group_count_y, group_count_z);
    fflush(stderr);
  }
  auto& compute = device_.compute();
  uint32_t binding_limit = compute.binding_limit();
  if (bindings.empty() || bindings.size() > binding_limit) {
    throw std::invalid_argument(
        "[omarchy] compute dispatch needs " +
        std::to_string(bindings.size()) +
        " storage-buffer bindings; this device allows " +
        std::to_string(binding_limit) + ".");
  }
  for (const auto& item : bindings) {
    note_binding_owner(item.owner);
  }
  group_count_x = std::min(group_count_x, kMaxComputeGroupCountX);
  group_count_y = std::min(group_count_y, kMaxComputeGroupCountX);
  group_count_z = std::min(group_count_z, kMaxComputeGroupCountX);
  // Bounded GPU-time proxy for the work budget (encoder.h): the open
  // batch's summed work-group counts. Each factor is already clamped to
  // kMaxComputeGroupCountX, so the product fits uint64_t; accumulate
  // saturating so a pathologic sequence cannot wrap.
  const uint64_t groups = static_cast<uint64_t>(group_count_x) *
      group_count_y * group_count_z;
  batch_work_ =
      UINT64_MAX - batch_work_ > groups ? batch_work_ + groups : UINT64_MAX;

  auto& dt = vk::device_table();
  VkDescriptorSet descriptor_set = acquire_descriptor_set(compute);

  std::array<VkDescriptorBufferInfo, kComputeBindingBudget> buffer_info{};
  std::array<VkWriteDescriptorSet, kComputeBindingBudget> writes{};
  for (uint32_t index = 0; index < bindings.size(); ++index) {
    buffer_info[index] = {
        bindings[index].buffer, bindings[index].offset, bindings[index].range};
    writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[index].dstSet = descriptor_set;
    writes[index].dstBinding = index;
    writes[index].descriptorCount = 1;
    writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[index].pBufferInfo = &buffer_info[index];
  }
  dt.UpdateDescriptorSets(
      device_.handle(),
      static_cast<uint32_t>(bindings.size()),
      writes.data(),
      0,
      nullptr);
  trace::counters().vk_descriptor_update_writes += bindings.size();

  if (wave_sched()) {
    // Buffer the node; submit() records it in wave order (emit_pending).
    // Everything order-independent happens here: binding-owner pinning,
    // work accounting, descriptor allocation and writing.
    PendingNode& node = pending_.emplace_back();
    node.kind = PendingNode::Kind::Dispatch;
    node.prim = trace::current_prim();
    node.tape = in_tape_recording ? 1u : 0u;
    node.host_t0 = prof::get().profiling() ? prof::host_ns() : 0;
    node.pipeline = pipeline;
    node.profile_kernel = profile_kernel;
    for (size_t i = 0; i < bindings.size(); ++i) {
      node.bindings[i] = bindings[i];
    }
    node.binding_count = static_cast<uint32_t>(bindings.size());
    node.params = params;
    node.group_count_x = group_count_x;
    node.group_count_y = group_count_y;
    node.group_count_z = group_count_z;
    node.descriptor_set = descriptor_set;
    const auto access = compute.binding_access(pipeline);
    for (size_t i = 0; i < bindings.size(); ++i) {
      TrackedRange range{bindings[i].buffer,
                         bindings[i].offset,
                         tracked_range_end(bindings[i].offset, bindings[i].range)};
      if (((access.read_mask >> i) & 1u) != 0u) {
        node.rd[node.nr++] = range;
      }
      if (((access.write_mask >> i) & 1u) != 0u) {
        node.wr[node.nw++] = range;
      }
    }
    node_count_++;
    trace::counters().vk_compute_dispatches++;
    return;
  }

  ensure_recording();
  uint64_t host_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  // MLX_OMARCHY_TAPE_FULL_BARRIERS (diagnostic, docs/install-omarchy.md):
  // the heaviest correct dependency - all commands, all memory access,
  // both directions - ahead of every dispatch, on top of the regular
  // dependency below. Probes whether the driver drops an in-buffer
  // dependency the regular dependency already expresses. It also
  // restarts the gated tracker: the full barrier orders everything
  // recorded before it.
  if (tape_full_barriers()) {
    record_dependency_barrier();
  }
  // MLX_OMARCHY_GATED_BARRIERS (default off): one full barrier before
  // the dispatch only when a binding overlaps unsynced work of the open
  // batch (RAW/WAW against tracked writes, WAR against tracked reads)
  // or when this is the first node of a fresh command buffer. Off: the
  // historic unconditional pre-dispatch barrier. Bindings carry no
  // read/write split, so each binding is tracked as both read and
  // write - the tracker may barrier a read-read pair, never skip a
  // real hazard.
  bool barrier_recorded = false;
  if (gated_barriers()) {
    std::array<TrackedRange, kComputeBindingBudget> ranges{};
    for (size_t i = 0; i < bindings.size(); ++i) {
      ranges[i] = {bindings[i].buffer,
                   bindings[i].offset,
                   tracked_range_end(bindings[i].offset, bindings[i].range)};
    }
    // Per-binding access reflected from the pipeline's SPIR-V (readonly /
    // writeonly block qualifiers): read-only bindings no longer count as
    // writes, so read-read pairs (two consumers of one activation) skip the
    // barrier. Unknown bindings stay read+write. MLX_OMARCHY_DEP_RW=0
    // restores the all-read+write tracking.
    const auto access = compute.binding_access(pipeline);
    std::array<TrackedRange, kComputeBindingBudget> rd{};
    std::array<TrackedRange, kComputeBindingBudget> wr{};
    size_t nr = 0;
    size_t nw = 0;
    for (size_t i = 0; i < bindings.size(); ++i) {
      if (((access.read_mask >> i) & 1u) != 0u) {
        rd[nr++] = ranges[i];
      }
      if (((access.write_mask >> i) & 1u) != 0u) {
        wr[nw++] = ranges[i];
      }
    }
    if (!head_synced_ ||
        batch_needs_barrier({rd.data(), nr}, {wr.data(), nw})) {
      record_dependency_barrier();
      trace::counters().barriers_emitted++;
      prof::get().on_barrier(true);
      barrier_recorded = true;
    } else {
      trace::counters().barriers_skipped++;
      prof::get().on_barrier(false);
    }
    for (size_t i = 0; i < nr; ++i) {
      tracked_reads_.push_back(rd[i]);
    }
    for (size_t i = 0; i < nw; ++i) {
      tracked_writes_.push_back(wr[i]);
    }
    if (export_dep_masks()) {
      // Compute the two exported signals (design 22b395d): (a) whether
      // this dispatch's ranges are provably disjoint from everything
      // since the last explicit dependency (same proof
      // batch_needs_barrier just ran, inverted); (b) whether the launch
      // reuses the previous dispatch's binding-layout signature (the
      // mlx-level analogue of the driver's USC register configuration).
      uint32_t usc_signature = 0;
      for (size_t i = 0; i < bindings.size(); ++i) {
        usc_signature = usc_signature * 31 +
                        uint32_t((uintptr_t)ranges[i].buffer ^ ranges[i].offset ^
                                 ranges[i].end);
      }
      uint32_t usc_changed =
          (dep_have_prev_ && usc_signature != dep_prev_signature_) ? 1 : 0;
      dep_records_.push_back({static_cast<uint32_t>(barrier_recorded ? 0 : 1),
                              usc_changed, ++dep_seq_});
      dep_prev_signature_ = usc_signature;
      dep_have_prev_ = 1;
    }
  } else {
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask =
        VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;
    before.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    dt.CmdPipelineBarrier(
        cmd_,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &before,
        0,
        nullptr,
        0,
        nullptr);
    head_synced_ = true;
    trace::counters().barriers_emitted++;
    prof::get().on_barrier(true);
    barrier_recorded = true;
  }
  prof::get().before_dispatch(this, current_slot_, cmd_, barrier_recorded);

  VkPipelineLayout pipeline_layout = compute.pipeline_layout();
  dt.CmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  dt.CmdBindDescriptorSets(
      cmd_,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipeline_layout,
      0,
      1,
      &descriptor_set,
      0,
      nullptr);
  dt.CmdPushConstants(
      cmd_,
      pipeline_layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(params),
      &params);
  dt.CmdDispatch(cmd_, group_count_x, group_count_y, group_count_z);
  prof::get().after_dispatch(
      this,
      current_slot_,
      cmd_,
      profile_kernel,
      params,
      bindings,
      group_count_x,
      group_count_y,
      group_count_z,
      host_t0 != 0 ? prof::host_ns() - host_t0 : 0,
      in_tape_recording ? 1u : 0u);

  // Gated mode tracks this dispatch's writes instead of recording a
  // post barrier; the next node's overlap test consumes the tracking.
  if (!gated_barriers()) {
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
        VK_ACCESS_HOST_READ_BIT;
    dt.CmdPipelineBarrier(
        cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
            VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &after,
        0,
        nullptr,
        0,
        nullptr);
    trace::counters().barriers_emitted++;
    prof::get().on_barrier(true);
  }
  if (tape_full_barriers()) {
    // Diagnostic: matching full barrier out of this dispatch, so every
    // dependency between two dispatches is the heaviest form (see the
    // pre-dispatch barrier above).
    record_dependency_barrier();
  }

  node_count_++;
  trace::counters().vk_compute_dispatches++;

  // Injected-work sensitivity probe (diagnostic, default off): the
  // real dispatch above left the pipeline, descriptor set, and push
  // constants bound, so a repeat is exactly one more CmdDispatch of
  // the same command — the kernel recomputes identical values from
  // unchanged inputs into the same outputs. Not counted anywhere:
  // no profiler node, no tracker range, no work-budget, no dispatch
  // counter.
  const auto& dbg = debug_repeat_config();
  if (dbg.any()) {
    uint8_t cls = static_cast<uint8_t>(debug_class_of(profile_kernel));
    uint32_t total = cls < (static_cast<uint8_t>(DebugClass::Count) - 1)
        ? dbg.repeat[cls]
        : 0;
    for (uint32_t r = 1; r < total; ++r) {
      dt.CmdDispatch(cmd_, group_count_x, group_count_y, group_count_z);
    }
    if (dbg.empty > 0) {
      VkPipeline null_pipeline = compute.pipeline(ComputeKernel::DebugNull);
      // The null shader touches nothing; bind an allocated (never
      // written) set so no unbound-descriptor edge exists.
      VkDescriptorSet null_set = acquire_descriptor_set(compute);
      dt.CmdBindPipeline(
          cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, null_pipeline);
      dt.CmdBindDescriptorSets(
          cmd_,
          VK_PIPELINE_BIND_POINT_COMPUTE,
          compute.pipeline_layout(),
          0,
          1,
          &null_set,
          0,
          nullptr);
      for (uint32_t r = 0; r < dbg.empty; ++r) {
        dt.CmdDispatch(cmd_, 1, 1, 1);
      }
    }
  }
}

void CommandEncoder::emit_pending() {
  if (pending_.empty()) {
    return;
  }
  auto& dt = vk::device_table();
  auto& compute = device_.compute();
  ensure_recording();

  std::vector<WaveNode> nodes;
  nodes.reserve(pending_.size());
  for (const auto& p : pending_) {
    nodes.push_back({{p.rd.data(), p.nr}, {p.wr.data(), p.nw}});
  }
  const std::vector<uint32_t> levels = wave_levels(nodes);
  uint32_t waves = 0;
  for (uint32_t l : levels) {
    waves = std::max(waves, l);
  }

  if (wave_diag()) {
    // Classify the level-determining edge of every node: which access
    // class (RAW/WAW/WAR) against the deepest hazard source forces this
    // node into its wave. Nodes at level 0 are unforced.
    auto overlaps = [](const TrackedRange& a, const TrackedRange& b) {
      return a.buffer == b.buffer && a.offset < b.end && b.offset < a.end;
    };
    uint64_t hist[4] = {0, 0, 0, 0}; // none, raw, waw, war
    uint64_t depth_sum = 0;
    for (size_t i = 0; i < pending_.size(); ++i) {
      depth_sum += levels[i];
      if (levels[i] == 0) {
        hist[0]++;
        continue;
      }
      uint32_t best = 0;
      for (size_t j = 0; j < i; ++j) {
        if (levels[j] + 1 == levels[i]) {
          best = static_cast<uint32_t>(j);
          // keep the LAST node one wave below; any is representative
        }
      }
      const WaveNode& src = nodes[best];
      bool raw = false, waw = false, war = false;
      for (const auto& r : nodes[i].reads) {
        for (const auto& w : src.writes) {
          raw = raw || overlaps(r, w);
        }
      }
      for (const auto& w : nodes[i].writes) {
        for (const auto& tw : src.writes) {
          waw = waw || overlaps(w, tw);
        }
        for (const auto& tr : src.reads) {
          war = war || overlaps(w, tr);
        }
      }
      hist[raw ? 1 : (waw ? 2 : (war ? 3 : 0))]++;
    }
    std::fprintf(
        stderr,
        "[wave-diag] nodes=%zu waves=%u levels_sum=%llu class "
        "none=%llu raw=%llu waw=%llu war=%llu\n",
        pending_.size(),
        waves + 1,
        static_cast<unsigned long long>(depth_sum),
        static_cast<unsigned long long>(hist[0]),
        static_cast<unsigned long long>(hist[1]),
        static_cast<unsigned long long>(hist[2]),
        static_cast<unsigned long long>(hist[3]));
    std::fflush(stderr);
  }

  // The batch-head dependency: a freshly begun command buffer records its
  // first barrier before any command (host writes, allocator flush, the
  // previous submission's writes), exactly like the tape-order path.
  const bool head_needs_barrier = !head_synced_;
  const std::string_view submit_prim = trace::current_prim();

  for (uint32_t w = 0; w <= waves; ++w) {
    bool barrier_owed = (w > 0) || head_needs_barrier;
    for (size_t i = 0; i < pending_.size(); ++i) {
      if (levels[i] != w) {
        continue;
      }
      // One full dependency barrier per wave, on the wave head; every
      // later node of the wave rides it hazard-free.
      bool preceded_by_barrier = barrier_owed;
      if (barrier_owed) {
        record_dependency_barrier();
        trace::counters().barriers_emitted++;
        prof::get().on_barrier(true);
        barrier_owed = false;
      } else {
        trace::counters().barriers_skipped++;
        prof::get().on_barrier(false);
      }
      const PendingNode& node = pending_[i];
      if (node.kind == PendingNode::Kind::Dispatch) {
        if (tape_full_barriers()) {
          // Diagnostic: the heaviest dependency before every dispatch
          // (degenerates the wave schedule; not counted as a decision).
          record_dependency_barrier();
          preceded_by_barrier = true;
        }
        prof::get().before_dispatch(
            this, current_slot_, cmd_, preceded_by_barrier);
        VkPipelineLayout pipeline_layout = compute.pipeline_layout();
        dt.CmdBindPipeline(
            cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, node.pipeline);
        dt.CmdBindDescriptorSets(
            cmd_,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            0,
            1,
            &node.descriptor_set,
            0,
            nullptr);
        dt.CmdPushConstants(
            cmd_,
            pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(node.params),
            &node.params);
        dt.CmdDispatch(
            cmd_,
            node.group_count_x,
            node.group_count_y,
            node.group_count_z);
        trace::current_prim() = node.prim;
        prof::get().after_dispatch(
            this,
            current_slot_,
            cmd_,
            node.profile_kernel,
            node.params,
            std::span<const ComputeBinding>(
                node.bindings.data(), node.binding_count),
            node.group_count_x,
            node.group_count_y,
            node.group_count_z,
            node.host_t0 != 0 ? prof::host_ns() - node.host_t0 : 0,
            node.tape);
        trace::current_prim() = submit_prim;
        if (export_dep_masks()) {
          // A node that rode its wave's barrier is provably disjoint from
          // everything since that barrier (the level assignment proved no
          // hazard against any earlier node); a wave-head node recorded
          // the barrier itself.
          uint32_t usc_signature = 0;
          for (uint32_t b = 0; b < node.nr + node.nw; ++b) {
            const TrackedRange& r =
                b < node.nr ? node.rd[b] : node.wr[b - node.nr];
            usc_signature = usc_signature * 31 +
                uint32_t((uintptr_t)r.buffer ^ r.offset ^ r.end);
          }
          dep_records_.push_back(
              {static_cast<uint32_t>(preceded_by_barrier ? 0 : 1),
               static_cast<uint32_t>(
                   (dep_have_prev_ && usc_signature != dep_prev_signature_)
                       ? 1
                       : 0),
               ++dep_seq_});
          dep_prev_signature_ = usc_signature;
          dep_have_prev_ = 1;
        }
      } else if (node.kind == PendingNode::Kind::Copy) {
        VkBufferCopy region{};
        region.srcOffset = node.src_offset;
        region.dstOffset = node.dst_offset;
        region.size = node.size;
        dt.CmdCopyBuffer(cmd_, node.src, node.dst, 1, &region);
      } else {
        dt.CmdFillBuffer(
            cmd_, node.dst, node.write_offset, node.size, node.value);
      }
      for (uint32_t r = 0; r < node.nr; ++r) {
        tracked_reads_.push_back(node.rd[r]);
      }
      for (uint32_t r = 0; r < node.nw; ++r) {
        tracked_writes_.push_back(node.wr[r]);
      }
    }
  }
  pending_.clear();
}

void CommandEncoder::commit() {
  if (!recording_ && pending_.empty() && wait_semaphores_.empty() &&
      signal_semaphores_.empty() && completed_handlers_.empty()) {
    trace::counters().commit_calls_noop++;
    if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH")) {
      fprintf(stderr, "[rtmod] COMMIT-NOOP\n");
    }
    return;
  }
  trace::counters().commit_calls_with_work++;
  submit();
}

void CommandEncoder::synchronize(const char* reason) {
  commit();
  join_last_completion(reason);
}

void CommandEncoder::wait_outstanding_submissions() {
  // Timeline wait on the device completion semaphore at the newest
  // reserved value: the next submission this encoder commits starts only
  // after every submission reserved so far, on any stream, has completed.
  // The wait rides that next submission, so no extra submission is
  // created and the host never blocks. Reading the value here is safe
  // against self-deadlock: this stream's own next completion value is
  // reserved strictly later (at commit), so the waited value always
  // belongs to a submission queued ahead of ours.
  uint64_t value = device_.completions().last_reserved();
  if (value == 0) {
    return;
  }
  add_semaphore_wait(device_.completions().semaphore(), value, nullptr);
}

void CommandEncoder::submit() {
  auto& dt = vk::device_table();
  if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH")) {
    fprintf(stderr, "[rtmod] SUBMIT-ENTER tid=%lu\n", (unsigned long)syscall(SYS_gettid));
  }
  // Wave scheduling: record the buffered nodes into the command buffer
  // now, in wave order, before the readback barrier and EndCommandBuffer.
  emit_pending();
  bool was_recording = recording_;
  uint64_t submit_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  uint64_t close_t = 0;
  uint64_t queue_t0 = 0;
  uint64_t queue_t1 = 0;
  uint64_t submitted = 0;

  if (recording_) {
    if (gated_barriers()) {
      VkMemoryBarrier readback{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      readback.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
      readback.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      dt.CmdPipelineBarrier(
          cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &readback,
          0, nullptr, 0, nullptr);
      trace::counters().barriers_emitted++;
      prof::get().on_barrier(true);
    }
    VKX_CHECK(dt.EndCommandBuffer(cmd_));
    close_t = prof::get().profiling() ? prof::host_ns() : 0;
  }

  std::vector<VkSemaphore> wait_sems;
  std::vector<uint64_t> wait_values;
  wait_sems.reserve(wait_semaphores_.size() + 1);
  wait_values.reserve(wait_semaphores_.size() + 1);
  // In-order stream: this submission waits for the stream's previous
  // submission. Vulkan defines no execution or memory dependency between
  // submissions without a semaphore wait, and the per-dispatch barriers
  // cover hazards inside one command buffer only. Without this wait, a
  // submission queued behind long work can read a prior tiny submission's
  // output before that write lands (Honeykrisp: compiled 4-bit decode
  // read an eager one-element f32 as recycled page garbage, 20/20).
  // Waiting on an already-signaled timeline value is a driver
  // pass-through, so the shallow-queue case pays nothing.
  if (last_completion_ != 0) {
    wait_sems.push_back(device_.completions().semaphore());
    wait_values.push_back(last_completion_);
  }
  for (auto& pending : wait_semaphores_) {
    wait_sems.push_back(pending.semaphore);
    wait_values.push_back(pending.value);
  }
  std::vector<VkPipelineStageFlags> wait_stages(
      wait_sems.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

  VkTimelineSemaphoreSubmitInfo timeline{
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  timeline.waitSemaphoreValueCount = static_cast<uint32_t>(wait_values.size());
  timeline.pWaitSemaphoreValues = wait_values.data();

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.pNext = &timeline;
  si.waitSemaphoreCount = static_cast<uint32_t>(wait_sems.size());
  si.pWaitSemaphores = wait_sems.data();
  si.pWaitDstStageMask = wait_stages.data();
  si.commandBufferCount = recording_ ? 1u : 0u;
  si.pCommandBuffers = recording_ ? &cmd_ : nullptr;

  // Every submission also signals the device completion timeline; user
  // (event) signals ride along in the same submission, in order.
  std::vector<VkSemaphore> signal_sems;
  std::vector<uint64_t> signal_values;
  signal_sems.reserve(signal_semaphores_.size() + 1);
  signal_values.reserve(signal_semaphores_.size() + 1);
  for (auto& pending : signal_semaphores_) {
    signal_sems.push_back(pending.semaphore);
    signal_values.push_back(pending.value);
  }

  // Ownership moves to the dispatcher entry: each pending-semaphore
  // keepalive keeps its Event alive, and handlers run on the completion
  // thread. The batch's buffers need no shared_ptr: they were stamped
  // with the completion value below, and the allocator quarantine keeps
  // any freed-but-in-flight buffer out of the reuse cache until its
  // generation drains.
  std::vector<std::shared_ptr<void>> keepalive;
  for (auto& pool : retired_pools_) {
    keepalive.push_back(std::move(pool));
  }
  retired_pools_.clear();
  for (auto& pending : wait_semaphores_) {
    keepalive.push_back(std::move(pending.keepalive));
  }
  for (auto& pending : signal_semaphores_) {
    keepalive.push_back(std::move(pending.keepalive));
  }

  {
    // VkQueue is externally synchronized: the lock covers completion-value
    // assignment (timeline signals must increase along the queue) through
    // QueueSubmit and the dispatcher enqueue. It is never held across a
    // host wait.
    std::lock_guard<std::mutex> lk(device_.queue_mutex());
    // Khronos guidance: HOST_VISIBLE memory without HOST_COHERENT needs an
    // explicit flush before submission.
    omarchy::allocator().flush_noncoherent(device_.handle());
    uint64_t completion_value = device_.completions().reserve();
    // Stamp every buffer referenced by this batch with the completion
    // value just reserved. Free-before-drain sends such buffers to the
    // allocator quarantine; release_quarantine recycles them one
    // generation later. Semaphore keepalives still move into the
    // dispatcher payload below.
    omarchy::allocator().stamp_batch(batch_buffers_, completion_value);
    batch_buffers_.clear();
    VkSemaphore completion_sem = device_.completions().semaphore();
    signal_sems.push_back(completion_sem);
    signal_values.push_back(completion_value);
    timeline.signalSemaphoreValueCount =
        static_cast<uint32_t>(signal_values.size());
    timeline.pSignalSemaphoreValues = signal_values.data();
    si.signalSemaphoreCount = static_cast<uint32_t>(signal_sems.size());
    si.pSignalSemaphores = signal_sems.data();
    try {
      queue_t0 = prof::get().profiling() ? prof::host_ns() : 0;
      // Deterministic swallow simulation for the recovery ladder, in the
      // two field-observed classes:
      //
      // MLX_OMARCHY_TEST_DROP_SUBMIT drops the Nth (1-based) QueueSubmit
      // entirely, the way Honeykrisp drops submissions in burst windows:
      // the batch never begins, so its started event never sets and
      // recovery rung 1 (kick + resubmit at fresh values) must fire.
      //
      // MLX_OMARCHY_TEST_DROP_SIGNAL submits the Nth batch but strips
      // every timeline signal from it (ride-along user events AND the
      // device completion): the kernels execute and the started event
      // sets, but no signal ever publishes - the executed-but-unsignaled
      // class - so recovery rung 2 (host vkSignalSemaphore) must fire.
      //
      // Both share one submit sequence; the host still publishes the
      // completion entry, so the watchdog waits on a signal the GPU will
      // never (submit-drop) / did not (signal-strip) deliver. Comma-
      // separated 1-based ordinals: "1" hits the first submit, "1,2" the
      // first two (consecutive-swallow recovery proof).
      static std::atomic<uint64_t> submit_sequence{0};
      uint64_t ordinal = submit_sequence.fetch_add(1) + 1;
      auto env_hits_ordinal = [ordinal](const char* name) {
        const char* e = std::getenv(name);
        if (!e) {
          return false;
        }
        char* p = const_cast<char*>(e);
        while (*p) {
          if (strtoull(p, &p, 10) == ordinal) {
            return true;
          }
          if (*p == ',') {
            ++p;
          } else {
            break;
          }
        }
        return false;
      };
      bool simulate_drop = env_hits_ordinal("MLX_OMARCHY_TEST_DROP_SUBMIT");
      bool simulate_strip =
          !simulate_drop && env_hits_ordinal("MLX_OMARCHY_TEST_DROP_SIGNAL");
      if (simulate_strip) {
        si.signalSemaphoreCount = 0;
        si.pSignalSemaphores = nullptr;
        timeline.signalSemaphoreValueCount = 0;
        timeline.pSignalSemaphoreValues = nullptr;
      }
      if (simulate_drop) {
        fprintf(stderr, "[rtmod] TEST-DROP tid=%lu cv=%lu\n",
                (unsigned long)syscall(SYS_gettid), (unsigned long)completion_value);
      } else if (simulate_strip) {
        fprintf(stderr, "[rtmod] TEST-STRIP tid=%lu cv=%lu\n",
                (unsigned long)syscall(SYS_gettid), (unsigned long)completion_value);
      } else {
        VKX_CHECK(dt.QueueSubmit(device_.queue(), 1, &si, VK_NULL_HANDLE));
      }
      queue_t1 = prof::get().profiling() ? prof::host_ns() : 0;
    } catch (...) {
      // The submission never reached the driver: the ended command buffer
      // and the pending semaphore lists are dead (their keepalives have
      // already moved into the local payload and die with this frame).
      // Reset the encoder so it can be reused or destroyed cleanly; the
      // typed error propagates to the stream's error handling. The
      // batch's buffer stamps are harmless: those buffers stay alive via
      // their arrays or the quarantine.
      batch_buffers_.clear();
      pending_.clear();
      recording_ = false;
      node_count_ = 0;
      batch_work_ = 0;
      wait_semaphores_.clear();
      signal_semaphores_.clear();
      completed_handlers_.clear();
      reset_dependency_tracking();
      // The reserved completion value can never signal GPU-side: without
      // a pending entry, later joins would block on drained_value_
      // forever. Publish the empty entry - its waiters get the typed
      // watchdog error instead of an unbounded join (reserved values must
      // always have a completion entry).
      device_.completions().enqueue(completion_value, {}, {});
      throw;
    }
    // Publish only after the submit: the dispatcher must never wait on a
    // value whose submission has not been handed to the driver.
    // Retain everything the batch put on the queue so the watchdog's
    // recovery ladder can resubmit it if Honeykrisp drops the submission
    // (erased when the completion drains).
    {
      CompletionDispatcher::ResubmitBatch batch;
      batch.value = completion_value;
      batch.cmd = recording_ ? cmd_ : VK_NULL_HANDLE;
      batch.wait_sems = wait_sems;
      batch.wait_values = wait_values;
      batch.signal_sems = signal_sems;
      batch.signal_values = signal_values;
      device_.completions().retain_for_resubmit(
          completion_value, std::move(batch));
    }
    device_.completions().enqueue(
        completion_value,
        std::move(keepalive),
        std::move(completed_handlers_),
        was_recording ? slots_[current_slot_].started : VK_NULL_HANDLE);
    last_completion_ = completion_value;
    submitted = completion_value;
    if (was_recording) {
      slots_[current_slot_].in_flight = completion_value;
    }
    trace::counters().vk_submissions++;
    if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH")) {
      fprintf(stderr,
              "[rtmod] SUBMIT tid=%lu cv=%lu waits=%lu sigs=%lu cmds=%u\n",
              (unsigned long)syscall(SYS_gettid),
              (unsigned long)completion_value,
              (unsigned long)wait_sems.size(),
              (unsigned long)signal_values.size(),
              (unsigned)si.commandBufferCount);
    }
  }

  recording_ = false;
  node_count_ = 0;
  batch_work_ = 0;
  pending_.clear();
  wait_semaphores_.clear();
  signal_semaphores_.clear();
  completed_handlers_.clear();
  // The submission's in-order wait (last_completion_) provides the
  // cross-submission dependency, so the open batch's unsynced ranges
  // die here either way.
  reset_dependency_tracking();
  prof::get().on_submit_boundary(submitted, close_t, queue_t0, queue_t1);
  prof::get().on_submit_end(
      this,
      submitted,
      submit_t0 != 0 ? prof::host_ns() - submit_t0 : 0,
      current_slot_);

  // No vkResetCommandBuffer: the buffer may still be executing. The ring
  // slot is marked in flight above; ensure_recording() only reuses a slot
  // whose submission completed (or joins the oldest), and BeginCommandBuffer
  // then resets the buffer implicitly (the pool was created with
  // RESET_COMMAND_BUFFER_BIT).
}

CommandEncoder& get_command_encoder(Stream s) {
  // Mirrors the CUDA backend: the per-thread table misses for a stream
  // created on another thread, so fall back to the global thread-unsafe
  // table and finally raise the upstream std::runtime_error contract.
  // unordered_map::at would throw std::out_of_range, which escapes the
  // caller's catch(std::runtime_error) and terminates the process.
  auto& encoders = get_command_encoders();
  auto it = encoders.find(s.index);
  if (it == encoders.end()) {
    auto& global_encoders = get_global_command_encoders();
    it = global_encoders.find(s.index);
    if (it == global_encoders.end()) {
      throw std::runtime_error(
          "There is no Stream(gpu, " + std::to_string(s.index) +
          ") in current thread.");
    }
  }
  return it->second;
}

std::unordered_map<int, CommandEncoder>& get_command_encoders() {
  static thread_local std::unordered_map<int, CommandEncoder> encoders;
  return encoders;
}

std::unordered_map<int, CommandEncoder>& get_global_command_encoders() {
  static std::unordered_map<int, CommandEncoder> global_encoders;
  return global_encoders;
}

} // namespace mlx::core::omarchy
