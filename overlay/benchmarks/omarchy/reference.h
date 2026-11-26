// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// CPU reference implementations used to validate every spike kernel before
// it is timed. The Q4_K helpers mirror the exact block layout of the pinned
// llama.cpp revision (f280b26983ad0fdb705a0d9ebf0503e76f2899b0):
// 256 elements per 144-byte block; d/dmin fp16; 12 packed 6-bit scale/min
// bytes via get_scale_min_k4; qs[128] nibbles.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace omarchy_spike::ref {

inline uint16_t f32_to_f16(float value) {
  uint32_t x;
  std::memcpy(&x, &value, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int32_t biased = int32_t((x >> 23) & 0xff);
  uint32_t mant = x & 0x7fffff;
  if (biased == 0xff) {
    return uint16_t(sign | 0x7c00 | (mant ? 0x200 : 0));
  }
  if (biased == 0) {
    return uint16_t(sign);
  }
  int32_t exp = biased - 127 + 15;
  if (exp >= 0x1f) {
    return uint16_t(sign | 0x7c00);
  }
  if (exp <= 0) {
    if (exp < -10) {
      return uint16_t(sign);
    }
    mant |= 0x800000;
    int shift = 14 - exp;
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1);
    uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1))) {
      half++;
    }
    return uint16_t(sign | half);
  }
  uint32_t half = (uint32_t(exp) << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) {
    half++;
  }
  return uint16_t(sign | half);
}

inline float f16_to_f32(uint16_t h) {
  uint32_t sign = uint32_t(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3ff;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000 | (mant << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// --- Plain kernels -----------------------------------------------------------

inline void gemm(
    const std::vector<float>& a,
    const std::vector<float>& w,
    std::vector<float>& c,
    uint32_t m,
    uint32_t n,
    uint32_t k) {
  c.assign(size_t(m) * n, 0.0f);
  for (uint32_t i = 0; i < m; i++) {
    for (uint32_t j = 0; j < n; j++) {
      double acc = 0.0;
      for (uint32_t l = 0; l < k; l++) {
        acc += double(a[size_t(i) * k + l]) * double(w[size_t(j) * k + l]);
      }
      c[size_t(i) * n + j] = float(acc);
    }
  }
}

inline void gemv(
    const std::vector<float>& x,
    const std::vector<float>& w,
    std::vector<float>& out,
    uint32_t n,
    uint32_t k) {
  out.assign(n, 0.0f);
  for (uint32_t j = 0; j < n; j++) {
    double acc = 0.0;
    for (uint32_t l = 0; l < k; l++) {
      acc += double(x[l]) * double(w[size_t(j) * k + l]);
    }
    out[j] = float(acc);
  }
}

// --- Q4_K --------------------------------------------------------------------

constexpr uint32_t kQkK = 256;
constexpr uint32_t kQ4kWordsPerBlock = 36; // 144 bytes

inline uint32_t
q4k_scale_byte(const std::vector<uint32_t>& words, uint32_t base, uint32_t j) {
  uint32_t word = words[base + 1 + j / 4];
  return (word >> (8 * (j % 4))) & 0xFF;
}

// get_scale_min_k4 from the pinned ggml revision.
inline void q4k_get_scale_min_k4(
    uint32_t j,
    const std::vector<uint32_t>& words,
    uint32_t base,
    uint32_t& sc,
    uint32_t& mn) {
  if (j < 4) {
    sc = q4k_scale_byte(words, base, j) & 63;
    mn = q4k_scale_byte(words, base, j + 4) & 63;
  } else {
    sc = (q4k_scale_byte(words, base, j + 4) & 0xF) |
        ((q4k_scale_byte(words, base, j - 4) >> 6) << 4);
    mn = (q4k_scale_byte(words, base, j + 4) >> 4) |
        ((q4k_scale_byte(words, base, j) >> 6) << 4);
  }
}

// Dequantize one row of packed Q4_K words into fp32.
inline void q4k_dequant_row(
    const std::vector<uint32_t>& words,
    std::vector<float>& out,
    uint32_t k) {
  uint32_t nb = k / kQkK;
  out.assign(k, 0.0f);
  for (uint32_t b = 0; b < nb; b++) {
    uint32_t base = b * kQ4kWordsPerBlock;
    uint32_t word = words[base];
    float d = f16_to_f32(word & 0xFFFF);
    float dmin = f16_to_f32(word >> 16);
    for (uint32_t pair = 0; pair < 4; pair++) {
      uint32_t sc, mn;
      q4k_get_scale_min_k4(2 * pair, words, base, sc, mn);
      float d1 = d * float(sc);
      float m1 = dmin * float(mn);
      q4k_get_scale_min_k4(2 * pair + 1, words, base, sc, mn);
      float d2 = d * float(sc);
      float m2 = dmin * float(mn);
      for (uint32_t m = 8 * pair; m < 8 * pair + 8; m++) {
        uint32_t w = words[base + 4 + m];
        for (uint32_t lane = 0; lane < 4; lane++) {
          uint32_t shift = 8 * lane;
          uint32_t byte = (w >> shift) & 0xFF;
          uint32_t e = 4 * (m % 8) + lane;
          out[256 * b + 64 * pair + e] = d1 * float(byte & 0xF) - m1;
          out[256 * b + 64 * pair + 32 + e] = d2 * float(byte >> 4) - m2;
        }
      }
    }
  }
}

// Synthesize one row of random Q4_K blocks with the packed layout.
inline std::vector<uint32_t> q4k_make_row(std::mt19937& rng, uint32_t k) {
  uint32_t nb = k / kQkK;
  std::vector<uint32_t> words(size_t(nb) * kQ4kWordsPerBlock, 0);
  std::uniform_int_distribution<uint32_t> nib(0, 15);
  std::uniform_int_distribution<uint32_t> sc6(0, 63);
  for (uint32_t b = 0; b < nb; b++) {
    uint32_t base = b * kQ4kWordsPerBlock;
    float d = 0.25f + 0.25f * float(b % 7);
    float dmin = 0.5f + 0.125f * float(b % 5);
    words[base] = uint32_t(f32_to_f16(d)) | (uint32_t(f32_to_f16(dmin)) << 16);
    // 12 scale bytes packed into words 1..3.
    for (uint32_t j = 0; j < 12; j++) {
      uint32_t byte = sc6(rng);
      words[base + 1 + j / 4] |= byte << (8 * (j % 4));
    }
    // 128 qs bytes packed into words 4..35.
    for (uint32_t m = 0; m < 32; m++) {
      uint32_t word = 0;
      for (uint32_t lane = 0; lane < 4; lane++) {
        word |= nib(rng) << (8 * lane);
        word |= nib(rng) << (8 * lane + 4);
      }
      words[base + 4 + m] = word;
    }
  }
  return words;
}

// --- Attention ---------------------------------------------------------------

// Naive SDPA with GQA in fp64 accumulation. q is [m,h,hd] (or [h,hd] when
// m == 1), k/v are [kv,hkv,hd], o matches q. causal=true limits each query
// row to keys t <= row (prefill); causal=false attends the full KV cache
// (decode: the single query sees every cached token).
inline void sdpa(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    std::vector<float>& o,
    uint32_t m,
    uint32_t kv,
    uint32_t hd,
    uint32_t h,
    uint32_t hkv,
    double scale,
    bool causal = true) {
  o.assign(q.size(), 0.0f);
  double log2e = 1.4426950408889634;
  for (uint32_t row = 0; row < m; row++) {
    for (uint32_t head = 0; head < h; head++) {
      uint32_t hk = head / (h / hkv);
      std::vector<double> scores(kv, -1e30);
      double mmax = -1e30;
      for (uint32_t t = 0; t < kv; t++) {
        if (causal && t > row) {
          break;
        }
        double s = 0.0;
        for (uint32_t d = 0; d < hd; d++) {
          s += double(q[(size_t(row) * h + head) * hd + d]) *
              double(k[(size_t(t) * hkv + hk) * hd + d]);
        }
        scores[t] = s * scale;
        mmax = std::max(mmax, scores[t]);
      }
      double lsum = 0.0;
      for (uint32_t t = 0; t < kv; t++) {
        if (causal && t > row) {
          break;
        }
        scores[t] = std::exp2((scores[t] - mmax) * log2e);
        lsum += scores[t];
      }
      for (uint32_t d = 0; d < hd; d++) {
        double acc = 0.0;
        for (uint32_t t = 0; t < kv; t++) {
          if (causal && t > row) {
            break;
          }
          acc += scores[t] * double(v[(size_t(t) * hkv + hk) * hd + d]);
        }
        o[(size_t(row) * h + head) * hd + d] = float(acc / lsum);
      }
    }
  }
}

// Faithful fp64 simulation of the gemm.comp tile algorithm, including the
// shared-tile loads. Positive control: the fixed transposed W load
// (tb[ly][lx] = W[col, kt+ly]) reproduces the naive reference. Negative
// control: the historical W load indexed K by lx (making every kk in the
// consume loop read the same element) must disagree.
inline void gemm_tile_simulate(
    const std::vector<float>& a,
    const std::vector<float>& w,
    std::vector<float>& c,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    bool legacy_w_load) {
  c.assign(size_t(m) * n, 0.0);
  for (uint32_t gy = 0; gy < (m + 15u) / 16u; gy++) {
    for (uint32_t gx = 0; gx < (n + 15u) / 16u; gx++) {
      double acc[16][16] = {};
      for (uint32_t kt = 0; kt < k; kt += 16u) {
        double ta[16][16], tb[16][16];
        for (uint32_t ly = 0; ly < 16u; ly++) {
          for (uint32_t lx = 0; lx < 16u; lx++) {
            uint32_t row = gy * 16u + ly;
            uint32_t col = gx * 16u + lx;
            ta[ly][lx] = (row < m && kt + lx < k)
                ? double(a[size_t(row) * k + kt + lx])
                : 0.0;
            if (legacy_w_load) {
              tb[ly][lx] = (col < n && kt + lx < k)
                  ? double(w[size_t(col) * k + kt + lx])
                  : 0.0;
            } else {
              tb[ly][lx] = (col < n && kt + ly < k)
                  ? double(w[size_t(col) * k + kt + ly])
                  : 0.0;
            }
          }
        }
        for (uint32_t ly = 0; ly < 16u; ly++) {
          for (uint32_t lx = 0; lx < 16u; lx++) {
            for (uint32_t kk = 0; kk < 16u; kk++) {
              acc[ly][lx] += ta[ly][kk] * tb[kk][lx];
            }
          }
        }
      }
      for (uint32_t ly = 0; ly < 16u; ly++) {
        for (uint32_t lx = 0; lx < 16u; lx++) {
          uint32_t row = gy * 16u + ly;
          uint32_t col = gx * 16u + lx;
          if (row < m && col < n) {
            c[size_t(row) * n + col] = float(acc[ly][lx]);
          }
        }
      }
    }
  }
}

// Faithful fp64 simulation of the sdpa_decode kernel algorithm, including
// the shared-memory reduction gather. Used as a positive control (fixed
// reduction matches the naive reference) and a negative control (the
// historical spart[4*j+c] gather must disagree), so a regression to the
// broken reduction is caught before any hardware timing.
inline void sdpa_decode_simulate(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    std::vector<float>& o,
    uint32_t kv,
    uint32_t hd,
    uint32_t h,
    uint32_t hkv,
    double scale,
    bool legacy_reduction) {
  o.assign(size_t(h) * hd, 0.0f);
  const double log2e = 1.4426950408889634;
  for (uint32_t head = 0; head < h; head++) {
    uint32_t hk = head / (h / hkv);
    double mrun = -1e30, lrun = 0.0;
    std::vector<double> acc(hd, 0.0);
    uint32_t tiles = (kv + 31u) / 32u;
    for (uint32_t tile = 0; tile < tiles; tile++) {
      uint32_t t0 = 32u * tile;
      // Thread (chunk c, elem e) holds the dot over hd slice
      // [c*64 + 2e, c*64 + 2e + 1]; partial slot is c*32 + (kv row).
      std::vector<double> spart(128, 0.0);
      for (uint32_t c = 0; c < 4u; c++) {
        for (uint32_t e = 0; e < 32u; e++) {
          uint32_t t = t0 + e;
          if (t >= kv)
            continue;
          double p = 0.0;
          for (uint32_t i = 0; i < 64u; i++) {
            uint32_t d = c * 64u + i;
            p += double(q[head * hd + d]) *
                double(k[(size_t(t) * hkv + hk) * hd + d]);
          }
          spart[c * 32u + e] = p;
        }
      }
      // Reduction over the 4 chunk partials for each kv row.
      std::vector<double> srow(32, -1e30);
      for (uint32_t j = 0; j < 32u; j++) {
        uint32_t t = t0 + j;
        if (t >= kv)
          continue;
        double s;
        if (legacy_reduction) {
          s = spart[4u * j] + spart[4u * j + 1u] + spart[4u * j + 2u] +
              spart[4u * j + 3u];
        } else {
          s = spart[j] + spart[32u + j] + spart[64u + j] + spart[96u + j];
        }
        srow[j] = s * scale;
      }
      double mnew = mrun;
      for (uint32_t j = 0; j < 32u; j++) {
        if (t0 + j >= kv)
          continue;
        mnew = std::max(mnew, srow[j]);
      }
      double corr = (mrun == -1e30) ? 0.0 : std::exp2((mrun - mnew) * log2e);
      double lsum = 0.0;
      for (uint32_t j = 0; j < 32u; j++) {
        if (t0 + j >= kv)
          continue;
        lsum += std::exp2((srow[j] - mnew) * log2e);
      }
      lrun = lrun * corr + lsum;
      mrun = mnew;
      for (uint32_t d = 0; d < hd; d++) {
        acc[d] *= corr;
      }
      for (uint32_t j = 0; j < 32u; j++) {
        uint32_t t = t0 + j;
        if (t >= kv)
          continue;
        double pj = std::exp2((srow[j] - mnew) * log2e);
        for (uint32_t d = 0; d < hd; d++) {
          acc[d] += pj * double(v[(size_t(t) * hkv + hk) * hd + d]);
        }
      }
    }
    for (uint32_t d = 0; d < hd; d++) {
      o[head * hd + d] = float(acc[d] / lrun);
    }
  }
}

} // namespace omarchy_spike::ref
