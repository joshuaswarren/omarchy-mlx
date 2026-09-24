// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/bundle.h"

#include <json.hpp>

#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <vector>

namespace mlx::core::omarchy::ane {
namespace {

std::runtime_error bundle_error(const std::string& reason) {
  return std::runtime_error("[omarchy-ane] bundle: " + reason + ".");
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Descriptor digests ground in the receipts: the export
// receipt hashes model.hwx with `shasum -a 256`, and every validation JSON
// records hwx_sha256 / anec_sha256 digests.
//
// The compress step dispatches to the ARMv8 crypto extension when the CPU
// reports it (getauxval at first use, one binary, portable path otherwise).
// The scalar path here is ~100 MB/s, which made payload re-verification the
// dominant cost of loading the 96 8.4 MB FFN islands each pass (measured
// 2026-09-17: 6.8 s of a 7.1 s resident session open); the hardware path
// removes that share without changing a single digest.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <cstring>

#if defined(__aarch64__) && defined(__linux__)
#include <arm_neon.h>
#include <sys/auxv.h>
#define ANE_SHA256_AARCH64 1
#ifndef HWCAP_SHA256
#define HWCAP_SHA256 (1 << 6)
#endif
#endif

bool sha256_crypto_available() {
#if defined(ANE_SHA256_AARCH64)
  static const bool kCrypto = (getauxval(AT_HWCAP) & HWCAP_SHA256) != 0;
  return kCrypto;
#else
  return false;
#endif
}

#if defined(ANE_SHA256_AARCH64)
// One block through the ARMv8 SHA-256 instructions. The round sequence is
// the canonical public-domain one (Jeffrey Walton's sha256-arm.c, after
// ARM's mbedTLS work): hq updates state0 (a-d), h2 updates state1 (e-h)
// from the saved pre-round state0, and the message schedule stays one group
// ahead of the rounds that consume it. K is renamed to this file's kK.
__attribute__((target("+crypto")))
void sha256_compress_crypto(uint32_t state[8], const uint8_t block[64]) {
  static constexpr uint32_t kK[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  uint32x4_t STATE0, STATE1, ABEF_SAVE, CDGH_SAVE;
  uint32x4_t MSG0, MSG1, MSG2, MSG3;
  uint32x4_t TMP0, TMP1, TMP2;

  STATE0 = vld1q_u32(&state[0]);
  STATE1 = vld1q_u32(&state[4]);

  ABEF_SAVE = STATE0;
  CDGH_SAVE = STATE1;

  MSG0 = vld1q_u32((const uint32_t *)(block +  0));
  MSG1 = vld1q_u32((const uint32_t *)(block + 16));
  MSG2 = vld1q_u32((const uint32_t *)(block + 32));
  MSG3 = vld1q_u32((const uint32_t *)(block + 48));

  /* Reverse for little endian */
  MSG0 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(MSG0)));
  MSG1 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(MSG1)));
  MSG2 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(MSG2)));
  MSG3 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(MSG3)));

  TMP0 = vaddq_u32(MSG0, vld1q_u32(&kK[0x00]));

  /* Rounds 0-3 */
  MSG0 = vsha256su0q_u32(MSG0, MSG1);
  TMP2 = STATE0;
  TMP1 = vaddq_u32(MSG1, vld1q_u32(&kK[0x04]));
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
  MSG0 = vsha256su1q_u32(MSG0, MSG2, MSG3);

  /* Rounds 4-7 */
  MSG1 = vsha256su0q_u32(MSG1, MSG2);
  TMP2 = STATE0;
  TMP0 = vaddq_u32(MSG2, vld1q_u32(&kK[0x08]));
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
  MSG1 = vsha256su1q_u32(MSG1, MSG3, MSG0);

  /* Rounds 8-11 */
  MSG2 = vsha256su0q_u32(MSG2, MSG3);
  TMP2 = STATE0;
  TMP1 = vaddq_u32(MSG3, vld1q_u32(&kK[0x0c]));
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
  MSG2 = vsha256su1q_u32(MSG2, MSG0, MSG1);

  /* Rounds 12-15 */
  MSG3 = vsha256su0q_u32(MSG3, MSG0);
  TMP2 = STATE0;
  TMP0 = vaddq_u32(MSG0, vld1q_u32(&kK[0x10]));
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
  MSG3 = vsha256su1q_u32(MSG3, MSG1, MSG2);

  /* Rounds 16-19 */
  MSG0 = vsha256su0q_u32(MSG0, MSG1);
  TMP2 = STATE0;
  TMP1 = vaddq_u32(MSG1, vld1q_u32(&kK[0x14]));
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
  MSG0 = vsha256su1q_u32(MSG0, MSG2, MSG3);

  /* Rounds 20-23 */
  MSG1 = vsha256su0q_u32(MSG1, MSG2);
  TMP2 = STATE0;
  TMP0 = vaddq_u32(MSG2, vld1q_u32(&kK[0x18]));
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
  MSG1 = vsha256su1q_u32(MSG1, MSG3, MSG0);

  /* Rounds 24-27 */
  MSG2 = vsha256su0q_u32(MSG2, MSG3);
  TMP2 = STATE0;
  TMP1 = vaddq_u32(MSG3, vld1q_u32(&kK[0x1c]));
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
  MSG2 = vsha256su1q_u32(MSG2, MSG0, MSG1);

  /* Rounds 28-31 */
  MSG3 = vsha256su0q_u32(MSG3, MSG0);
  TMP2 = STATE0;
  TMP0 = vaddq_u32(MSG0, vld1q_u32(&kK[0x20]));
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
  MSG3 = vsha256su1q_u32(MSG3, MSG1, MSG2);

  /* Rounds 32-35 */
  MSG0 = vsha256su0q_u32(MSG0, MSG1);
  TMP2 = STATE0;
  TMP1 = vaddq_u32(MSG1, vld1q_u32(&kK[0x24]));
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
  MSG0 = vsha256su1q_u32(MSG0, MSG2, MSG3);

  /* Rounds 36-39 */
  MSG1 = vsha256su0q_u32(MSG1, MSG2);
  TMP2 = STATE0;
  TMP0 = vaddq_u32(MSG2, vld1q_u32(&kK[0x28]));
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
  MSG1 = vsha256su1q_u32(MSG1, MSG3, MSG0);

  /* Rounds 40-43 */
  MSG2 = vsha256su0q_u32(MSG2, MSG3);
  TMP2 = STATE0;
  TMP1 = vaddq_u32(MSG3, vld1q_u32(&kK[0x2c]));
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
  MSG2 = vsha256su1q_u32(MSG2, MSG0, MSG1);

  /* Rounds 44-47 */
  MSG3 = vsha256su0q_u32(MSG3, MSG0);
  TMP2 = STATE0;
  TMP0 = vaddq_u32(MSG0, vld1q_u32(&kK[0x30]));
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
  MSG3 = vsha256su1q_u32(MSG3, MSG1, MSG2);

  /* Rounds 48-51 */
  TMP2 = STATE0;
  TMP1 = vaddq_u32(MSG1, vld1q_u32(&kK[0x34]));
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);

  /* Rounds 52-55 */
  TMP2 = STATE0;
  TMP0 = vaddq_u32(MSG2, vld1q_u32(&kK[0x38]));
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);

  /* Rounds 56-59 */
  TMP2 = STATE0;
  TMP1 = vaddq_u32(MSG3, vld1q_u32(&kK[0x3c]));
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);

  /* Rounds 60-63 */
  TMP2 = STATE0;
  STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
  STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);

  /* Combine state */
  STATE0 = vaddq_u32(STATE0, ABEF_SAVE);
  STATE1 = vaddq_u32(STATE1, CDGH_SAVE);

  vst1q_u32(&state[0], STATE0);
  vst1q_u32(&state[4], STATE1);
}
#endif

struct Sha256Context {
  uint32_t state[8];
  uint8_t buffer[64];
  size_t buffered;
  uint64_t total;
};

uint32_t rotate_right(uint32_t value, uint32_t bits) {
  return (value >> bits) | (value << (32 - bits));
}

void sha256_compress_scalar(uint32_t state[8], const uint8_t block[64]) {
  static constexpr uint32_t kK[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (uint32_t(block[i * 4]) << 24) | (uint32_t(block[i * 4 + 1]) << 16) |
           (uint32_t(block[i * 4 + 2]) << 8) | uint32_t(block[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    uint32_t s0 = rotate_right(w[i - 15], 7) ^ rotate_right(w[i - 15], 18) ^
                  (w[i - 15] >> 3);
    uint32_t s1 = rotate_right(w[i - 2], 17) ^ rotate_right(w[i - 2], 19) ^
                  (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
  for (int i = 0; i < 64; ++i) {
    uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
    uint32_t ch = (e & f) ^ (~e & g);
    uint32_t t1 = h + s1 + ch + kK[i] + w[i];
    uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t t2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
  if (sha256_crypto_available()) {
#if defined(ANE_SHA256_AARCH64)
    sha256_compress_crypto(state, block);
    return;
#endif
  }
  sha256_compress_scalar(state, block);
}

// Appends bytes and compresses full blocks. Does not touch the message
// length: padding bytes go through the same path without being counted.
void sha256_feed(Sha256Context& context, const uint8_t* data, size_t size) {
  while (size > 0) {
    size_t take = 64 - context.buffered;
    if (take > size) {
      take = size;
    }
    for (size_t i = 0; i < take; ++i) {
      context.buffer[context.buffered + i] = data[i];
    }
    context.buffered += take;
    data += take;
    size -= take;
    if (context.buffered == 64) {
      sha256_compress(context.state, context.buffer);
      context.buffered = 0;
    }
  }
}

std::string sha256_digest(const Sha256Context& context) {
  static const char* hex = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (uint32_t word : context.state) {
    for (int shift = 28; shift >= 0; shift -= 4) {
      out.push_back(hex[(word >> shift) & 0xf]);
    }
  }
  return out;
}

std::string sha256_pad_and_digest(Sha256Context& context) {
  uint64_t bits = context.total * 8;
  const uint8_t pad = 0x80;
  const uint8_t zero = 0;
  sha256_feed(context, &pad, 1);
  while (context.buffered != 56) {
    sha256_feed(context, &zero, 1);
  }
  uint8_t length[8];
  for (int i = 0; i < 8; ++i) {
    length[i] = uint8_t(bits >> (56 - 8 * i));
  }
  sha256_feed(context, length, 8);
  return sha256_digest(context);
}

Sha256Context sha256_begin() {
  Sha256Context context{};
  context.state[0] = 0x6a09e667u;
  context.state[1] = 0xbb67ae85u;
  context.state[2] = 0x3c6ef372u;
  context.state[3] = 0xa54ff53au;
  context.state[4] = 0x510e527fu;
  context.state[5] = 0x9b05688cu;
  context.state[6] = 0x1f83d9abu;
  context.state[7] = 0x5be0cd19u;
  context.buffered = 0;
  context.total = 0;
  return context;
}

} // namespace

std::string sha256_hex(const uint8_t* data, size_t size) {
  Sha256Context context = sha256_begin();
  context.total += size;
  sha256_feed(context, data, size);
  return sha256_pad_and_digest(context);
}

std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw bundle_error("cannot read payload " + path.string());
  }
  Sha256Context context = sha256_begin();
  std::vector<char> chunk(64 * 1024);
  while (input) {
    input.read(chunk.data(), chunk.size());
    size_t got = size_t(input.gcount());
    if (got == 0) {
      break;
    }
    context.total += got;
    sha256_feed(context, reinterpret_cast<const uint8_t*>(chunk.data()), got);
  }
  if (input.bad()) {
    throw bundle_error("read failed on payload " + path.string());
  }
  return sha256_pad_and_digest(context);
}

struct DigestCacheKey {
  std::string path;
  uint64_t dev;
  uint64_t ino;
  uint64_t size;
  uint64_t mtime_ns;

  std::string serialize() const {
    char hex[64];
    std::snprintf(
        hex,
        sizeof(hex),
        "|%llx|%llx|%llx|%llx",
        static_cast<unsigned long long>(dev),
        static_cast<unsigned long long>(ino),
        static_cast<unsigned long long>(size),
        static_cast<unsigned long long>(mtime_ns));
    return path + hex;
  }

  bool operator<(const DigestCacheKey& rhs) const {
    return serialize() < rhs.serialize();
  }
};

namespace {

std::string payload_collection_sha256(const std::vector<AnePayload>& payloads) {
  std::vector<const AnePayload*> ordered;
  ordered.reserve(payloads.size());
  for (const auto& payload : payloads) {
    ordered.push_back(&payload);
  }
  std::sort(ordered.begin(), ordered.end(), [](const auto* lhs, const auto* rhs) {
    return lhs->path < rhs->path;
  });

  nlohmann::json records = nlohmann::json::array();
  for (const auto* payload : ordered) {
    records.push_back({
        {"role", payload->role},
        {"path", payload->path},
        {"byte_size", payload->byte_size},
        {"sha256", payload->sha256},
    });
  }
  const std::string encoded = records.dump(-1, ' ', true);
  return sha256_hex(
      reinterpret_cast<const uint8_t*>(encoded.data()), encoded.size());
}

uint64_t file_size_checked(const std::filesystem::path& path) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) {
    throw bundle_error("cannot stat file " + path.string());
  }
  return static_cast<uint64_t>(size);
}

uint64_t checked_add(uint64_t lhs, uint64_t rhs, const std::string& label) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
    throw bundle_error(label + " overflows uint64");
  }
  return lhs + rhs;
}

uint64_t checked_mul(uint64_t lhs, uint64_t rhs, const std::string& label) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    throw bundle_error(label + " overflows uint64");
  }
  return lhs * rhs;
}

uint64_t align_up(uint64_t value, uint64_t alignment, const std::string& label) {
  const uint64_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  return checked_add(value, alignment - remainder, label);
}

template <typename T>
T read_le(const std::array<unsigned char, kAnecHeaderSize>& bytes, size_t offset) {
  static_assert(std::is_unsigned_v<T>);
  if (offset > bytes.size() - sizeof(T)) {
    throw bundle_error("ANEC header read escapes header");
  }
  T value = 0;
  for (size_t i = 0; i < sizeof(T); ++i) {
    value |= static_cast<T>(bytes[offset + i]) << (i * 8);
  }
  return value;
}

uint64_t shape_elements(const std::vector<uint64_t>& shape, const std::string& label) {
  uint64_t total = 1;
  for (auto dim : shape) {
    total = checked_mul(total, dim, label);
  }
  return total;
}

uint64_t channel_size_bytes(
    const AneAnecHeader& header,
    uint32_t bdx,
    uint64_t tile_alignment) {
  return checked_mul(header.tiles.at(bdx), tile_alignment, "ANEC channel size");
}

constexpr uint32_t kBindFirstSurface = 4;
constexpr uint32_t kBindDmaDisabled = 0x00008880u;
constexpr uint32_t kBindDstRegister = 0x17800u;
constexpr uint32_t kBindSelectorMask = 0x1fu;
constexpr uint64_t kBindMinTaskBytes = 40;
constexpr uint32_t kBindSelectors[3][2] = {
    {0x13800u, 0}, {0x13804u, 6}, {kBindDstRegister, 12}};

uint32_t bind_word(const uint8_t* task, uint64_t index) {
  return uint32_t(task[index * 4]) | (uint32_t(task[index * 4 + 1]) << 8) |
      (uint32_t(task[index * 4 + 2]) << 16) |
      (uint32_t(task[index * 4 + 3]) << 24);
}

int bind_task_dma(const uint8_t* task, uint64_t bytes, uint32_t dma[3]) {
  const uint64_t words = bytes / 4;
  for (int slot = 0; slot < 3; ++slot) {
    dma[slot] = kBindDmaDisabled;
  }
  if (bytes < kBindMinTaskBytes || bytes % 4 != 0) {
    return -1;
  }
  uint64_t index = 10 + (((bind_word(task, 9) & 3) == 3) ? 1 : 0);
  while (index < words) {
    const uint32_t header = bind_word(task, index);
    const uint64_t count = (header >> 26) + 1;
    const uint32_t base = header & 0x03ffffffu;
    if (index + count >= words) {
      return -1;
    }
    for (uint64_t offset = 0; offset < count; ++offset) {
      for (int slot = 0; slot < 3; ++slot) {
        if (base + offset * 4 == kBindSelectors[slot][0]) {
          dma[slot] = bind_word(task, index + 1 + offset);
        }
      }
    }
    index += 1 + count;
  }
  return 0;
}

int bind_walk(
    const AneAnecHeader& header,
    const uint8_t* stream,
    uint64_t stream_size,
    uint8_t* is_src,
    uint8_t* is_dst) {
  uint64_t offset = 0;
  uint64_t bytes = header.task_descriptor_size;
  if (header.task_descriptor_count == 0) {
    return -1;
  }
  for (uint32_t index = 0; index < header.task_descriptor_count; ++index) {
    if (bytes < kBindMinTaskBytes || offset > stream_size ||
        bytes > stream_size - offset) {
      return -1;
    }
    uint32_t dma[3];
    if (bind_task_dma(stream + offset, bytes, dma) < 0) {
      return -1;
    }
    const uint32_t selectors = bind_word(stream + offset, 8);
    for (int slot = 0; slot < 3; ++slot) {
      const uint32_t channel =
          (selectors >> kBindSelectors[slot][1]) & kBindSelectorMask;
      if (dma[slot] == kBindDmaDisabled) {
        continue;
      }
      if (channel < kBindFirstSurface || channel >= kAnecTileCount ||
          header.tiles[channel] == 0) {
        continue;
      }
      if (kBindSelectors[slot][0] == kBindDstRegister) {
        is_dst[channel] = 1;
      } else {
        is_src[channel] = 1;
      }
    }
    if (index + 1 == header.task_descriptor_count) {
      break;
    }
    const uint64_t next = bind_word(stream + offset, 7);
    bytes = (((bind_word(stream + offset, 1) >> 16) & 0x1ff) + 1) * 4;
    if (next % 4 != 0 || next > stream_size) {
      return -1;
    }
    offset = next;
  }
  return 0;
}

bool derive_role_channels(
    const AneAnecHeader& header,
    const uint8_t* stream,
    uint64_t stream_size,
    std::vector<uint32_t>& src,
    std::vector<uint32_t>& dst) {
  src.clear();
  dst.clear();
  for (uint32_t i = 0; i < header.destination_count; ++i) {
    dst.push_back(kBindFirstSurface + i);
  }
  for (uint32_t i = 0; i < header.source_count; ++i) {
    src.push_back(kBindFirstSurface + header.destination_count + i);
  }
  if (header.source_count > kAnecTileCount ||
      header.destination_count > kAnecTileCount) {
    return false;
  }
  uint8_t is_src[kAnecTileCount] = {};
  uint8_t is_dst[kAnecTileCount] = {};
  if (bind_walk(header, stream, stream_size, is_src, is_dst) < 0) {
    return false;
  }
  std::vector<uint32_t> derived_src;
  std::vector<uint32_t> derived_dst;
  for (uint32_t channel = kBindFirstSurface; channel < kAnecTileCount;
       ++channel) {
    if (is_dst[channel]) {
      if (derived_dst.size() == kAnecTileCount) {
        return false;
      }
      derived_dst.push_back(channel);
    } else if (is_src[channel]) {
      if (derived_src.size() == kAnecTileCount) {
        return false;
      }
      derived_src.push_back(channel);
    }
  }
  // Apple's streams leave some surfaces unnamed by the selector
  // registers (island-pv never enables the second source selector, yet
  // binds probs on the next allocated channel). Such surfaces bind
  // positionally: first unused allocated channel ascending,
  // destinations first, then sources. A manifest that declares a
  // channel outside this map still fails the declared-vs-derived
  // comparison in validate_binding.
  const auto fill = [&](std::vector<uint32_t>& derived, uint32_t needed) {
    for (uint32_t channel = kBindFirstSurface;
         channel < kAnecTileCount && derived.size() < needed; ++channel) {
      if (is_dst[channel] || is_src[channel] ||
          header.tiles[channel] == 0) {
        continue;
      }
      derived.push_back(channel);
    }
  };
  fill(derived_dst, header.destination_count);
  fill(derived_src, header.source_count);
  if (derived_src.size() != header.source_count ||
      derived_dst.size() != header.destination_count) {
    return false;
  }
  src = std::move(derived_src);
  dst = std::move(derived_dst);
  return true;
}

std::vector<uint8_t> read_anec_payload(
    const std::filesystem::path& path,
    uint64_t payload_size) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw bundle_error("cannot read payload " + path.string());
  }
  in.seekg(static_cast<std::streamoff>(kAnecPayloadOffset));
  std::vector<uint8_t> payload(payload_size);
  in.read(reinterpret_cast<char*>(payload.data()),
          static_cast<std::streamsize>(payload_size));
  if (in.gcount() != static_cast<std::streamsize>(payload_size)) {
    throw bundle_error("cannot read ANEC payload " + path.string());
  }
  return payload;
}

// Derived role-channel cache: the (src, dst) surface map that
// derive_role_channels can only produce by walking the task stream inside
// the full ANEC payload, keyed on the same cheap file identity as the
// digest cache (path|dev|ino|size|mtime_ns) and persisted in a sidecar the
// same way. Without it every open re-reads the whole payload per program
// just to re-derive channels the file cannot have changed without its
// identity changing. The per-binding validation that consumes the channels
// still runs on every open, so a wrong cached map fails the open loudly
// instead of misbinding. MLX_OMARCHY_ANE_CHANNEL_CACHE forces re-derivation
// every open when set to 0/false/no/off/empty (case-insensitive);
// MLX_OMARCHY_ANE_CHANNEL_CACHE_PATH overrides the sidecar location.
struct ChannelCacheEntry {
  std::vector<uint32_t> src;
  std::vector<uint32_t> dst;
};

bool channel_cache_enabled() {
  const char* value = std::getenv("MLX_OMARCHY_ANE_CHANNEL_CACHE");
  if (value == nullptr) {
    return true;
  }
  std::string lowered(value);
  for (char& c : lowered) {
    c = char(std::tolower(static_cast<unsigned char>(c)));
  }
  return lowered != "0" && lowered != "false" && lowered != "no" &&
      lowered != "off" && !lowered.empty();
}

std::filesystem::path channel_cache_path() {
  if (const char* override_path =
          std::getenv("MLX_OMARCHY_ANE_CHANNEL_CACHE_PATH")) {
    return std::filesystem::path(override_path);
  }
  std::filesystem::path base;
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
    base = xdg;
  } else if (const char* home = std::getenv("HOME"); home && *home) {
    base = std::filesystem::path(home) / ".cache";
  } else {
    return {};
  }
  return base / "mlx-omarchy" / "ane-channel-cache.txt";
}

std::mutex g_channel_cache_mutex;
std::map<DigestCacheKey, ChannelCacheEntry> g_channel_cache;

// Sidecar line: <serialized identity key> <nsrc>:c,c,... <ndst>:c,c,...
// The identity key may contain spaces, so fields are split from the end and
// the channel fields are validated structurally before use.
std::optional<std::pair<uint64_t, std::vector<uint32_t>>> parse_channel_field(
    const std::string& field) {
  const size_t colon = field.find(':');
  if (colon == std::string::npos || colon == 0) {
    return std::nullopt;
  }
  uint64_t count = 0;
  try {
    count = std::stoull(field.substr(0, colon));
  } catch (const std::exception&) {
    return std::nullopt;
  }
  std::vector<uint32_t> channels;
  size_t begin = colon + 1;
  while (true) {
    const size_t comma = field.find(',', begin);
    const std::string token = field.substr(
        begin, (comma == std::string::npos ? field.size() : comma) - begin);
    if (token.empty()) {
      return std::nullopt;
    }
    uint64_t channel = 0;
    try {
      channel = std::stoull(token);
    } catch (const std::exception&) {
      return std::nullopt;
    }
    if (channel >= kAnecTileCount) {
      return std::nullopt;
    }
    channels.push_back(uint32_t(channel));
    if (comma == std::string::npos) {
      break;
    }
    begin = comma + 1;
  }
  if (channels.size() != count) {
    return std::nullopt;
  }
  return std::make_pair(count, std::move(channels));
}

std::map<DigestCacheKey, ChannelCacheEntry> load_channel_cache_disk() {
  std::map<DigestCacheKey, ChannelCacheEntry> entries;
  std::ifstream input(channel_cache_path());
  if (!input) {
    return entries;
  }
  std::string line;
  while (std::getline(input, line)) {
    const size_t dst_sep = line.rfind(' ');
    if (dst_sep == std::string::npos || dst_sep == 0) {
      continue;
    }
    const size_t src_sep = line.rfind(' ', dst_sep - 1);
    if (src_sep == std::string::npos || src_sep == 0) {
      continue;
    }
    const std::string key = line.substr(0, src_sep);
    std::vector<size_t> pipes;
    for (size_t i = key.size(); i-- > 0;) {
      if (key[i] == '|') {
        pipes.push_back(i);
        if (pipes.size() == 4) {
          break;
        }
      }
    }
    if (pipes.size() != 4) {
      continue;
    }
    auto src = parse_channel_field(line.substr(src_sep + 1, dst_sep - src_sep - 1));
    auto dst = parse_channel_field(line.substr(dst_sep + 1));
    if (!src || !dst) {
      continue;
    }
    DigestCacheKey parsed;
    parsed.path = key.substr(0, pipes[3]);
    const auto field = [&](size_t n) -> uint64_t {
      return std::strtoull(key.c_str() + pipes[n] + 1, nullptr, 16);
    };
    parsed.dev = field(3);
    parsed.ino = field(2);
    parsed.size = field(1);
    parsed.mtime_ns = field(0);
    ChannelCacheEntry entry;
    entry.src = std::move(src->second);
    entry.dst = std::move(dst->second);
    entries.emplace(std::move(parsed), std::move(entry));
  }
  return entries;
}

void store_channel_cache_disk(
    const std::filesystem::path& sidecar,
    const std::string& serialized_key,
    const ChannelCacheEntry& entry) {
  std::error_code ec;
  std::filesystem::create_directories(sidecar.parent_path(), ec);
  if (ec) {
    return;
  }
  std::ofstream output(sidecar, std::ios::binary | std::ios::app);
  if (!output) {
    return;
  }
  const auto join = [](const std::vector<uint32_t>& channels) {
    std::string joined;
    for (size_t i = 0; i < channels.size(); ++i) {
      if (i != 0) {
        joined += ',';
      }
      joined += std::to_string(channels[i]);
    }
    return joined;
  };
  output << serialized_key << ' ' << entry.src.size() << ':' << join(entry.src)
         << ' ' << entry.dst.size() << ':' << join(entry.dst) << '\n';
}

bool channel_entry_plausible(
    const ChannelCacheEntry& entry,
    const AneAnecHeader& header) {
  if (entry.src.size() != header.source_count ||
      entry.dst.size() != header.destination_count) {
    return false;
  }
  const auto plausible = [&header](uint32_t channel) {
    return channel >= kBindFirstSurface && channel < kAnecTileCount &&
        header.tiles[channel] != 0;
  };
  return std::all_of(entry.src.begin(), entry.src.end(), plausible) &&
      std::all_of(entry.dst.begin(), entry.dst.end(), plausible);
}

// Returns the derived role channels, deriving from a full payload read only
// on a cache miss (kill-switch, unknown identity, or missing/implausible
// sidecar entry). Returns false when the task stream does not name every
// surface, exactly like derive_role_channels, and that failure is never
// cached.
bool derived_role_channels(
    const AneAnecHeader& header,
    const std::filesystem::path& anec_path,
    ChannelCacheEntry& entry) {
  struct ::stat st {};
  if (::stat(anec_path.c_str(), &st) != 0) {
    throw bundle_error("cannot stat file " + anec_path.string());
  }
#if defined(__APPLE__)
  const uint64_t mtime_ns = uint64_t(st.st_mtimespec.tv_sec) * 1000000000ull +
      uint64_t(st.st_mtimespec.tv_nsec);
#else
  const uint64_t mtime_ns = uint64_t(st.st_mtim.tv_sec) * 1000000000ull +
      uint64_t(st.st_mtim.tv_nsec);
#endif
  DigestCacheKey key{
      anec_path.string(),
      uint64_t(st.st_dev),
      uint64_t(st.st_ino),
      uint64_t(st.st_size),
      mtime_ns};
  if (channel_cache_enabled()) {
    std::lock_guard<std::mutex> lock(g_channel_cache_mutex);
    auto it = g_channel_cache.find(key);
    if (it != g_channel_cache.end() &&
        channel_entry_plausible(it->second, header)) {
      entry = it->second;
      return true;
    }
    auto disk = load_channel_cache_disk();
    auto dit = disk.find(key);
    if (dit != disk.end() && channel_entry_plausible(dit->second, header)) {
      g_channel_cache.emplace(key, dit->second);
      entry = dit->second;
      return true;
    }
  }
  const auto payload = read_anec_payload(anec_path, header.payload_size);
  if (!derive_role_channels(
          header,
          payload.data(),
          payload.size(),
          entry.src,
          entry.dst)) {
    return false;
  }
  if (channel_cache_enabled()) {
    const std::string serialized = key.serialize();
    std::lock_guard<std::mutex> lock(g_channel_cache_mutex);
    g_channel_cache.emplace(std::move(key), entry);
    if (auto sidecar = channel_cache_path(); !sidecar.empty()) {
      store_channel_cache_disk(sidecar, serialized, entry);
    }
  }
  return true;
}

void validate_binding(
    const std::string& label,
    const AneProgramBinding& binding,
    const AneAnecHeader& header,
    uint32_t expected_channel,
    uint64_t tile_alignment) {
  if (binding.channel != expected_channel || binding.channel >= kAnecTileCount) {
    throw bundle_error(label + " channel does not match ANEC binding order");
  }
  const uint64_t allocation =
      channel_size_bytes(header, expected_channel, tile_alignment);
  if (allocation != binding.allocation_bytes) {
    throw bundle_error(label + " allocation_bytes does not match ANEC channel allocation");
  }
  if (binding.dtype != "float16" && binding.dtype != "bfloat16" &&
      binding.dtype != "bool") {
    throw bundle_error(
        label + " ANEC channel requires a 16-bit or 1-byte bool tensor dtype");
  }
  // Raw bindings are selector-addressed: the ANEC header's NCHW for the
  // channel describes the engine's staging window, not a dense tile
  // placement, so only the allocation is checked (parse already enforced
  // logical <= allocation and alignment).
  if (binding.raw) {
    return;
  }
  if (header.nchw[expected_channel] != binding.nchw) {
    throw bundle_error(label + " NCHW does not match ANEC channel geometry");
  }
}

void validate_program_contract(
    const AneProgram& program,
    const AneAnecHeader& header,
    const std::filesystem::path& anec_path,
    size_t program_index,
    uint64_t tile_alignment) {
  const std::string prefix = "program " + std::to_string(program_index);
  if (program.task_descriptors != header.task_descriptor_count) {
    throw bundle_error(prefix + " task_descriptors does not match ANEC header");
  }
  if (program.inputs.size() != header.source_count) {
    throw bundle_error(prefix + " input count does not match ANEC source_count");
  }
  if (program.outputs.size() != header.destination_count) {
    throw bundle_error(prefix + " output count does not match ANEC destination_count");
  }
  if (channel_size_bytes(header, 3, tile_alignment) != program.scratch_bytes) {
    throw bundle_error(prefix + " scratch_bytes does not match ANEC channel 3 allocation");
  }
  ChannelCacheEntry channels;
  if (!derived_role_channels(header, anec_path, channels)) {
    throw bundle_error(
        prefix +
        " task stream does not name every surface; channel map is positional");
  }
  const std::vector<uint32_t>& src = channels.src;
  const std::vector<uint32_t>& dst = channels.dst;
  for (uint32_t i = 0; i < program.outputs.size(); ++i) {
    validate_binding(
        prefix + " output " + program.outputs[i].tensor,
        program.outputs[i],
        header,
        dst[i],
        tile_alignment);
  }
  for (uint32_t i = 0; i < program.inputs.size(); ++i) {
    validate_binding(
        prefix + " input " + program.inputs[i].tensor,
        program.inputs[i],
        header,
        src[i],
        tile_alignment);
  }
}

} // namespace

AneAnecHeader parse_anec_header(
    const std::filesystem::path& path,
    uint64_t tile_shift) {
  const uint64_t tile_alignment = uint64_t{1} << tile_shift;
  const auto file_size = file_size_checked(path);
  if (file_size < kAnecHeaderSize) {
    throw bundle_error("ANEC file is smaller than libane header");
  }

  std::array<unsigned char, kAnecHeaderSize> bytes{};
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw bundle_error("cannot read payload " + path.string());
  }
  in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (in.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw bundle_error("cannot read ANEC header " + path.string());
  }

  AneAnecHeader header;
  header.payload_size = read_le<uint64_t>(bytes, 0);
  header.task_descriptor_size = read_le<uint32_t>(bytes, 8);
  header.task_descriptor_count = read_le<uint32_t>(bytes, 12);
  header.task_size = read_le<uint64_t>(bytes, 16);
  header.kernel_size = read_le<uint64_t>(bytes, 24);
  header.source_count = read_le<uint32_t>(bytes, 32);
  header.destination_count = read_le<uint32_t>(bytes, 36);

  size_t offset = 40;
  for (auto& tile : header.tiles) {
    tile = read_le<uint32_t>(bytes, offset);
    offset += sizeof(uint32_t);
  }
  for (auto& dims : header.nchw) {
    for (auto& dim : dims) {
      dim = read_le<uint64_t>(bytes, offset);
      offset += sizeof(uint64_t);
    }
  }

  if (header.payload_size == 0) {
    throw bundle_error("ANEC payload size is zero");
  }
  if (header.task_descriptor_size == 0 || header.task_descriptor_count == 0) {
    throw bundle_error("ANEC task descriptor table is empty");
  }
  if (header.task_descriptor_size % sizeof(uint32_t) != 0) {
    throw bundle_error("ANEC task descriptor size is not a multiple of 4");
  }
  if (header.task_descriptor_count > 0xffff) {
    throw bundle_error("ANEC task descriptor count exceeds driver limit 0xffff");
  }
  if (header.task_descriptor_size > 0x40000) {
    throw bundle_error("ANEC task descriptor size exceeds driver limit 0x40000");
  }
  if (header.task_size == 0) {
    throw bundle_error("ANEC task size is zero");
  }
  if (uint64_t{4} + header.destination_count + header.source_count >
      kAnecTileCount) {
    throw bundle_error("ANEC source/destination count exceeds libane channel table");
  }
  if (checked_add(kAnecPayloadOffset, header.payload_size, "ANEC payload end") !=
      file_size) {
    throw bundle_error("ANEC file size does not match payload_size");
  }
  if (header.tiles[0] == 0) {
    throw bundle_error("ANEC command channel allocation is zero");
  }
  if (header.tiles[1] != 0) {
    throw bundle_error("ANEC reserved kernel channel allocation must be zero");
  }

  const uint64_t command_channel_size =
      channel_size_bytes(header, 0, tile_alignment);
  if (header.payload_size > command_channel_size) {
    throw bundle_error("ANEC executable payload exceeds command channel allocation");
  }
  if (header.task_descriptor_size > header.payload_size) {
    throw bundle_error("ANEC task descriptor bytes exceed executable payload");
  }
  if (header.task_size >= command_channel_size) {
    throw bundle_error("ANEC task size reaches command channel end");
  }

  const uint64_t kernel_offset = align_up(header.task_size, 16, "ANEC kernel offset");
  if (kernel_offset > header.payload_size ||
      header.kernel_size > header.payload_size - kernel_offset) {
    throw bundle_error("ANEC task plus kernel bytes exceed executable payload");
  }
  header.bootstrap_channel_size = align_up(
      header.task_descriptor_size, tile_alignment, "ANEC bootstrap channel");
  return header;
}

// Digest cache: verified payload digests keyed on cheap file identity
// (resolved path + device + inode + size + mtime_ns), persisted across
// processes in a sidecar file so a fresh worker process per utterance does
// not re-read + re-hash an unchanged payload. A hit skips the payload read
// and hash; the digest is still compared against the manifest expectation
// on every open, and any identity change forces a full re-verify.
//
// Security/consistency story (stated for the receipt): the manifest declaring
// the expected digest sits in the same directory as the payload, so an
// attacker able to swap the payload can also swap the manifest — the digest
// check was never an anti-tamper boundary, only a mismatch detector for
// accidental corruption or stale deploy. The sidecar lives under the user's
// cache dir, which requires the same write privilege as the bundle itself,
// so the cache adds no attack capability a same-user adversary lacked; it
// does keep the mismatch detector for accidental corruption (any content
// change that alters dev/inode/size/mtime_ns re-verifies). The deliberate
// trade: a content change that restores the identical identity tuple AND a
// matching forged sidecar entry is served from cache without re-hash —
// reachable only by the same-user adversary who could have rewritten the
// manifest instead. MLX_OMARCHY_ANE_DIGEST_CACHE forces full verification
// every open (no sidecar reads or writes) when set to 0/false/no/off/empty
// (case-insensitive); unset or any other value keeps the cache enabled.
// MLX_OMARCHY_ANE_DIGEST_CACHE_PATH overrides the sidecar location.
bool digest_cache_enabled() {
  const char* value = std::getenv("MLX_OMARCHY_ANE_DIGEST_CACHE");
  if (value == nullptr) {
    return true;
  }
  std::string lowered(value);
  for (char& c : lowered) {
    c = char(std::tolower(static_cast<unsigned char>(c)));
  }
  return lowered != "0" && lowered != "false" && lowered != "no" &&
      lowered != "off" && !lowered.empty();
}

std::filesystem::path digest_cache_path() {
  if (const char* override_path = std::getenv("MLX_OMARCHY_ANE_DIGEST_CACHE_PATH")) {
    return std::filesystem::path(override_path);
  }
  std::filesystem::path base;
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
    base = xdg;
  } else if (const char* home = std::getenv("HOME"); home && *home) {
    base = std::filesystem::path(home) / ".cache";
  } else {
    return {};
  }
  return base / "mlx-omarchy" / "ane-digest-cache.txt";
}

std::mutex g_digest_cache_mutex;
std::map<DigestCacheKey, std::string> g_digest_cache;

std::map<DigestCacheKey, std::string> load_digest_cache_disk() {
  std::map<DigestCacheKey, std::string> entries;
  std::ifstream input(digest_cache_path());
  if (!input) {
    return entries;
  }
  std::string line;
  while (std::getline(input, line)) {
    const size_t sep = line.rfind(' ');
    if (sep == std::string::npos || sep == 0 ||
        sep + 1 + 64 != line.size()) {
      continue; // skip malformed or truncated lines
    }
    std::string key = line.substr(0, sep);
    std::string digest = line.substr(sep + 1);
    // Key layout: <path>|<dev>|<ino>|<size>|<mtime_ns> — split at the LAST
    // four pipes; the path itself may contain anything but the digest tail
    // and pipe count are validated structurally.
    std::vector<size_t> pipes;
    for (size_t i = key.size(); i-- > 0;) {
      if (key[i] == '|') {
        pipes.push_back(i);
        if (pipes.size() == 4) {
          break;
        }
      }
    }
    if (pipes.size() != 4) {
      continue;
    }
    DigestCacheKey parsed;
    parsed.path = key.substr(0, pipes[3]);
    const auto field = [&](size_t n) -> uint64_t {
      return std::strtoull(key.c_str() + pipes[n] + 1, nullptr, 16);
    };
    parsed.dev = field(3);
    parsed.ino = field(2);
    parsed.size = field(1);
    parsed.mtime_ns = field(0);
    entries.emplace(std::move(parsed), std::move(digest));
  }
  return entries;
}

void store_digest_cache_disk(
    const std::filesystem::path& sidecar,
    const std::string& serialized_key,
    const std::string& digest) {
  std::error_code ec;
  std::filesystem::create_directories(sidecar.parent_path(), ec);
  if (ec) {
    return;
  }
  std::ofstream output(sidecar, std::ios::binary | std::ios::app);
  if (!output) {
    return;
  }
  output << serialized_key << ' ' << digest << '\n';
}

// Returns the payload digest, re-hashing only when the kill-switch forces it,
// the file identity has no cached digest, or the sidecar has no entry. The
// returned digest is always compared against the manifest expectation by the
// caller; the cache only removes the read+hash, never the comparison.
std::string sha256_file_cached(const std::filesystem::path& path) {
  struct ::stat st {};
  if (::stat(path.c_str(), &st) != 0) {
    throw bundle_error("cannot stat file " + path.string());
  }
#if defined(__APPLE__)
  const uint64_t mtime_ns = uint64_t(st.st_mtimespec.tv_sec) * 1000000000ull +
      uint64_t(st.st_mtimespec.tv_nsec);
#else
  const uint64_t mtime_ns = uint64_t(st.st_mtim.tv_sec) * 1000000000ull +
      uint64_t(st.st_mtim.tv_nsec);
#endif
  DigestCacheKey key{
      path.string(),
      uint64_t(st.st_dev),
      uint64_t(st.st_ino),
      uint64_t(st.st_size),
      mtime_ns};
  if (digest_cache_enabled()) {
    std::lock_guard<std::mutex> lock(g_digest_cache_mutex);
    auto it = g_digest_cache.find(key);
    if (it != g_digest_cache.end()) {
      return it->second;
    }
    // Not in this process: consult the sidecar (small file, only on a
    // process's first use of a bundle). Stale entries are harmless — the
    // digest is verified against the manifest below.
    auto disk = load_digest_cache_disk();
    auto dit = disk.find(key);
    if (dit != disk.end()) {
      g_digest_cache.emplace(key, dit->second);
      return dit->second;
    }
  }
  std::string digest = sha256_file(path);
  if (digest_cache_enabled()) {
    const std::string serialized = key.serialize();
    std::lock_guard<std::mutex> lock(g_digest_cache_mutex);
    g_digest_cache.emplace(std::move(key), digest);
    if (auto sidecar = digest_cache_path(); !sidecar.empty()) {
      store_digest_cache_disk(sidecar, serialized, digest);
    }
  }
  return digest;
}

AneBundle load_bundle_snapshot(
    const std::filesystem::path& manifest_path,
    const std::map<std::string, std::filesystem::path>& payload_paths) {
  AneManifest manifest = parse_ane_manifest(manifest_path);
  const std::string payload_identity = payload_collection_sha256(manifest.payloads);
  if (manifest.release_asset.model_sha256 != payload_identity) {
    throw bundle_error(
        "release_asset.model_sha256 does not match compiled payload collection");
  }


  std::vector<std::filesystem::path> resolved;
  resolved.reserve(manifest.payloads.size());
  for (const auto& payload : manifest.payloads) {
    auto path = payload_paths.find(payload.path);
    if (path == payload_paths.end()) {
      throw bundle_error("payload file missing: " + payload.path);
    }
    resolved.push_back(path->second);
  }
  for (const auto& [name, path] : payload_paths) {
    if (std::none_of(
            manifest.payloads.begin(),
            manifest.payloads.end(),
            [&](const AnePayload& payload) { return payload.path == name; })) {
      throw bundle_error("unknown snapshot payload '" + name + "'");
    }
  }
  for (size_t i = 0; i < manifest.payloads.size(); ++i) {
    const auto& payload = manifest.payloads[i];
    std::error_code size_error;
    uint64_t actual = uint64_t(std::filesystem::file_size(resolved[i], size_error));
    if (size_error) {
      throw bundle_error("cannot stat payload " + payload.path);
    }
    if (actual != payload.byte_size) {
      throw bundle_error(
          "payload " + payload.path + " byte size " + std::to_string(actual) +
          " does not match manifest " + std::to_string(payload.byte_size));
    }
    std::string digest = sha256_file_cached(resolved[i]);
    if (digest != payload.sha256) {
      throw bundle_error(
          "payload " + payload.path + " sha256 mismatch: manifest " +
          payload.sha256 + " actual " + digest);
    }
  }

  AneBundle bundle;
  bundle.manifest = std::move(manifest);
  for (size_t i = 0; i < bundle.manifest.payloads.size(); ++i) {
    if (bundle.manifest.payloads[i].role == "weights") {
      bundle.weights = resolved[i];
    }
  }
  for (uint64_t p : bundle.manifest.dispatch_plan) {
    const AneProgram& program = bundle.manifest.programs[p];
    std::filesystem::path path;
    for (size_t i = 0; i < bundle.manifest.payloads.size(); ++i) {
      if (bundle.manifest.payloads[i].path == program.payload) {
        path = resolved[i];
        break;
      }
    }
    if (path.empty()) {
      throw bundle_error("program payload mapping disappeared after manifest validation");
    }
    AneAnecHeader header =
        parse_anec_header(path, bundle.manifest.tile_shift);
    validate_program_contract(
        program, header, path, p, uint64_t{1} << bundle.manifest.tile_shift);
    bundle.programs.push_back(
        {p, std::move(header), std::move(path), bundle.manifest.tile_shift});
  }
  return bundle;
}

AneBundle load_bundle(const std::filesystem::path& dir) {
  if (!std::filesystem::is_directory(dir)) {
    throw AneBundleNotFound(
        "[omarchy-ane] bundle directory not found: " + dir.string() +
        " (the affected region stays on Vulkan)");
  }

  AneManifest manifest = parse_ane_manifest(dir / "manifest.json");
  std::map<std::string, std::filesystem::path> payloads;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    std::error_code status_error;
    const auto status = entry.symlink_status(status_error);
    if (status_error) {
      throw bundle_error(
          "cannot stat '" + entry.path().filename().string() +
          "' inside bundle");
    }
    if (std::filesystem::is_symlink(status)) {
      throw bundle_error(
          "unexpected link '" + entry.path().filename().string() +
          "' inside bundle");
    }
    if (std::filesystem::is_directory(status)) {
      throw bundle_error(
          "unexpected directory '" + entry.path().filename().string() +
          "' inside bundle");
    }
    if (!std::filesystem::is_regular_file(status)) {
      throw bundle_error(
          "unexpected non-regular file '" + entry.path().filename().string() +
          "' inside bundle");
    }
    std::string name = entry.path().filename().string();
    if (name == "manifest.json") {
      continue;
    }
    if (std::none_of(
            manifest.payloads.begin(),
            manifest.payloads.end(),
            [&](const AnePayload& payload) { return payload.path == name; })) {
      throw bundle_error("unknown payload file '" + name + "' not listed in manifest");
    }
    payloads.emplace(std::move(name), entry.path());
  }
  return load_bundle_snapshot(dir / "manifest.json", payloads);
}

} // namespace mlx::core::omarchy::ane
