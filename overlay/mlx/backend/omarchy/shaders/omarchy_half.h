// Shared float32 -> float16 conversion for omarchy compute shaders.
//
// GLSL leaves the rounding of packHalf2x16 to the implementation: Mesa
// 26.x lowers it to round-toward-zero, older Mesa and other drivers round
// to nearest even. Shaders that must reproduce the eager float16_t store
// (and the CPU reference) therefore never use packHalf2x16 as a rounding
// step. omarchy_f16_bits_rne is an integer round-to-nearest-even
// conversion that gives the same bits on every driver (the float16_t
// round trip is avoided on purpose: it does not survive every compiler
// in registers, see fused_chain.comp and fast_trio.comp history).
//
// Overflow (|x| >= 65520) gives infinity, NaN stays a quiet NaN with the
// top ten payload bits kept, signed zero and subnormals are exact.
//
// Packing already-exact halves (unpackHalf2x16, packHalf2x16 of values
// that are representable) needs no helper.
//
// Do not copy this logic into another shader; include this header.

uint omarchy_f16_bits_rne(float value) {
  uint x = floatBitsToUint(value);
  uint sign = (x >> 16u) & 0x8000u;
  uint ax = x & 0x7FFFFFFFu;
  if (ax >= 0x7F800000u) {
    return sign | (ax > 0x7F800000u ? (0x7E00u | ((ax >> 13u) & 0x3FFu))
                                    : 0x7C00u);
  }
  if (ax >= 0x47800000u) {
    return sign | 0x7C00u;
  }
  if (ax >= 0x38800000u) {
    // Normal half: rebias the exponent, round on the 13 dropped bits.
    uint r = ax - 0x38000000u;
    return sign | ((r + 0xFFFu + ((r >> 13u) & 1u)) >> 13u);
  }
  if (ax < 0x33000000u) {
    return sign;  // at most 2^-25: rounds to zero (tie goes to even)
  }
  // Subnormal half: unit 2^-24, shift 14..24.
  uint mant = (ax & 0x7FFFFFu) | 0x800000u;
  uint shift = 126u - (ax >> 23u);
  uint q = mant >> shift;
  uint rem = mant & ((1u << shift) - 1u);
  uint half_way = 1u << (shift - 1u);
  if (rem > half_way || (rem == half_way && (q & 1u) != 0u)) {
    q += 1u;
  }
  return sign | q;
}

// The float32 value of x rounded to float16 (RNE).
float omarchy_round_f16(float value) {
  return unpackHalf2x16(omarchy_f16_bits_rne(value)).x;
}

// Two floats rounded to float16 (RNE) and packed low/high.
uint omarchy_pack_f16x2_rne(float lo, float hi) {
  return omarchy_f16_bits_rne(lo) | (omarchy_f16_bits_rne(hi) << 16u);
}
