// Shared Q4 bf16 reference-cast products; exponent carrying avoids premature range loss.
#ifndef OMARCHY_Q4_BF16_PRECISION_H
#define OMARCHY_Q4_BF16_PRECISION_H

struct Q4Scaled { float hi; float lo; int exponent; };
Q4Scaled q4_scaled_zero() {
  return Q4Scaled(0.0, 0.0, 0);
}
// NaNs use a canonical quiet value; infinities retain IEEE signs.
// Finite additions use a 48-bit-significand source hypothesis, not a
// universal strict no-worse proof.
Q4Scaled q4_scaled_nan() {
  return Q4Scaled(uintBitsToFloat(0x7fc00000u), 0.0, 0);
}
Q4Scaled q4_normalize(float hi, float lo, int exponent) {
  precise float rounded = hi + lo;
  if (isnan(rounded)) {
    return q4_scaled_nan();
  }
  if (isinf(rounded)) {
    return Q4Scaled(rounded, 0.0, 0);
  }
  precise float virtual_low = rounded - hi;
  precise float tail = (hi - (rounded - virtual_low)) + (lo - virtual_low);
  if (rounded == 0.0) {
    return Q4Scaled(0.0, 0.0, 0);
  }
  int shift;
  float mantissa = frexp(rounded, shift);
  return Q4Scaled(mantissa, ldexp(tail, -shift), exponent + shift);
}
// Exponent-aligned accumulate: high words TwoSum, low words plus the
// carry error ride below, result renormalized to hi in [0.5, 1).
// Infinities pass through with IEEE addition semantics.
Q4Scaled q4_scaled_add(Q4Scaled a, Q4Scaled b) {
  if (isnan(a.hi) || isnan(a.lo) || isnan(b.hi) || isnan(b.lo)) {
    return q4_scaled_nan();
  }
  if (isinf(a.hi)) {
    if (isinf(b.hi) && a.hi != b.hi) {
      return q4_scaled_nan();
    }
    return a;
  }
  if (isinf(b.hi)) {
    return b;
  }
  if (a.hi == 0.0) {
    return b;
  }
  if (b.hi == 0.0) {
    return a;
  }
  int exponent = max(a.exponent, b.exponent);
  float ah = ldexp(a.hi, a.exponent - exponent);
  float al = ldexp(a.lo, a.exponent - exponent);
  float bh = ldexp(b.hi, b.exponent - exponent);
  float bl = ldexp(b.lo, b.exponent - exponent);
  precise float total = ah + bh;
  precise float virtual_low = total - ah;
  precise float error = (ah - (total - virtual_low)) + (bh - virtual_low);
  precise float low = (al + bl) + error;
  return q4_normalize(total, low, exponent);
}
// The reference's per-weight fp32 dequant: one rounding per multiply and
// per add (precise keeps NIR from contracting this into an fma, which
// would round once and disagree with the reference cast).
float q4_cast(float scale, float q, float bias) {
  precise float scaled = scale * q;
  precise float weight = scaled + bias;
  return weight;
}
// x (widened bf16) times the fp32-cast weight: exact 8x24 significand
// product, sign, exponent; no fp32 product rounding and no under/overflow.
// IEEE non-finite semantics are preserved: any NaN infects, Inf times zero
// is NaN, Inf times finite nonzero keeps the sign.
Q4Scaled q4_product(float x, float weight) {
  uint xb = floatBitsToUint(x);
  uint wb = floatBitsToUint(weight);
  uint xe = (xb >> 23u) & 0xffu;
  uint we = (wb >> 23u) & 0xffu;
  bool negative = ((xb ^ wb) & 0x80000000u) != 0u;
  if (xe == 0xffu || we == 0xffu) {
    bool nan = (xe == 0xffu && (xb & 0x7fffffu) != 0u)
        || (we == 0xffu && (wb & 0x7fffffu) != 0u);
    bool x_zero = xe == 0u && (xb & 0x7fffffffu) == 0u;
    bool w_zero = we == 0u && (wb & 0x7fffffffu) == 0u;
    bool x_inf = xe == 0xffu && !nan;
    bool w_inf = we == 0xffu && !nan;
    if (nan || (x_inf && w_zero) || (w_inf && x_zero)) {
      return q4_scaled_nan();
    }
    return Q4Scaled(uintBitsToFloat((negative ? 0xff800000u : 0x7f800000u)),
                    0.0, 0);
  }
  uint xm = ((xb >> 16u) & 0x7fu) | (xe == 0u ? 0u : 0x80u);
  uint wm = (wb & 0x7fffffu) | (we == 0u ? 0u : 0x800000u);
  uint significand = xm * wm;
  if (significand == 0u) {
    return Q4Scaled(0.0, 0.0, 0);
  }
  float high = float(significand);
  uint high_units = uint(high);
  // float(significand) can round UP, so uint(high) may exceed the
  // significand; the residual keeps its sign (subtract magnitudes).
  float low = significand >= high_units
      ? float(significand - high_units)
      : -float(high_units - significand);
  int exponent = (xe == 0u ? -133 : int(xe) - 134)
      + (we == 0u ? -149 : int(we) - 150) + 32;
  float signed_high = negative ? -ldexp(high, -32) : ldexp(high, -32);
  float signed_low = negative ? -ldexp(low, -32) : ldexp(low, -32);
  return q4_normalize(signed_high, signed_low, exponent);
}
// RNE bf16 store straight off the normalized (hi, lo, exponent): normal
// quant mantissa*256, subnormal mantissa*2^(exponent+133), guard and
// sticky from the low word, exponent < -133 rounds to zero, exponent
// > 128 saturates to infinity. The low word never travels through fp32
// scaling before the rounding decision, so no double rounding occurs.
uint16_t q4_scaled_store(Q4Scaled pair) {
  Q4Scaled n = q4_normalize(pair.hi, pair.lo, pair.exponent);
  if (isnan(n.hi)) {
    return uint16_t(0x7fc0u);
  }
  if (isinf(n.hi)) {
    return uint16_t(floatBitsToUint(n.hi) >> 16);
  }
  bool negative = n.hi < 0.0;
  uint sign = negative ? 0x8000u : 0u;
  float hi = negative ? -n.hi : n.hi;
  float lo = negative ? -n.lo : n.lo;
  int exponent = n.exponent;
  if (hi == 0.0 || exponent < -133) {
    return uint16_t(sign);
  }
  if (exponent > 128) {
    return uint16_t(sign | 0x7f80u);
  }
  bool normal = exponent >= -125;
  int shift = normal ? 8 : exponent + 133;
  float units = ldexp(hi, shift);
  float tail = ldexp(lo, shift);
  int integer = int(units);
  precise float fractional = units - float(integer);
  bool increment = fractional > 0.5
      || (fractional == 0.5
          && (tail > 0.0 || (tail == 0.0 && (integer & 1) == 1)));
  int result = integer + (increment ? 1 : 0);
  if (normal) {
    result += ((exponent + 126) << 7) - 128;
  }
  return uint16_t(sign | uint(result));
}

#define Q4_SCALED_TERMS(dst, p, lo, hi, sc, bi) { \
  float x0 = uintBitsToFloat(((lo) & 0xFFFFu) << 16u); \
  float x1 = uintBitsToFloat((lo) & 0xFFFF0000u); \
  float x2 = uintBitsToFloat(((hi) & 0xFFFFu) << 16u); \
  float x3 = uintBitsToFloat((hi) & 0xFFFF0000u); \
  (dst) = q4_scaled_add((dst), \
      q4_product(x0, q4_cast((sc), Q4_NIB(p, 0u), (bi)))); \
  (dst) = q4_scaled_add((dst), \
      q4_product(x1, q4_cast((sc), Q4_NIB(p, 4u), (bi)))); \
  (dst) = q4_scaled_add((dst), \
      q4_product(x2, q4_cast((sc), Q4_NIB(p, 8u), (bi)))); \
  (dst) = q4_scaled_add((dst), \
      q4_product(x3, q4_cast((sc), Q4_NIB(p, 12u), (bi)))); \
}

// One nonzero lane plus zeros copies exactly using the existing ARITHMETIC contract.
#define Q4_SCALED_FOLD_BROADCAST(total, mine) { \
  Q4Scaled q4_fold_source = (mine); \
  (total) = q4_scaled_zero(); \
  for (uint i = 0u; i < 32u; ++i) { \
    (total) = q4_scaled_add((total), Q4Scaled( \
        subgroupAdd(lane_in_slot == i ? q4_fold_source.hi : 0.0), \
        subgroupAdd(lane_in_slot == i ? q4_fold_source.lo : 0.0), \
        int(subgroupAdd(lane_in_slot == i ? uint(q4_fold_source.exponent) : 0u)))); \
  } \
}

#endif
