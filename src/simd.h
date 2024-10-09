// Copyright (C) 2019, 2023 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef LINALG_SIMD_H
#define LINALG_SIMD_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include "synced_queue.h"

typedef __m128 float4;
typedef __m128i int4;

// float4 functions
inline float4
f4_load(float const *ptr) {
    return _mm_load_ps(ptr);
}

inline float4
f4_tern(float4 mask, float4 a, float4 b) {
    return _mm_blendv_ps(b, a, mask);
}

inline float4
f4_set_4x_i(int32_t a, int32_t b, int32_t c, int32_t d) {
    return _mm_castsi128_ps(_mm_set_epi32(a, b, c, d));
}

inline float4
f4_abs(float4 a) {
    float4 signmask =
        _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
    return _mm_andnot_ps(signmask, a);
}

inline float4
f4_signmask(float4 a) {
    float4 signmask =
        _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
    return _mm_and_ps(a, signmask);
}

inline bool
f4_all_eq(float4 a, float4 b) {
    __m128 cmp = _mm_cmpeq_ps(a, b);
    return _mm_movemask_ps(cmp) == 0xf;
}

inline float4
f4_scale(float4 a, float s) {
    return _mm_mul_ps(a, _mm_set1_ps(s));
}

// Check if this can be improved.
inline float4
f4_broadcast(float4 a, int i) {
    union { float4 reg; float f[4]; } r = { .reg = a };
    return _mm_set1_ps(r.f[i]);
}

inline float4
madd(float4 a, float4 b, float4 c) {
#if defined(__AVX2__)
    return _mm_fmadd_ps(a, b, c);
#else
    return _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
}

// int4 functions
inline int4
i4_load(int32_t *ptr) {
    return _mm_load_si128((const __m128i *)ptr);
}

inline int4
i4_set_4x(int32_t a, int32_t b, int32_t c, int32_t d) {
    // Note order
    return _mm_set_epi32(d, c, b, a);
}

inline int4
i4_set_1x(int32_t a) {
    return _mm_set1_epi32(a);
}

inline int4
i4_0() {
    return _mm_set1_epi32(0);
}

inline int4
i4_1() {
    return _mm_set1_epi32(1);
}

inline int4
i4_sub(int4 a, int4 b) {
    return _mm_sub_epi32(a, b);
}

inline bool
i4_all_eq(int4 a, int4 b) {
    int4 cmp = _mm_cmpeq_epi32(a, b);
    return _mm_movemask_ps((float4)cmp) == 0xf;
}

inline int4
i4_test(int4 a) {
    return _mm_cmpeq_epi32(a, i4_set_1x(0));
}

// Chooses a if mask is negative, else b.
inline int4
i4_tern(int4 mask, int4 a, int4 b) {
    return (int4)f4_tern((float4)mask, (float4)a, (float4)b);
}

inline void
i4_storeu(int4 a, int32_t *d) {
    _mm_storeu_si128((int4 *)d, a);
}

inline void
i4_store(int4 a, int32_t *d) {
    _mm_store_si128((int4 *)d, a);
}

inline void
i4_print(int4 r) {
    int32_t d[4];
    i4_storeu(r, d);
    // Note order
    printf("{%d, %d, %d, %d}", d[0], d[1], d[2], d[3]);
}

#ifdef __AVX2__

typedef __m256d double4;
typedef __m256i int8;
typedef __m256i long4;

// double4
inline double4
d4_andnot(double4 a, double4 b) {
    return _mm256_andnot_pd(a, b);
}

inline uint32_t
d4_movemask(double4 a) {
    return _mm256_movemask_pd(a);
}

inline double4
d4_load(double *ptr) {
    return _mm256_load_pd(ptr);
}

inline double4
d4_set_1x(double a) {
    return _mm256_set1_pd(a);
}

inline double4
d4_set_1x_f4(float4 a) {
    return _mm256_cvtps_pd(a);
}

inline double4
d4_0() {
    return d4_set_1x(0);
}

inline double4
d4_mul(double4 a, double4 b) {
    return _mm256_mul_pd(a, b);
}

inline double4
d4_add(double4 a, double4 b) {
    return _mm256_add_pd(a, b);
}

inline double4
d4_set_4x(double a, double b, double c, double d) {
    // Note order
    return _mm256_set_pd(d, c, b, a);
}

inline double4
d4_set_4x_i(int32_t a, int32_t b, int32_t c, int32_t d) {
    return _mm256_set_pd((double)d, (double)c, (double)b, (double)a);
}

inline double4
d4_cmp_gte(double4 a, double4 b) {
    return _mm256_cmp_pd(a, b, _CMP_GE_OQ);
}

inline double4
d4_tern(double4 mask, double4 a, double4 b) {
    return _mm256_blendv_pd(b, a, mask);
}

inline void
d4_store(double4 r, double *ptr) {
    _mm256_store_pd(ptr, r);
}

inline void
d4_storeu(double4 r, double *ptr) {
    _mm256_storeu_pd(ptr, r);
}

// long4
inline long4
l4_tern(long4 mask, long4 a, long4 b) {
    return (long4)d4_tern((double4)mask, (double4)a, (double4)b);
}

inline long4
l4_load(int64_t *ptr) {
    return _mm256_load_si256((long4 *)ptr);
}

inline long4
l4_sub(long4 a, long4 b) {
    return _mm256_sub_epi64(a, b);
}

inline long4
l4_set_1x(int64_t a) {
    return _mm256_set1_epi64x(a);
}

inline long4
l4_set_4x(int64_t a, int64_t b, int64_t c, int64_t d) {
    // Note order
    return _mm256_set_epi64x(d, c, b, a);
}

inline void
l4_print(long4 r) {
    int64_t d[4];
    _mm256_storeu_si256((__m256i *)d, r);
    printf("{%ld, %ld, %ld, %ld}", d[0], d[1], d[2], d[3]);
}

inline long4
l4_1() {
    return l4_set_1x(1);
}

inline long4
l4_and(long4 a, long4 b) {
    return _mm256_and_si256(a, b);
}

inline void
l4_store(long4 a, int64_t *d) {
    _mm256_store_si256((long4 *)d, a);
}

// int8
inline int8
i8_set_8x(int32_t a, int32_t b, int32_t c, int32_t d,
          int32_t e, int32_t f, int32_t g, int32_t h) {
    return _mm256_set_epi32(h, g, f, e, d, c, b, a);
}

inline void
i8_print(int8 r) {
    int32_t d[8];
    _mm256_storeu_si256((__m256i *)d, r);
    printf("{%d, %d, %d, %d, %d, %d, %d, %d}",
           d[0], d[1], d[2], d[3],
           d[4], d[5], d[6], d[7]);
}

inline int4
l4_cvt_i4(long4 r) {
    // Emulates _mm256_cvtepi64_epi32
    int8 pat = i8_set_8x(0, 2, 4, 6, 0, 0, 0, 0);
    return _mm256_castsi256_si128(
        _mm256_permutevar8x32_epi32(r, pat));
}

inline long4
i4_cvt_l4(int4 r) {
    return _mm256_cvtepi32_epi64(r);
}

inline double4
i4_cvt_d4(int4 r) {
    return (double4)i4_cvt_l4(r);
}

inline int4
d4_cvt_i4(double4 r) {
    return l4_cvt_i4((long4)r);
}

void
psc_update_avx2(uint32_t cnt, double c1, double c2,
                double * restrict psc,
                uint8_t * restrict psn,
                float * restrict ibuf);

uint32_t
update_vm_avx2(uint32_t cnt,
               double * restrict x_ptr,
               double * restrict y_ptr,
               int8_t *c_ptr,
               double c1, double c2, double c3,
               double thr, int8_t rst,
               synced_queue **queues, uint32_t at);
#endif

#endif
