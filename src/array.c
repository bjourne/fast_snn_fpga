// Copyright (C) 2019, 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "array.h"

typedef struct {
    void *ctx;
    array_cmp_fun *fun;
} compare_context;

static int
cmp_fun(
#if defined (_MSC_VER)
    void *ctx, const void *a, const void *b
#else
    const void *a, const void *b, void *ctx
#endif
) {
    compare_context *outer = (compare_context *)ctx;
    // I don't know if this is kosher...
    return outer->fun(a, b, outer->ctx);
}

void
array_qsort(void *base, size_t nmemb, size_t size,
            array_cmp_fun *fun, void *ctx) {
    compare_context outer_ctx = { ctx, fun };
#if defined(_MSC_VER)
    qsort_s(base, nmemb, size, cmp_fun, (void *)&outer_ctx);
#else
    qsort_r(base, nmemb, size, cmp_fun, (void *)&outer_ctx);
#endif
}

typedef struct {
    void *ctx;
    array_key_fun *fun;
} key_compare_context;

static int
key_cmp_fun(const void *a, const void *b, void *ctx) {
    key_compare_context *wrap = (key_compare_context *)ctx;
    int k1 = wrap->fun(a, wrap->ctx);
    int k2 = wrap->fun(b, wrap->ctx);
    return k1 - k2;
}

void
array_qsort_by_key(void *base, size_t nmemb, size_t size,
                   array_key_fun *fun, void *ctx) {
    key_compare_context wrap = { ctx, fun };
    array_qsort(base, nmemb, size, key_cmp_fun, (void *)&wrap);
}

typedef struct {
    void *ctx;
    array_cmp_fun *fun;
    void *base;
    size_t size;
} indirect_compare_context;

static int
ind_cmp_fun(const void *a, const void *b, void *ctx) {
    indirect_compare_context *wrap = (indirect_compare_context *)ctx;
    size_t ai = (size_t)a & 0xffffffff;
    size_t bi = (size_t)b & 0xffffffff;
    void *av = wrap->base + wrap->size * ai;
    void *bv = wrap->base + wrap->size * bi;
    return wrap->fun(*(void **)av, *(void **)bv, wrap->ctx);
}

size_t*
array_qsort_indirect(void *base, size_t nmemb, size_t size,
                     array_cmp_fun *fun, void *ctx) {
    size_t *indices = malloc(nmemb * sizeof(size_t));
    for (size_t i = 0; i < nmemb; i++) {
        indices[i] = i;
    }
    indirect_compare_context wrap = { ctx, fun, base, size };
    array_qsort(indices, nmemb, sizeof(size_t),
                ind_cmp_fun, (void *)&wrap);


    return indices;
}

void
array_permute(void *base, size_t nmemb, size_t size, size_t *indices) {
    for (size_t i = 0; i < nmemb - 1; i++) {
        size_t ind = indices[i];
        while (ind < i) {
            ind = indices[ind];
        }
        if (size == 1) {
            uint8_t *base8 = (uint8_t *)base;
            uint8_t tmp = base8[i];
            base8[i] = base8[ind];
            base8[ind] = tmp;
        } else if (size == 4) {
            uint32_t *base32 = (uint32_t *)base;
            uint32_t tmp = base32[i];
            base32[i] = base32[ind];
            base32[ind] = tmp;
        } else if (size == 8) {
            uint64_t *base64 = (uint64_t *)base;
            uint64_t tmp = base64[i];
            base64[i] = base64[ind];
            base64[ind] = tmp;
        } else {
            assert(false);
        }
    }
}

int
array_ord_asc_uint8_t(const void *a,
                      const void *b,
                      void *ctx __attribute__((unused))) {
    uint8_t ai = (int)((uintptr_t)a & 0xff);
    uint8_t bi = (int)((uintptr_t)b & 0xff);
    return ai - bi;
}
