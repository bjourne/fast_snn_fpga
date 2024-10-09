// Copyright (C) 2019, 2023 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef ARRAY_BL_H
#define ARRAY_BL_H

#include <stdio.h>

typedef int (array_cmp_fun)(const void *a, const void *b, void *ctx);
typedef int (array_key_fun)(const void *a, void *ctx);

void
array_qsort(void *base, size_t nmemb, size_t size,
            int (*cmp_fun)(const void *a, const void *b, void *ctx),
            void *ctx);

size_t*
array_qsort_indirect(void *base, size_t nmemb, size_t size,
                     int (*cmp_fun)(const void *a, const void *b, void *ctx),
                     void *ctx);

// Quicksorting with a custom key function.
void
array_qsort_by_key(void *base, size_t nmemb, size_t size,
                   array_key_fun *fun, void *ctx);

void
array_permute(void *base, size_t nmemb, size_t size, size_t *indices);

// Builtin sorts
int
array_ord_asc_uint8_t(const void *a, const void *b, void *ctx);


#endif
