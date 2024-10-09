// Copyright (C) 2023 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "npy.h"

static bool
consume(FILE *f, char *str) {
    for (size_t i = 0; i < strlen(str); i++) {
        char c = fgetc(f);
        if (feof(f) || c != str[i]) {
            return false;
        }
    }
    return true;
}

static bool
consume_until(FILE *f, char c) {
    while (!feof(f)) {
        if (fgetc(f) == c) {
            return true;
        }
    }
    return false;
}

void
npy_format_dims(npy_arr *arr, char *buf) {
    char *ptr = buf;
    int *dims = arr->dims;
    int n_dims = arr->n_dims;

    if (n_dims == 1) {
        sprintf(ptr, "(%d,)", dims[0]);
    } else {
        ptr += sprintf(ptr, "(");
        for (int i = 0; i < n_dims - 1; i++) {
            ptr += sprintf(ptr, "%d, ", dims[i]);
        }
        ptr += sprintf(ptr, "%d)", dims[n_dims - 1]);
    }
}

size_t
npy_n_elements(npy_arr *me) {
    size_t tot = 1;
    for (int i = 0; i < me->n_dims; i++) {
        tot *= me->dims[i];
    }
    return tot;
}

#define DOUBLE_CAST(tp_val, v_val, cast) \
    if (tp == (tp_val) && v == (v_val)) \
        return (double)((cast)me->data)[i];

double
npy_value_at_as_double(npy_arr *me, size_t i) {
    char tp = me->type;
    int v = me->el_size;

    // Macro-magic here.
    DOUBLE_CAST('i', 1, int8_t *);
    DOUBLE_CAST('i', 4, int32_t *);
    DOUBLE_CAST('i', 8, int64_t *);
    DOUBLE_CAST('u', 1, uint8_t *);
    DOUBLE_CAST('u', 4, uint32_t *);
    DOUBLE_CAST('u', 8, uint64_t *);
    DOUBLE_CAST('f', 4, float *);
    DOUBLE_CAST('f', 8, double *);
    assert(false);
}

static void
transpose_data(npy_arr *me) {
    assert(me->n_dims == 2);
    size_t n_els = npy_n_elements(me);
    int *dims = me->dims;
    int rows = dims[0];
    int cols = dims[1];
    int el_size = me->el_size;
    char *new = malloc(el_size * n_els);
    for (int x = 0; x < cols; x++) {
        for (int y = 0; y < rows; y++) {
            char *from = me->data + el_size * (x * rows + y);
            char *to = new + el_size * (y * cols + x);
            memcpy(to, from, el_size);
            from += el_size;
        }
    }
    free(me->data);
    me->data = new;
}

npy_arr *
npy_init(char type, int el_size,
         int n_dims, int *dims,
         void *data, bool copy) {
    npy_arr *me = malloc(sizeof(npy_arr));
    me->ver_maj = 1;
    me->ver_min = 0;
    me->type = type;
    me->el_size = el_size;
    memcpy(me->dims, dims, n_dims * sizeof(int));
    me->n_dims = n_dims;
    if (copy) {
        size_t n_bytes = el_size * npy_n_elements(me);
        me->data = malloc(n_bytes);
        memcpy(me->data, data, n_bytes);
    } else {
        me->data = data;
    }
    me->error_code = NPY_ERR_NONE;
    return me;
}


npy_arr *
npy_load(const char *fname) {
    npy_arr *me = malloc(sizeof(npy_arr));
    me->error_code = NPY_ERR_NONE;
    me->n_dims = 0;
    me->data = NULL;

    FILE *f = fopen(fname, "rb");
    if (!f) {
        me->error_code = NPY_ERR_OPEN_FILE;
        return me;
    }
    char buf[10];
    if (fread(buf, 1, 10, f) != 10) {
        goto truncation_error;
    }
    if (memcmp(buf, "\x93NUMPY", 6)) {
        me->error_code = NPY_ERR_READ_MAGIC;
        goto done;
    }
    me->ver_maj = buf[6];
    me->ver_min = buf[7];

    if (!consume(f, "{")) {
        goto truncation_error;
    }
    bool fortran_order = false;
    for (int k = 0; k < 3; k++) {
        char rbuf[128];
        if (fscanf(f, "%s", rbuf) != 1) {
            goto parse_error;
        }
        if (strstr(rbuf, "'descr':")) {
            if (fscanf(f, "%s", rbuf) != 1) {
                goto parse_error;
            }
            me->type = rbuf[2];
            if (sscanf(&rbuf[3], "%d", &me->el_size) != 1) {
                goto parse_error;
            }
        } else if (strstr(rbuf, "'fortran_order':")) {
            if (fscanf(f, "%s", rbuf) != 1) {
                goto parse_error;
            }
            if (strstr(rbuf, "False")) {
                fortran_order = false;
            } else if (strstr(rbuf, "True")) {
                fortran_order = true;
            } else {
                goto parse_error;
            }
        } else if (strstr(rbuf, "'shape':")) {
            if (!consume(f, " (")) {
                goto parse_error;
            }
            int dim;
            while (fscanf(f, " %d,", &dim) == 1) {
                me->dims[me->n_dims] = dim;
                me->n_dims++;
            }
            if (!consume(f, "),")) {
                goto parse_error;
            }
        } else {
            goto parse_error;
        }
    }
    if (!consume_until(f, '\n')) {
        goto parse_error;
    }
    size_t n_els = npy_n_elements(me);
    size_t n_bytes = n_els * me->el_size;
    me->data = (char *)malloc(n_bytes);
    if (fread(me->data, 1, n_bytes, f) != n_bytes) {
        goto truncation_error;
    }
    if (fortran_order) {
        transpose_data(me);
    }
 done:
    fclose(f);
    return me;
 parse_error:
    me->error_code = NPY_ERR_READ_HEADER;
    fclose(f);
    return me;
 truncation_error:
    me->error_code = NPY_ERR_READ_PAYLOAD;
    fclose(f);
    return me;
}

npy_error
npy_save(npy_arr *me, const char *fname) {
    FILE *f = fopen(fname, "wb");
    if (!f) {
        return NPY_ERR_OPEN_FILE;
    }
    char buf[2048];

    // Fix buffer overflows
    char shape[256];
    npy_format_dims(me, shape);

    // First format the descriptor so that we can compute the header
    // length.
    char byte_order = '<';
    if (me->type == 'S' || me->el_size == 1) {
        byte_order = '|';
    }
    char *fmt =
        "{'descr': '%c%c%d', 'fortran_order': False, 'shape': %s, }";
    size_t n_data = sprintf(buf + 10, fmt,
                            byte_order, me->type, me->el_size, shape);
    n_data += 10;
    size_t n_padding = 64 - n_data + n_data / 64 * 64;
    size_t n_header = n_data + n_padding;
    assert(n_header < 0xffff);

    memset(&buf[n_data], ' ', n_padding);
    buf[n_header - 1] = '\n';

    char *ptr = buf;
    ptr += sprintf(ptr, "\x93NUMPY");
    *ptr++ = me->ver_maj;
    *ptr++ = me->ver_min;
    *ptr++ = (n_header - 10) & 0xff;
    *ptr++ = (n_header - 10) >> 8;
    if (fwrite(buf, 1, n_header, f) != n_header) {
        return NPY_ERR_WRITE_HEADER;
    }
    // Then the data
    size_t n_els = npy_n_elements(me);
    if (fwrite(me->data, me->el_size, n_els, f) != n_els) {
        return NPY_ERR_WRITE_DATA;
    }
    fclose(f);
    return NPY_ERR_NONE;
}

void
npy_free(npy_arr *me) {
    if (me->data) {
        free(me->data);
    }
    free(me);
}

////////////////////////////////////////////////////////////////////////
// Pretty printing
////////////////////////////////////////////////////////////////////////
npy_pp *
npy_pp_init(int n_decimals, int n_columns, const char *sep) {
    npy_pp *me = (npy_pp *)malloc(sizeof(npy_pp));
    me->n_decimals = n_decimals;
    me->n_columns = n_columns;
    me->sep = sep;
    return me;
}

void
npy_pp_free(npy_pp *me) {
    free(me);
}

static void
indent(int n) {
    for (int i = 0; i < n; i++) {
        printf(" ");
    }
}

#define PRINT_EL_AT(tp_char, v_val, tp_tp)  \
    if (tp == (tp_char) && v == (v_val)) {  \
        printf(fmt, ((tp_tp *)data)[i]);    \
        printed = true;                     \
    }

static void
pp_value(npy_pp *me, int row_idx) {
    npy_arr *arr = me->arr;
    char tp = arr->type;
    char *data = arr->data;
    int n_dims = arr->n_dims;
    int v = arr->el_size;
    char *fmt = me->fmt;
    size_t i = me->value_idx;
    if (me->break_lines &&
        row_idx % me->n_items_per_line == 0 &&
        row_idx > 0) {
        printf("\n");
        indent(n_dims);
    }
    bool printed = false;
    PRINT_EL_AT('i', 1, int8_t);
    PRINT_EL_AT('i', 4, int32_t);
    PRINT_EL_AT('i', 8, int64_t);
    PRINT_EL_AT('u', 1, uint8_t);
    PRINT_EL_AT('u', 4, uint32_t);
    PRINT_EL_AT('u', 8, uint64_t);
    PRINT_EL_AT('f', 4, float);
    PRINT_EL_AT('f', 8, double);
    if (!printed) {
        const char *s = data + v * i;
        printf(fmt, s);
    }
    me->value_idx++;
}


static void
pp_row(npy_pp *me, bool is_first) {
    npy_arr *arr = me->arr;
    int n_dims = arr->n_dims;
    if (!is_first) {
        indent(n_dims - 1);
    }
    printf("[");

    int cnt = arr->dims[n_dims - 1];
    for (int i = 0; i < cnt - 1; i++) {
        pp_value(me, i);
        fputs(me->sep, stdout);
    }
    if (cnt) {
        pp_value(me, cnt - 1);
    }
    printf("]");
}


static void
pp_rec(npy_pp *me, int dim_idx, bool is_first) {
    npy_arr *arr = me->arr;
    int n_dims = arr->n_dims;
    int *dims = arr->dims;
    if (dim_idx == n_dims - 1) {
        pp_row(me, is_first);
    } else {
        if (!is_first) {
            indent(dim_idx);
        }
        printf("[");
        int n_els = dims[dim_idx];
        for (int i = 0; i < n_els - 1; i++) {
            pp_rec(me, dim_idx + 1, i == 0);
            printf(",\n");
        }
        pp_rec(me, dim_idx + 1, n_els == 1);
        printf("]");
    }
    if (dim_idx == 0) {
        printf("\n");
    }
}

// Figure out a good cell-width for the array
static size_t
cell_width(npy_pp *me, npy_arr *arr) {
    char tp = arr->type;
    size_t n_els = npy_n_elements(arr);
    size_t el_size = arr->el_size;
    size_t width = 0;
    if (tp == 'S') {
        for (size_t i = 0; i < n_els; i++) {
            size_t width0 = strnlen(arr->data + i * el_size, el_size);
            width = MAX(width, width0);
        }
        return width;
    }
    double max = 0;
    double min = 0;
    for (size_t i = 0; i < n_els; i++) {
        double at = npy_value_at_as_double(arr, i);
        if (at < min) {
            min = at;
        } else if (at > max) {
            max = at;
        }
    }
    int max_width = (int)ceil(log10(max));
    int min_width = (int)ceil(log10(-min)) + 1;
    width = MAX(max_width, min_width);
    if (tp  == 'f') {
        width += 1 + me->n_decimals;
    }
    return width;
}

void
npy_pp_print_arr(npy_pp *me, npy_arr *arr) {
    me->value_idx = 0;

    // Find suitable format string and element width.
    char tp = arr->type;
    int width = cell_width(me, arr);
    if (tp == 'S') {
        // Note the period which limits the maximum number of
        // characters printed.
        sprintf(me->fmt, "%%-%d.%ds", width, width);
    } else {
        if (arr->type == 'i') {
            sprintf(me->fmt, "%%%dd", width);
        } else if (arr->type == 'u') {
            sprintf(me->fmt, "%%%du", width);
        } else  {
            sprintf(me->fmt, "%%%d.%df", width, me->n_decimals);
        }
    }

    // Estimate line-width
    int n_dims = arr->n_dims;
    int n_sep = strlen(me->sep);
    int n_els = arr->dims[n_dims - 1];
    int line_length = n_els * (width + n_sep);
    me->break_lines = line_length > me->n_columns;
    me->n_items_per_line = (me->n_columns - n_dims) / (width + n_sep);

    // Begin recursive printing.
    me->value_idx = 0;
    me->arr = arr;
    pp_rec(me, 0, true);
}
