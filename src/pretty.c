// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "pretty.h"

static void
indent(size_t n) {
    for (size_t i = 0; i < n; i++) {
        printf(" ");
    }
}

pretty_printer *
pp_init() {
    pretty_printer *me = malloc(sizeof(pretty_printer));
    me->indent = 0;
    me->indent_width = 2;
    me->key_width = 15;
    me->n_decimals = 3;
    me->n_columns = 80;
    me->sep = " ";
    return me;
}

void
pp_free(pretty_printer *me) {
    free(me);
}

static void
pp_print_prefix(pretty_printer *me) {
    indent(me->indent * me->indent_width);
}

void
pp_print_key_value(
    pretty_printer *me,
    char *key,
    char *value_fmt, ...) {

    char buf[2048];
    sprintf(buf, "%%-%lds: ", me->key_width);
    pp_print_prefix(me);
    printf(buf, key);

    va_list ap;
    va_start(ap, value_fmt);
    vsprintf(buf, value_fmt, ap);
    va_end(ap);
    printf("%s\n", buf);
}

void
pp_print_key_value_with_unit(
    pretty_printer *me,
    char *key,
    double quantity,
    char *unit
) {
    char buf[256];
    pp_humanize_quantity(quantity, unit, me->n_decimals, buf);
    pp_print_key_value(me, key, buf);
}

static char *
prefixes[] ={"", "k", "M", "G"};

void
pp_humanize_quantity(double q, char *unit, size_t n_decimals, char *buf) {
    size_t i = 0;
    while (q > 1000.0) {
        i++;
        q = q / 1000.0;
    }
    assert(i <= 3);
    char *pf = prefixes[i];
    char fmt[256];
    sprintf(fmt, "%%.%ldf %%s%%s", n_decimals);
    sprintf(buf, fmt, q, pf, unit);
}

#define PRINT_EL_AT(tp_char, v_val, tp_tp)  \
    if (tp == (tp_char) && v == (v_val)) {  \
        printf(fmt, ((tp_tp *)data)[i]);    \
        printed = true;                     \
    }

static void
pp_value(pretty_printer *me, size_t row_idx) {
    char tp = me->type;
    char *data = me->arr;
    size_t v = me->el_size;
    char *fmt = me->fmt;
    size_t i = me->value_idx;
    if (me->break_lines &&
        row_idx % me->n_items_per_line == 0 &&
        row_idx > 0) {
        size_t n_dims = me->n_dims;
        printf("\n");
        pp_print_prefix(me);
        indent(n_dims);
    }
    bool printed = false;
    PRINT_EL_AT('i', 1, int8_t);
    PRINT_EL_AT('i', 4, int32_t);
    PRINT_EL_AT('i', 8, int64_t);
    PRINT_EL_AT('u', 1, uint8_t);
    PRINT_EL_AT('b', 1, uint8_t);
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

// Prints WITHOUT newline ending
static void
pp_puts(pretty_printer *me, char *s) {
    if (me->is_first_on_line) {
        pp_print_prefix(me);
        me->is_first_on_line = false;
    }
    printf("%s", s);
}

static void
pp_linebreak(pretty_printer *me) {
    printf("\n");
    me->is_first_on_line = true;
}

static void
pp_row(pretty_printer *me, size_t dim_idx) {
    size_t n_dims = me->n_dims;
    size_t cnt = me->dims[n_dims - 1];
    if (me->is_first_on_line) {
        indent(dim_idx);
    }
    pp_puts(me, "[");
    if (cnt > 0) {
        for (size_t i = 0; i < cnt - 1; i++) {
            pp_value(me, i);
            fputs(me->sep, stdout);
        }
        if (cnt) {
            pp_value(me, cnt - 1);
        }
    }
    pp_puts(me, "]");
}

static void
pp_rec(pretty_printer *me, size_t dim_idx) {
    size_t n_dims = me->n_dims;
    size_t *dims = me->dims;
    if (dim_idx == n_dims - 1) {
        pp_row(me, dim_idx);
    } else {
        if (me->is_first_on_line) {
            indent(dim_idx);
        }
        pp_puts(me, "[");
        size_t n_els = dims[dim_idx];
        for (size_t i = 0; i < n_els - 1; i++) {
            pp_rec(me, dim_idx + 1);
            pp_puts(me, me->sep);
            pp_linebreak(me);
        }
        pp_rec(me, dim_idx + 1);
        pp_puts(me, "]");
    }
    if (dim_idx == 0) {
        pp_linebreak(me);
    }
}

static size_t
n_elements(size_t n_dims, size_t dims[]) {
    size_t tot = 1;
    for (size_t i = 0; i < n_dims; i++) {
        tot *= dims[i];
    }
    return tot;
}

#define DOUBLE_CAST(tp_val, v_val, cast) \
    if (type == (tp_val) && el_size == (v_val)) \
        return (double)((cast)arr)[i];

static double
value_at_as_double(char type, size_t el_size, size_t i, void *arr) {
    // Macro-magic here.
    DOUBLE_CAST('i', 1, int8_t *);
    DOUBLE_CAST('i', 4, int32_t *);
    DOUBLE_CAST('i', 8, int64_t *);
    DOUBLE_CAST('b', 1, uint8_t *);
    DOUBLE_CAST('u', 1, uint8_t *);
    DOUBLE_CAST('u', 4, uint32_t *);
    DOUBLE_CAST('u', 8, uint64_t *);
    DOUBLE_CAST('f', 4, float *);
    DOUBLE_CAST('f', 8, double *);
    assert(false);
}

// Figure out a good cell-width for the array
static size_t
cell_width(
    size_t n_decimals,
    char type, size_t el_size,
    size_t n_els, void *arr) {
    size_t width = 0;
    if (type == 'S') {
        for (size_t i = 0; i < n_els; i++) {
            size_t width0 = strnlen(arr + i * el_size, el_size);
            width = MAX(width, width0);
        }
        return width;
    }
    double max = 1;
    double min = 1;
    for (size_t i = 0; i < n_els; i++) {
        double at = value_at_as_double(type, el_size, i, arr);
        if (at < min) {
            min = at;
        } else if (at > max) {
            max = at;
        }
    }
    int max_width = (int)ceil(log10(max));
    int min_width = (int)ceil(log10(-min)) + 1;
    width = MAX(max_width, min_width);
    if (type  == 'f') {
        width += 1 + n_decimals;
    }
    return width;
}

void
pp_print_array(
    pretty_printer *me,
    char type, size_t el_size,
    size_t n_dims, size_t dims[],
    void *arr
) {
    me->type = type;
    me->el_size = el_size;
    me->n_dims = n_dims;
    memcpy(me->dims, dims, sizeof(size_t) * n_dims);

    // Find suitable format string and element width.
    size_t n_els = n_elements(n_dims, dims);
    size_t width = cell_width(me->n_decimals,
                              type, el_size,
                              n_els, arr);
    if (type == 'S') {
        // Note the period which limits the maximum number of
        // characters printed.
        sprintf(me->fmt, "%%-%ld.%lds", width, width);
    } else if (type == 'i') {
        sprintf(me->fmt, "%%%ldd", width);
    } else if (type == 'u' || type == 'b') {
        sprintf(me->fmt, "%%%ldu", width);
    } else  {
        sprintf(me->fmt, "%%%ld.%ldf", width, me->n_decimals);
    }

    // Estimate line-width
    size_t n_sep = strlen(me->sep);
    size_t n_els_per_row = dims[n_dims - 1];
    size_t line_length = n_els_per_row * (width + n_sep);
    me->break_lines = line_length > me->n_columns;
    me->n_items_per_line = (me->n_columns - n_dims) / (width + n_sep);

    me->value_idx = 0;
    me->arr = arr;
    me->is_first_on_line = true;
    pp_rec(me, 0);
}
