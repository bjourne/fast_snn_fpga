// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef PRETTY_H
#define PRETTY_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define PP_MAX_N_DIMS   10

typedef struct {
    size_t indent;
    size_t indent_width;
    size_t key_width;
    size_t n_decimals;
    size_t n_columns;
    char *sep;

    // Data set when pretty-printing arrays.
    void *arr;
    char type;
    size_t el_size;
    size_t n_dims;
    size_t dims[PP_MAX_N_DIMS];

    // Get updated when pp_print_array runs.
    bool is_first_on_line;
    char fmt[256];
    size_t value_idx;
    bool break_lines;
    size_t n_items_per_line;
} pretty_printer;

pretty_printer *
pp_init();

void
pp_free(pretty_printer *me);

void
pp_print_key_value(pretty_printer *me,
                   char *key,
                   char *value_fmt, ...);
void
pp_print_key_value_with_unit(
    pretty_printer *me,
    char *key,
    double quantity,
    char *unit
);

void
pp_print_array(
    pretty_printer *me,
    char type, size_t el_size,
    size_t n_dims, size_t dims[],
    void *arr
);

void
pp_humanize_quantity(double q, char *unit, size_t n_decimals, char *buf);


#endif
