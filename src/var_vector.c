// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "var_vector.h"

var_vector *
var_vec_init(size_t capacity, size_t el_size) {
    var_vector *me = malloc(sizeof(var_vector));
    me->array = malloc(el_size * capacity);
    me->capacity = capacity;
    me->used = 0;
    me->el_size = el_size;
    return me;
}

void
var_vec_free(var_vector *me) {
    free(me->array);
    free(me);
}

void
var_vec_grow(var_vector *me, size_t req) {
    size_t s1 = me->capacity + me->capacity / 2;
    size_t new_capacity = MAX(s1, req);
    me->array = realloc(me->array, new_capacity * me->el_size);
    me->capacity = new_capacity;
}

void
var_vec_add(var_vector *me, void *src) {
    me->used++;
    if (me->used > me->capacity) {
        var_vec_grow(me, 0);
    }
    memcpy(me->array + (me->used - 1) * me->el_size, src, me->el_size);
}

void
var_vec_add_sparse(var_vector *me,
                   size_t index, void *src, void *fill) {

    for (size_t i = me->used; i < index + 1; i++) {
        var_vec_add(me, fill);
    }
    assert(index < me->used);
    memcpy(me->array + index * me->el_size, src, me->el_size);
}


void
var_vec_remove(var_vector *me, void *dst) {
    assert(me->used > 0);
    me->used--;
    memcpy(dst, me->array + me->used * me->el_size, me->el_size);
}

void
var_vec_remove_at(var_vector *me, size_t i, void *dst) {
    assert(i < me->used);
    void *el_addr = me->array + me->el_size * i;
    memcpy(dst, el_addr, me->el_size);
    me->used--;
    size_t n_copy = (me->used - i) * me->el_size;
    memmove(el_addr, el_addr + me->el_size, n_copy);
}
