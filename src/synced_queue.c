// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "synced_queue.h"

synced_queue *
synced_queue_init(size_t capacity) {
    synced_queue *me = malloc(sizeof(synced_queue));
    me->array = malloc(sizeof(uint32_t) * capacity);
    me->capacity = capacity;
    me->head = me->tail = me->n_elements = 0;
    assert(!pthread_spin_init(&me->lock, PTHREAD_PROCESS_PRIVATE));
    return me;
}

void
synced_queue_free(synced_queue *me) {
    assert(!pthread_spin_destroy(&me->lock));
    free(me->array);
    free(me);
}

void
synced_queue_spin_while_empty(volatile synced_queue *me) {
    while (me->n_elements == 0) {
    }
}

void
synced_queue_spin_while_full(volatile synced_queue *me) {
    while (me->n_elements == me->capacity) {
    }
}

void
synced_queue_spin_while_nonempty(volatile synced_queue *me) {
    while (me->n_elements > 0) {
    }
}

void
synced_queue_add(synced_queue *me, uint32_t value) {
    synced_queue_spin_while_full(me);
    pthread_spin_lock(&me->lock);
    uint32_t next = (me->head + 1) % (me->capacity + 1);
    me->array[me->head] = value;
    me->head = next;
    me->n_elements++;
    pthread_spin_unlock(&me->lock);
}

uint32_t
synced_queue_remove(synced_queue *me) {
    synced_queue_spin_while_empty(me);
    pthread_spin_lock(&me->lock);
    uint32_t next = (me->tail + 1) % (me->capacity + 1);
    uint32_t value = me->array[me->tail];
    me->tail = next;
    me->n_elements--;
    pthread_spin_unlock(&me->lock);
    return value;
}
