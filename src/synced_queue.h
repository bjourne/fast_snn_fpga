#ifndef SYNCED_QUEUE_H
#define SYNCED_QUEUE_H

#include <pthread.h>
#include <stdint.h>

typedef struct {
    pthread_spinlock_t lock;
    uint32_t *array;
    uint32_t capacity;
    uint32_t n_elements;
    uint32_t head;
    uint32_t tail;
} synced_queue;

synced_queue *synced_queue_init(size_t max);
void synced_queue_free(synced_queue *me);

void synced_queue_add(synced_queue *me, uint32_t value);
uint32_t synced_queue_remove(synced_queue *me);
void synced_queue_spin_while_nonempty(volatile synced_queue *me);
void synced_queue_spin_while_empty(volatile synced_queue *me);
void synced_queue_spin_while_full(volatile synced_queue *me);

#endif
