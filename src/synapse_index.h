// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef SYNAPSE_INDEX_H
#define SYNAPSE_INDEX_H

#define IS_HOST
#include "shared.h"

typedef struct {
    synapse *data;
    uint32_t *index;
    // Number of index entries per neuron
    uint32_t width;
    uint32_t n_neurons;
    uint32_t n_synapses;
} synapse_index;

synapse_index *
synapse_index_init(synapse *data, uint32_t *index, uint32_t width,
                   uint32_t n_neurons, uint32_t n_synapses);

void
synapse_index_free(synapse_index *me);

synapse_index *
synapse_index_frontier_index(synapse_index *me,
                             uint32_t n_padded_neurons,
                             uint32_t n_front,
                             uint32_t syn_align);

synapse_index *
synapse_index_partition_by_delay(synapse_index *me, uint32_t n_parts);

synapse_index *
synapse_index_merge_delays(synapse_index *me);

synapse_index *
synapse_index_align_delay_classes(synapse_index *me,
                                  uint32_t n_padded_neurons,
                                  uint32_t align);

#endif
