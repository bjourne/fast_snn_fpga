// Copyright (C) 2023-2024 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "common.h"
#include "synapse_index.h"
#include "utils.h"
#include "var_vector.h"

synapse_index *
synapse_index_init(synapse *data, uint32_t *index, uint32_t width,
                   uint32_t n_neurons, uint32_t n_synapses) {
    synapse_index *me = malloc(sizeof(synapse_index));
    me->data = data;
    me->index = index;
    me->width = width;
    me->n_neurons = n_neurons;
    me->n_synapses = n_synapses;
    return me;
}

void
synapse_index_free(synapse_index *me) {
    free(me->data);
    free(me->index);
    free(me);
}


static synapse
pack_synapse(uint32_t delay, uint32_t dst, float weight, uint32_t syn_align) {
    return (synapse){(delay << 24) | (dst / syn_align), weight};
}

static void
print_index_info(synapse_index *old, var_vector *syns,
                 uint32_t n_front, size_t n_els, uint32_t syn_align) {
    double occ = (double)old->n_synapses / (double)syns->used;
    printf("%d-tics-front index; %ld elements, syn. align %d, occ %.2f\n",
           n_front, n_els, syn_align, occ);
}

synapse_index *
synapse_index_frontier_index(synapse_index *me,
                             uint32_t n_padded_neurons,
                             uint32_t n_front,
                             uint32_t syn_align) {

    uint32_t n_neurons = me->n_neurons;
    uint32_t n_turns = 64 / n_front;


    var_vector *syns = var_vec_init(4, sizeof(synapse));
    size_t n_els = n_turns * n_neurons + 1;
    uint32_t *index = aligned_calloc(n_els, sizeof(uint32_t));

    uint32_t idemp_idx = n_padded_neurons - syn_align;
    synapse s_pad = pack_synapse(0, idemp_idx, 0, syn_align);

    uint32_t at = 0;
    index[at++] = 0;

    for (uint32_t i = 0; i < n_neurons; i++) {
        for (uint32_t j = 0; j < n_turns; j++) {
            uint32_t base = syns->used;
            uint32_t order[64] = {0};
            for (uint32_t k = 0; k < n_front; k++) {
                uint32_t del = n_front * j + k + 1;
                uint32_t o0 = me->index[MAX_D * i + MIN(del, 63)];
                uint32_t o1 = me->index[MAX_D * i + MIN(del + 1, 63)];
                for (uint32_t o = o0; o < o1; o++) {
                    synapse s = me->data[o];
                    uint32_t c = s.dst % syn_align;
                    uint32_t abs_idx = base + syn_align * order[c] + c;
                    assert(del != 0);
                    synapse s_new = pack_synapse(
                        del, s.dst, s.weight, syn_align);
                    var_vec_add_sparse(syns, abs_idx, &s_new, &s_pad);
                    order[c]++;
                }
            }
            uint32_t pad_to = syn_align - (syns->used % syn_align);
            pad_to = syns->used + (pad_to == syn_align ? 0 : pad_to);
            for (uint32_t k = syns->used; k < pad_to; k++) {
                var_vec_add(syns, &s_pad);
            }
            assert(syns->used % syn_align == 0);
            index[at++] = syns->used / syn_align;
        }
    }
    assert(at == n_turns * n_neurons + 1);
    print_index_info(me, syns, n_front, n_els, syn_align);

    synapse_index *new =
        synapse_index_init(syns->array, index, n_turns,
                           n_neurons, syns->used);
    free(syns);
    return new;
}

synapse_index *
synapse_index_align_delay_classes(synapse_index *me,
                                  uint32_t n_padded_neurons, uint32_t syn_align) {
    assert(syn_align > 0);
    assert(n_padded_neurons - syn_align > me->n_neurons);
    printf("Aligning delay classes to %d...\n", syn_align);


    var_vector *syns = var_vec_init(4, sizeof(synapse));
    size_t n_els = MAX_D * me->n_neurons + 1;
    uint32_t *index = aligned_calloc(n_els, sizeof(uint32_t));

    uint32_t at = 0;
    index[at++] = 0;

    // Index of filler neuron
    uint32_t idemp_idx = (n_padded_neurons - syn_align) / syn_align;
    synapse s_pad = (synapse){idemp_idx, 0};

    for (uint32_t i = 0; i < me->n_neurons; i++) {
        for (uint32_t j = 0; j < MAX_D; j++) {

            uint32_t k0 = me->index[MAX_D * i + j];
            uint32_t k1 = me->index[MAX_D * i + j + 1];
            k1 = MAX(k1, k0);
            assert(k0 <= k1);

            uint32_t order[64] = {0};
            uint32_t base = syns->used;

            assert(base % syn_align == 0);
            for (uint32_t k = k0; k < k1; k++) {
                synapse s = me->data[k];
                uint32_t c = s.dst % syn_align;
                uint32_t rel_idx = syn_align * order[c] + c;
                uint32_t abs_idx = base + rel_idx;

                synapse s_new = {s.dst / syn_align, s.weight};
                var_vec_add_sparse(syns, abs_idx, &s_new, &s_pad);
                order[c]++;
            }
            uint32_t pad_to = syn_align - (syns->used % syn_align);
            pad_to = syns->used + (pad_to == syn_align ? 0 : pad_to);
            for (uint32_t k = syns->used; k < pad_to; k++) {
                var_vec_add(syns, &s_pad);
            }
            assert(syns->used % syn_align == 0);
            index[at++] = syns->used / syn_align;
        }
    }
    print_index_info(me, syns, 1, n_els, syn_align);
    synapse_index *new =
        synapse_index_init(syns->array, index, MAX_D,
                           me->n_neurons, syns->used);
    free(syns);
    return new;
}


////////////////////////////////////////////////////////////////////////
// Gunky code here
////////////////////////////////////////////////////////////////////////

synapse_index *
synapse_index_partition_by_delay(synapse_index *me, uint32_t n_parts) {
    printf("Partitioning index into %d partition(s)...\n", n_parts);

    uint32_t *index = aligned_calloc(n_parts * 64 * me->n_neurons + 1,
                                     sizeof(uint32_t));
    uint32_t at = 0;
    var_vector *syns = var_vec_init(4, sizeof(synapse));
    index[at++] = 0;
    for (uint32_t i = 0; i < n_parts; i++) {
        for (uint32_t j = 0; j < me->n_neurons; j++) {
            for (uint32_t k = 0; k < 64; k++) {
                if (k % n_parts == i) {
                    uint32_t l0 = me->index[64 * j + k];
                    uint32_t l1 = me->index[64 * j + k + 1];

                    // Compensate for bug in index generation
                    l1 = MAX(l1, l0);

                    for (uint32_t l = l0; l < l1; l++) {
                        synapse s = me->data[l];
                        var_vec_add(syns, &s);
                    }
                }
                index[at++] = syns->used;
            }
        }
    }
    assert(syns->used == me->n_synapses);
    synapse_index *new =
        synapse_index_init(syns->array, index, 64,
                           me->n_neurons, syns->used);
    free(syns);
    return new;
}

static void
expand_for_neuron(synapse_index *me,
                  uint32_t nid, var_vector *syns) {
    uint32_t order[N_GPU_WORK_ITEMS] = {0};
    synapse fill = (synapse){0xffffffff, 0.0};
    uint32_t base = syns->used;
    for (uint32_t i = 0; i < 64; i++) {
        uint32_t o0 = me->index[64 * nid + i];
        uint32_t o1 = me->index[64 * nid + i + 1];
        for (uint32_t o = o0; o < o1; o++) {
            synapse s = me->data[o];
            uint32_t c = s.dst & N_GPU_WORK_ITEMS_MASK;
            uint32_t idx = N_GPU_WORK_ITEMS * order[c] + c;
            s.dst = (s.dst << 8) | i;
            var_vec_add_sparse(syns, base + idx, &s, &fill);
            order[c]++;
        }
    }
}

synapse_index *
synapse_index_merge_delays(synapse_index *me) {
    printf("Merging synapse index...\n");
    var_vector *syns = var_vec_init(4, sizeof(synapse));
    uint32_t *index = aligned_calloc(me->n_neurons + 1, sizeof(uint32_t));
    for (uint32_t i = 0; i < me->n_neurons; i++) {
        index[i] = syns->used;
        expand_for_neuron(me, i, syns);
    }
    index[me->n_neurons] = syns->used;
    synapse_index *new =
        synapse_index_init(syns->array, index, 1,
                           me->n_neurons, syns->used);
    free(syns);
    return new;
}
