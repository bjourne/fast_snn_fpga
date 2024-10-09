// Copyright (C) 2023-2024 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "npy.h"
#include "sim_params.h"
#include "synced_queue.h"
#include "utils.h"

#define N_SENDERS               7

// Types of events the main thread sends to the senders.
#define SIM_FINISHED 0xff
#define SET_TIME 0xfe
#define DELIVER_SPIKE 0x00

typedef struct {
    pthread_t handle;
    synced_queue *queue;

    // Spike propagation buffer
    float *ibuf;
    size_t rowstride;

    // Synapse index
    uint32_t id;
    synapse_index *syn_idx;
} spike_sender_ctx;

spike_sender_ctx*
spike_sender_ctx_init(float *ibuf,
                      size_t rowstride,
                      synced_queue *queue,
                      uint32_t id,
                      synapse_index *syn_idx) {
    spike_sender_ctx *me = malloc(sizeof(spike_sender_ctx));
    me->queue = queue;
    me->ibuf = ibuf;
    me->rowstride = rowstride;
    me->id = id;
    me->syn_idx = syn_idx;
    return me;
}

void
spike_sender_ctx_free(spike_sender_ctx *me) {
    free(me);
}

static void
deliver_spike(float *ibuf,
              uint32_t id,
              synapse_index *syn_idx,
              size_t rowstride,
              size_t t, uint32_t src) {
    uint32_t base = 64 * syn_idx->n_neurons * id + 64 * src;
    uint32_t *idx = syn_idx->index;
    synapse *syns = syn_idx->data;

    // No synapse has delay = 0 or delay = 63.
    for (uint32_t i = 1; i < 63; i++) {
        uint32_t addr = base + i;
        uint32_t o0 = idx[addr];
        uint32_t o1 = idx[addr + 1];

        for (uint32_t o = o0; o < o1; o++) {
            synapse s = syns[o];
            uint32_t row = (i + t) & I_BUF_ROW_MASK;
            size_t idx = rowstride * row + s.dst;
            ibuf[idx] += s.weight;
        }
    }
}

static void *
spike_sender_run(void *arg) {
    spike_sender_ctx *ctx = (spike_sender_ctx *)arg;
    float *ibuf = ctx->ibuf;
    size_t rowstride = ctx->rowstride;
    size_t t = 0;
    uint32_t id = ctx->id;
    synapse_index *syn_idx = ctx->syn_idx;

    while (true) {
        uint32_t packet = synced_queue_remove(ctx->queue);
        uint32_t msg = packet & 0xffffff;
        uint32_t tp = packet >> 24;
        if (tp == SIM_FINISHED) {
            return NULL;
        } else if (tp == SET_TIME) {
            t = msg;
        } else if (tp == DELIVER_SPIKE) {
            uint32_t nid = msg;
            deliver_spike(ibuf, id, syn_idx, rowstride, t, nid);
        } else {
            assert(false);
        }
    }
}

static void
psc_update_double(size_t cnt, double c1, double c2,
           double * restrict v_ptr,
           uint8_t * restrict p_ptr,
           float * restrict t_ptr) {
    for (size_t i = 0; i < cnt; i++) {
        *v_ptr = *v_ptr * c1 + *p_ptr * c2 + *t_ptr;
        v_ptr++;
        p_ptr++;
        t_ptr++;
    }
}

static void
psc_update_single(size_t cnt, float c1, float c2,
                  float * restrict v_ptr,
                  uint8_t * restrict p_ptr,
                  float * restrict t_ptr) {
    for (size_t i = 0; i < cnt; i++) {
        *v_ptr = *v_ptr * c1 + *p_ptr * c2 + *t_ptr;
        v_ptr++;
        p_ptr++;
        t_ptr++;
    }
}

static inline void
vm_update_single(
    uint32_t t,
    size_t n_neurons,
    uint32_t n_layers,
    uint8_t t_ref_tics,
    float p20,
    float p21,
    float p22,
    float rel_v_thr,
    uint32_t *cnt_per_layer,
    float *dc_per_layer,
    float *vm,
    float *psc,
    int8_t *ref,
    synced_queue **queues,
    bool *rec
) {
    uint32_t at = 0;
    for (uint32_t i = 0; i < n_layers; i++) {
        uint32_t cnt_layer = cnt_per_layer[i];
        float z = dc_per_layer[i] * p20;
        for (uint32_t j = 0; j < cnt_layer; j++) {
            if (!ref[at]) {
                float x = vm[at] * p22 + z + psc[at] * p21;
                if (x >= rel_v_thr) {
                    uint32_t data = (DELIVER_SPIKE << 24) | at;
                    for (uint32_t k = 0; k < N_SENDERS; k++) {
                        synced_queue_add(queues[k], data);
                    }
                    rec[t * n_neurons + at] = true;
                    vm[at] = 0.0;
                    ref[at] = t_ref_tics;
                } else {
                    vm[at] = x;
                }
            } else {
                ref[at] -= 1;
            }
            at++;
        }
    }
}

static inline void
vm_update_double(
    size_t t,
    size_t n_neurons,
    uint32_t n_layers,
    uint8_t t_ref_tics,
    double p20,
    double p21,
    double p22,
    double rel_v_thr,
    uint32_t *cnt_per_layer,
    double *dc_per_layer,
    double *vm,
    double *psc,
    int8_t *ref,
    synced_queue **queues,
    bool *rec
) {
    size_t at = 0;
    for (uint32_t i = 0; i < n_layers; i++) {
        uint32_t cnt_layer = cnt_per_layer[i];
        double z = dc_per_layer[i] * p20;
        for (uint32_t j = 0; j < cnt_layer; j++) {
            if (!ref[at]) {
                double x = vm[at] * p22 + z + psc[at] * p21;
                if (x >= rel_v_thr) {
                    uint32_t data = (DELIVER_SPIKE << 24) | at;
                    for (uint32_t k = 0; k < N_SENDERS; k++) {
                        synced_queue_add(queues[k], data);
                    }
                    rec[t * n_neurons + at] = true;
                    vm[at] = 0.0;
                    ref[at] = t_ref_tics;
                } else {
                    vm[at] = x;
                }
            } else {
                ref[at] -= 1;
            }
            at++;
        }
    }
}

sim_result
sim_config_run_on_cpu(sim_config *me) { // sim_params *me, sim_flags *flags) {

    size_t n_neurons = me->index->n_neurons;
    size_t rowstride = me->rowstride;
    size_t n_float = sizeof(float);
    size_t n_double = sizeof(double);
    size_t n_per_el = me->double_prec ? n_double : n_float;

    // Spike buffer
    float *ibuf = aligned_calloc(I_BUF_ROWS * rowstride, n_float);

    // Refractory counters
    int8_t *ref = aligned_calloc(n_neurons, sizeof(int8_t));

    // Presynaptic currents
    void *psc = aligned_calloc(n_neurons, n_per_el);

    // DC per layer
    void *dc_per_layer = aligned_calloc(me->n_layers, n_per_el);
    for (uint32_t i = 0; i < me->n_layers; i++) {
        mprec_write_double(dc_per_layer, i, me->dc_per_layer[i],
                           me->double_prec);
    }
    synapse_index *new = synapse_index_partition_by_delay(me->index, N_SENDERS);
    synapse_index_free(me->index);
    me->index = new;

    // Create dedicated thread for posting spikes.
    // Queues
    synced_queue *queues[N_SENDERS];
    spike_sender_ctx *senders[N_SENDERS];
    for (uint32_t i = 0; i < N_SENDERS; i++) {
        queues[i] = synced_queue_init(512);
        senders[i] = spike_sender_ctx_init(
            ibuf, rowstride, queues[i], i, me->index);
        assert(!pthread_create(&senders[i]->handle, NULL,
                               spike_sender_run, senders[i]));
    }

    // Spike record
    bool *rec = calloc(me->n_sim_tics * n_neurons, sizeof(bool));

    // Current
    void *vm = calloc(n_neurons, n_per_el);

    // Subtract reset
    double rel_v_thr = me->v_thr - me->v_r;

    for (size_t i = 0; i < n_neurons; i++) {
        double v = me->v_m_init[i] - me->v_r;
        mprec_write_double(vm, i, v, me->double_prec);
    }

    double p20 = me->p20;
    double p21 = me->p21;
    double p22 = me->p22;

    printf("Using %d sender thread(s)...\n\n", N_SENDERS);
    uint64_t start = nano_count();
    for (size_t t = 0; t < me->n_sim_tics; t++) {
        uint32_t time_msg = (SET_TIME << 24) | t;
        for (uint32_t i = 0; i < N_SENDERS; i++) {
            synced_queue_add(queues[i], time_msg);
            synced_queue_spin_while_nonempty(queues[i]);
        }

        uint32_t ibuf_row = (t & I_BUF_ROW_MASK) * rowstride;
        size_t psn_row = t * rowstride;
        float *ibuf_ptr = &ibuf[ibuf_row];
        uint8_t *psn_ptr = &me->psn[psn_row];

        if (me->double_prec) {
            vm_update_double(
                t, n_neurons, me->n_layers, me->t_ref_tics,
                p20, p21, p22, rel_v_thr,
                me->cnt_per_layer, dc_per_layer, vm, psc, ref, queues, rec);
            psc_update_double(n_neurons, me->p11, me->psn_w,
                              psc, psn_ptr, ibuf_ptr);
        } else {
            vm_update_single(
                t, n_neurons, me->n_layers, me->t_ref_tics,
                p20, p21, p22, rel_v_thr,
                me->cnt_per_layer, dc_per_layer, vm, psc, ref, queues, rec);
            psc_update_single(n_neurons, me->p11, me->psn_w,
                              psc, psn_ptr, ibuf_ptr);
        }
        memset(ibuf_ptr, 0, sizeof(float) * rowstride);
    }
    for (uint32_t i = 0; i < N_SENDERS; i++) {
        synced_queue_add(queues[i], SIM_FINISHED << 24);
        assert(!pthread_join(senders[i]->handle, NULL));
    }
    sim_result res = {0, nano_count() - start, NULL};

    for (size_t i = 0; i < me->n_sim_tics; i++) {
        size_t c = 0;
        for (size_t j = 0; j < n_neurons; j++) {
            if (rec[n_neurons * i + j]) {
                c++;
            }
        }

        res.n_spikes += c;
    }
    res.arr = npy_init('b', 1,
                       2, (int[]){me->n_sim_tics, n_neurons},
                       rec, true);
    for (uint32_t i = 0; i < N_SENDERS; i++) {
        spike_sender_ctx_free(senders[i]);
        synced_queue_free(queues[i]);
    }
    free(dc_per_layer);
    free(vm);
    free(psc);
    free(ref);
    free(ibuf);
    free(rec);
    return res;
}
