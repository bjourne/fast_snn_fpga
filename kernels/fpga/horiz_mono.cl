// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
// Flags  : horiz/mono/{s,d}
#define IS_DEVICE       1
#include "src/shared.h"

#define N_NEURON_QUEUE  128
#define N_SYNAPSE_QUEUE 4096
#define PSC_DEPTH  (N_FIXED_NEURONS / SYN_ALIGN)

__attribute__((uses_global_work_offset(0)))
__attribute__((max_global_work_dim(0)))
__kernel void
netsim(
    uint n_tics,
    char t_ref_tics,
    FTYPE rel_v_thr,
    FTYPE p11,
    FTYPE p21,
    FTYPE p22,
    FTYPE psn_w,
    __global const FTYPE * restrict gl_vm,
    __global const volatile uchar * restrict gl_psn,
    __global const volatile synapse * restrict gl_syns,
    __global uint * restrict spike_record,
    __global const volatile uint * restrict gl_idx
) {

    // Neuron data
    FTYPE vm[N_FIXED_NEURONS];
    char ref[N_FIXED_NEURONS];
    FTYPE psc[N_FIXED_NEURONS];

    #pragma disable_loop_pipelining
    for (uint i = 0; i < N_FIXED_NEURONS; i++) {
        vm[i] = gl_vm[i];
    }

    // Spiking queue
    uint nqueue[64][N_NEURON_QUEUE];
    uint nqueue_n[64];
    #pragma disable_loop_pipelining
    for (uint i = 0; i < 64; i++) {
        nqueue_n[i] = 0;
    }

    // Horizon
    float horizon[N_FRONT][PSC_DEPTH][SYN_ALIGN]
        __attribute__((numbanks(SYN_ALIGN), bankwidth(4)));
    #pragma disable_loop_pipelining
    for (uint i = 0; i < PSC_DEPTH; i++) {
        #pragma ivdep
        #pragma unroll SYN_ALIGN
        for (uint j = 0; j < SYN_ALIGN; j++) {
            for (uint k = 0; k < N_FRONT; k++) {
                horizon[k][i][j] = 0;
            }
        }
    }

    // Spike count
    uint n_spikes = 0;
    uint psn_t = 0;

    #pragma disable_loop_pipelining
    for (uint t = 0; t < n_tics; t++) {

        range arr[N_SYNAPSE_QUEUE];
        uint squeue_n = 0;

        // psn cache
        uchar psn_cache[N_FIXED_NEURONS];
        #pragma ivdep
        #pragma unroll 64
        #pragma ii 1
        for (uint i = 0; i < N_FIXED_NEURONS; i++) {
            psn_cache[i] = gl_psn[N_FIXED_NEURONS * psn_t + i];
        }

        #pragma ii 1
        #pragma ivdep
        for (uint i = 0; i < N_TURNS; i++) {
            uint t_from = (t - N_FRONT * i - 1) & 63;
            uint c = nqueue_n[t_from];

            #pragma ii 1
            #pragma ivdep
            for (uint j = 0; j < c; j++) {
                uint nid = nqueue[t_from][j];
                uint o0 = gl_idx[N_TURNS * nid + i];
                uint o1 = gl_idx[N_TURNS * nid + i + 1];
                if (o0 != o1) {
                    arr[squeue_n++] = (range){o0, o1};
                }
            }
        }

        for (uint i = 0; i < squeue_n; i++) {
            uint o0 = arr[i].o0;
            uint o1 = arr[i].o1;

            #pragma ii 1
            #pragma ivdep
            for (uint j = o0; j < o1; j++) {
                #pragma unroll
                for (uint k = 0; k < SYN_ALIGN; k++) {
                    synapse s = gl_syns[SYN_ALIGN * j + k];
                    uint dst = s.dst & 0xffffff;
                    uint del = s.dst >> 24;
                    uint row = (del + t - 1) & (N_FRONT - 1);
                    horizon[row][dst][k] += s.weight;
                }
            }
        }

        // send buf
        uint send_buf[64][SYN_ALIGN] __attribute__((numbanks(SYN_ALIGN), bankwidth(4)));
        uint n_send[SYN_ALIGN] = {0};

        // horizon pos
        uint t_h = t & (N_FRONT - 1);

        #pragma ii 1
        #pragma ivdep
        for (uint i = 0; i < PSC_DEPTH; i++) {
            #pragma ivdep
            #pragma unroll
            for (uint j = 0; j < SYN_ALIGN; j++) {
                uint nid = SYN_ALIGN * i + j;

                FTYPE p = 0;
                char cnt = 0;
                if (t) {
                    p = psc[nid];
                    cnt = ref[nid];
                }
                FTYPE v = vm[nid];
                FTYPE x = v * p22 + p * p21;

                char is_spike0 = x >= rel_v_thr;
                char is_send = !cnt && is_spike0;

                v = !cnt && !is_spike0 ? x : 0.0f;
                cnt = is_send ? t_ref_tics : (cnt != 0 ? cnt - 1 : 0);

                vm[nid] = v;
                ref[nid] = cnt;

                if (is_send) {
                    uint at = n_send[j];
                    send_buf[at][j] = nid;
                    n_send[j]++;
                }

                FTYPE psn0 = (FTYPE)psn_cache[nid];
                psc[nid] = p * p11 + psn0 * psn_w + horizon[t_h][i][j];
                horizon[t_h][i][j] = 0;
            }
        }

        // Queue marked neurons
        uint n = 0;
        uint t_at = t & (MAX_D - 1);

        for (uint i = 0; i < SYN_ALIGN; i++) {
            uint c = n_send[i];
            #pragma ivdep
            for (uint j = 0; j < c; j++) {
                uint nid = send_buf[j][i];
                nqueue[t_at][n + j] = nid;
            }
            n += c;
        }
        nqueue_n[t_at] = n;
        n_spikes += n;

        if (++psn_t == MAX_PSN_T) {
            psn_t = 0;
        }
    }

    spike_record[0] = n_spikes;
}
