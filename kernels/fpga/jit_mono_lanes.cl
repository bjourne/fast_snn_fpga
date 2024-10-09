// Copyright (C) 2023-2024 Björn A. Lindqvist <bjourne@gmail.com>
//
// Flags  : jit/mono/{s,d}
#define IS_DEVICE
#include "src/shared.h"

#define N_NEURON_QUEUE  128
#define N_SYNAPSE_QUEUE 2048
#define N_PSC_PER_BANK  (N_FIXED_NEURONS / SYN_ALIGN)
#define N_LANES         16

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
    __global uint * restrict gl_spike_record,
    __global const volatile uint * restrict gl_idx
) {
    // Initialize data
    FTYPE vm[N_FIXED_NEURONS];
    char ref[N_FIXED_NEURONS];
    FTYPE psc[N_FIXED_NEURONS];

    #pragma disable_loop_pipelining
    for (uint i = 0; i < N_FIXED_NEURONS; i++) {
        vm[i] = gl_vm[i];
        ref[i] = 0;
        psc[i] = 0;
    }

    uint nqueue[64][N_NEURON_QUEUE];
    uint nqueue_n[64];
    #pragma disable_loop_pipelining
    for (uint i = 0; i < 64; i++) {
        nqueue_n[i] = 0;
    }

    // Lanes variable
    float lane[N_PSC_PER_BANK][SYN_ALIGN][N_LANES];


    #pragma disable_loop_pipelining
    for (uint i = 0; i < N_PSC_PER_BANK; i++) {
        for (uint j = 0; j < SYN_ALIGN; j++) {
            #pragma unroll
            for (uint k = 0; k < N_LANES; k++) {
                lane[i][j][k] = 0;
            }
        }
    }

    // Total spike count
    uint n_spikes = 0;

    #pragma disable_loop_pipelining
    for (uint t = 0; t < n_tics; t++) {
        uchar psn_cache[N_FIXED_NEURONS];
        uint psn_row = N_FIXED_NEURONS * t;

        #pragma ivdep
        #pragma unroll 64
        #pragma ii 1
        for (uint i = 0; i < N_FIXED_NEURONS; i++) {
            psn_cache[i] = gl_psn[psn_row + i];
        }

        uint send_buf[64][SYN_ALIGN]
            __attribute__((numbanks(SYN_ALIGN), bankwidth(4)));
        uint n_send[SYN_ALIGN];
        #pragma unroll
        for (uint i = 0; i < SYN_ALIGN; i++) {
            n_send[i] = 0;
        }

        // This ivdep is not true but seem to work...
        #pragma ii 1
        #pragma ivdep
        for (uint i = 0; i < N_PSC_PER_BANK; i++) {
            #pragma ivdep
            #pragma unroll
            for (uint j = 0; j < SYN_ALIGN; j++) {

                // Add from lanes
                float tot = 0;
                #pragma unroll
                for (uint k = 0; k < N_LANES; k++) {
                    tot += lane[i][j][k];
                    lane[i][j][k] = 0;
                }
                uint nid = SYN_ALIGN * i + j;
                psc[nid] += tot;
                FTYPE p = psc[nid];

                FTYPE v = vm[nid];
                FTYPE x = v * p22 + p * p21;

                char cnt = ref[nid];
                char is_spike0 = x >= rel_v_thr;
                char is_send0 = !cnt && is_spike0;

                v = !cnt && !is_spike0 ? x : 0.0f;
                cnt = is_send0 ? t_ref_tics : (cnt != 0 ? cnt - 1 : 0);

                vm[nid] = v;
                ref[nid] = cnt;

                if (is_send0) {
                    uint at = n_send[j];
                    send_buf[at][j] = nid;
                    n_send[j]++;
                }

                FTYPE psn0 = (FTYPE)psn_cache[nid];
                p = p * p11 + psn0 * psn_w;
                psc[nid] = p;
            }
        }

        range arr[N_SYNAPSE_QUEUE];
        uint squeue_n = 0;

        for (uint i = 0; i < 64; i++) {
            uint del = (t - i) & 63;
            if (del >= 1 && del <= 62) {
                uint c = nqueue_n[i];
                for (uint j = 0; j < c; j++) {
                    uint nid = nqueue[i][j];
                    uint addr = 64 * nid + del;
                    uint o0 = gl_idx[addr];
                    uint o1 = gl_idx[addr + 1];
                    if (o0 != o1) {
                        arr[squeue_n++] = (range){o0, o1};
                    }
                }
            }
        }

        #pragma ii 1
        #pragma ivdep safelen(N_LANES)
        for (uint i = 0; i < squeue_n; i++) {
            uint o0 = arr[i].o0;
            uint o1 = arr[i].o1;
            #pragma ii 1
            #pragma ivdep
            for (uint j = o0; j < o1; j ++) {
                #pragma ivdep
                #pragma unroll
                for (uint k = 0; k < SYN_ALIGN; k++) {
                    synapse s = gl_syns[SYN_ALIGN * j + k];
                    lane[s.dst][k][i & (N_LANES - 1)] += s.weight;
                }
            }
        }

        // Queue marked neurons
        uint n = 0;
        uint t_at = t & (MAX_D - 1);

        for (uint i = 0; i < SYN_ALIGN; i++) {
            uint c = n_send[i];
            #pragma ivdep
            for (uint j = 0; j < c; j++) {
                nqueue[t_at][n + j] = send_buf[j][i];
            }
            n += c;
        }
        nqueue_n[t_at] = n;
        n_spikes += n;
    }
    gl_spike_record[0] = n_spikes;
}
