// Copyright (C) 2023-2024 Björn A. Lindqvist <bjourne@gmail.com>
// Flag   : fpga/gmem/mono/{s,d}
#define IS_DEVICE       1
#include "src/shared.h"

__attribute__((uses_global_work_offset(0)))
__attribute__((max_global_work_dim(0)))
__kernel void
netsim(
    uint n_tics,
    uchar t_ref_tics,
    FTYPE rel_v_thr,
    FTYPE p11,
    FTYPE p21,
    FTYPE p22,
    FTYPE psn_w,
    __global const volatile FTYPE * restrict gl_vm,
    __global volatile uchar * restrict gl_psn,
    __global const volatile synapse * restrict gl_syns,
    __global const uint volatile * restrict gl_idx,
    __global float volatile * restrict gl_ibuf,
    __global uint volatile * restrict gl_spike_record
) {

    // Initialize data
    FTYPE vm[DEPTH][WIDTH] __attribute__((numbanks(WIDTH), bankwidth(FTYPE_SIZE)));
    FTYPE psc[DEPTH][WIDTH] __attribute__((numbanks(WIDTH), bankwidth(FTYPE_SIZE)));
    uchar ref[DEPTH][WIDTH] __attribute__((numbanks(WIDTH), bankwidth(1)));

    #pragma disable_loop_pipelining
    for (uint i = 0; i < DEPTH; i++) {
        #pragma unroll
        for (uint j = 0; j < WIDTH; j++) {
            vm[i][j] = gl_vm[WIDTH * i + j];
            ref[i][j] = 0;
            psc[i][j] = 0;
        }
    }
#if RECORD_SPIKES == 1
    uint rec_ptr = 1;
#endif

    // Total spike count
    uint n_spikes = 0;
    uint psn_t = 0;

#pragma disable_loop_pipelining
    for (uint t = 0; t < n_tics; t++) {
        uchar psn_cache[N_FIXED_NEURONS];
        #pragma ivdep
        #pragma unroll 64
        #pragma ii 1
        for (uint i = 0; i < N_FIXED_NEURONS; i++) {
            psn_cache[i] = gl_psn[N_FIXED_NEURONS * psn_t + i];
        }
        uint ibuf_row = (t & I_BUF_ROW_MASK) * N_FIXED_NEURONS;
        uint send_buf[MAX_SPIKES_PER_TIC][WIDTH]
            __attribute__((numbanks(WIDTH), bankwidth(4)));
        ushort n_send[WIDTH];

        #pragma unroll
        for (uint i = 0; i < WIDTH; i++) {
            n_send[i] = 0;
        }
        for (uint i = 0; i < DEPTH; i++) {
            #pragma unroll
            for (uint j = 0; j < WIDTH; j++) {
                FTYPE psc0 = psc[i][j];
                FTYPE x = vm[i][j] * p22 + psc0 * p21;

                uchar cnt = ref[i][j];
                bool is_spike = x >= rel_v_thr;
                bool is_send = !cnt && is_spike;

                uint nid = WIDTH * i + j;
                vm[i][j] = !cnt && !is_spike ? x : 0;
                ref[i][j] = is_send ? t_ref_tics : (cnt ? cnt - 1 : 0);

                if (is_send) {
                    ushort at = n_send[j];
                    send_buf[at][j] = nid;
                    n_send[j]++;
                }
                FTYPE psn0 = (FTYPE)psn_cache[nid];
                psc[i][j] = psc0 * p11 + psn0 * psn_w
                    + gl_ibuf[ibuf_row + nid];
                gl_ibuf[ibuf_row + nid] = 0;
            }
        }

#if RECORD_SPIKES == 1
        uint n_tic_spikes = 0;
        for (uint i = 0; i < WIDTH; i++) {
            n_tic_spikes += n_send[i];
        }
        gl_spike_record[rec_ptr++] = n_tic_spikes;
        for (uint i = 0; i < WIDTH; i++) {
            for (uint j = 0; j < n_send[i]; j++) {
                gl_spike_record[rec_ptr + j] = send_buf[j][i];
            }
            rec_ptr += n_send[i];
        }
#endif

        for (uint i = 0; i < WIDTH; i++) {
            n_spikes += n_send[i];
            for (uint j = 0; j < n_send[i]; j++) {
                uint nid = send_buf[j][i];
                uint k0 = gl_idx[nid];
                uint k1 = gl_idx[nid + 1];
#pragma ii 1
#pragma ivdep
#pragma unroll SYN_UNROLL
                for (uint k = k0; k < k1; k++) {
                    synapse s = gl_syns[k];
                    uint dst = s.dst >> 8;
                    uint delay = s.dst & 0xff;
                    uint row = (delay + t) & I_BUF_ROW_MASK;
                    uint addr = N_FIXED_NEURONS * row + dst;
                    gl_ibuf[addr] += s.weight;
                }
            }
        }
        psn_t++;
        if (psn_t == MAX_PSN_T) {
            psn_t = 0;
        }
    }
    gl_spike_record[0] = n_spikes;
}
