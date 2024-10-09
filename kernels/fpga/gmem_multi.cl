// Copyright (C) 2023-2024 Björn A. Lindqvist <bjourne@gmail.com>
// Flags  : fpga/gmem/multi/{s,d}

#define IS_DEVICE       1
#include "src/shared.h"

#define DEPTH           (N_FIXED_NEURONS / WIDTH)
#define MSG_DONE 0xffffffff

channel uint to_spiker __attribute__((depth(512)));
channel bool to_main __attribute__((depth(1)));

__attribute__((uses_global_work_offset(0)))
__attribute__((max_global_work_dim(0)))
__kernel void
kern_spiker(
    uint n_tics,
    __global const volatile synapse * restrict gl_syns,
    __global const uint volatile * restrict gl_idx,
    __global float volatile * restrict gl_ibuf,
    __global uint volatile * restrict gl_spike_record
) {
    uint n_spikes = 0;

    #pragma disable_loop_pipelining
    for (uint t = 0; t < n_tics; t++) {
        #pragma disable_loop_pipelining
        while (true) {
            uint nid = read_channel_intel(to_spiker);
            if (nid == MSG_DONE) {
                break;
            }
            n_spikes++;
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
        write_channel_intel(to_main, true);
    }
    gl_spike_record[0] = n_spikes;
}

__attribute__((uses_global_work_offset(0)))
__attribute__((max_global_work_dim(0)))
__kernel void
kern_main(
    uint n_tics,
    uchar t_ref_tics,
    FTYPE rel_v_thr,
    FTYPE p11,
    FTYPE p21,
    FTYPE p22,
    FTYPE psn_w,
    __global const volatile FTYPE * restrict gl_vm,
    __global volatile uchar * restrict gl_psn,
    __global float volatile * restrict gl_ibuf
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
    #pragma disable_loop_pipelining
    for (uint t = 0; t < n_tics; t++) {
        uint psn_row = t * N_FIXED_NEURONS;
        uint ibuf_row = (t & I_BUF_ROW_MASK) * N_FIXED_NEURONS;

        #pragma ivdep
        for (uint i = 0; i < DEPTH; i++) {
            uchar spikes[WIDTH];
            #pragma unroll
            #pragma ivdep
            for (uint j = 0; j < WIDTH; j++) {
                uint nid = WIDTH * i + j;
                FTYPE psc0 = psc[i][j];
                FTYPE x = vm[i][j] * p22 + psc0 * p21;

                uchar cnt = ref[i][j];
                bool is_spike = x >= rel_v_thr;
                bool is_send = !cnt && is_spike;

                vm[i][j] = !cnt && !is_spike ? x : 0;
                ref[i][j] = is_send ? t_ref_tics : (cnt ? cnt - 1 : 0);

                spikes[j] = is_send;
                psc[i][j] = psc0 * p11
                    + (FTYPE)gl_psn[psn_row + nid] * psn_w
                    + gl_ibuf[ibuf_row + nid];
                gl_ibuf[ibuf_row + nid] = 0.0;
            }
            #pragma unroll
            #pragma ivdep
            for (uint j = 0; j < WIDTH; j++) {
                if (spikes[j]) {
                    write_channel_intel(to_spiker, WIDTH * i + j);
                }
            }
        }
        write_channel_intel(to_spiker, MSG_DONE);
        read_channel_intel(to_main);
    }
}
