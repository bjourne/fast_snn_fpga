// Copyright (C) 2023-2024 Björn A. Lindqvist <bjourne@gmail.com>
//
// 8 is best on my home laptop.
#define VECTOR_WIDTH 8
#define IS_DEVICE   1
#include "src/shared.h"

#define SEND_BUF_SIZE   128

__kernel void
netsim(
    // Must be divisible by VECTOR_WIDTH
    uint n_neurons,
    uint n_synapses,
    uint n_tics,
    uchar t_ref_tics,
    uint rowstride,
    FTYPE rel_v_thr,
    FTYPE p11,
    FTYPE p21,
    FTYPE p22,
    FTYPE psn_w,
    __global FTYPE * restrict gl_vm,

    // Should be zeroed
    __global uchar * restrict ref,

    // Should be zeroed
    __global FTYPE * restrict psc,
    __global uchar * restrict psn,
    __global synapse * restrict synapses,
    __global uint * restrict ret,

    // DC per neuron
    __global FTYPE * restrict dc,
    __global uint * restrict idx_src_del,
    __global float * restrict ibuf
) {
    __local uint q[64][120];
    __local uint q_n[64];
    for (uint i = 0; i < 64; i++) {
        q_n[i] = 0;
    }
    uint n_spikes = 0;
    for (uint t = 0; t < n_tics; t++) {

        uint psn_row = t * rowstride;
        uint send_buf[SEND_BUF_SIZE];
        uint n_to_send = 0;

        for (uint i = 0; i < n_neurons; i += VECTOR_WIDTH) {
            VFTYPE psc0 = VLOAD_AT(i, psc);
            VFTYPE vm0 = VLOAD_AT(i, gl_vm);
            VFTYPE dc0 = VLOAD_AT(i, dc);

            VFTYPE x = vm0 * p22 + dc0 + psc0 * p21;

            VITYPE cnt = VLOAD_AT_AS_ITYPE(i, ref);
            VITYPE is_spike = x >= rel_v_thr;
            VITYPE is_send = !cnt && is_spike;

            vm0 = !cnt && !is_spike ? x : 0.0f;
            cnt = is_send ? t_ref_tics : (cnt != 0 ? cnt - 1 : 0);

            VSTORE_AT(vm0, i, gl_vm);
            VSTORE_AT_AS_UCHAR(cnt, i, ref);
            if (any(is_send)) {
                for (uint j = 0; j < VECTOR_WIDTH; j++) {
                    if (is_send[j]) {
                        send_buf[n_to_send] = i + j;
                        n_to_send++;
                    }
                }
            }
            VFTYPE psn0 = VLOAD_AT_AS_FTYPE(psn_row + i, psn);
            psc0 = psc0 * p11 + psn0 * psn_w;
            VSTORE_AT(psc0, i, psc);
        }
        for (uint i = 0; i < 64; i++) {
            uint del = (t - i) & 63;
            if (del >= 1 && del <= 62) {
                uint c_n = q_n[i];
                for (uint j = 0; j < c_n; j++) {
                    uint src = q[i][j];
                    uint a = 64 * src + del;
                    uint o0 = idx_src_del[a];
                    uint o1 = idx_src_del[a + 1];

                    prefetch((const __global ulong *)&synapses[o0], o1 - o0);
                    for (uint o = o0; o < o1; o++) {
                        synapse s = synapses[o];
                        psc[s.dst] += s.weight;
                    }
                }
            }
        }
        uint at = t & 63;
        uint ofs = work_group_scan_exclusive_add(n_to_send);
        q_n[at] = work_group_reduce_add(n_to_send);
        for (uint i = 0; i < n_to_send; i++) {
            q[at][ofs + i] = send_buf[i];
        }
        n_spikes += n_to_send;
    }
    ret[0] = work_group_reduce_add(n_spikes);
}
