// Copyright (C) 2023-2024 Björn A. Lindqvist <bjourne@gmail.com>
#define IS_DEVICE 1
#define VECTOR_WIDTH 8
#include "src/config.h"

#if defined(cl_khr_fp64)
#define DOUBLE_SUPPORT_AVAILABLE
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#else
#error "Device doesn't support doubles!"
#endif

typedef struct {
    uint c0, c1, c2;
} ranges;

inline ranges
slice_work(uint n_items, uint n_workers, uint id, uint width) {
    ranges r;
    // Should work as long as n_items > 0.
    uint chunk = (n_items / n_workers + 1 + width) / width * width;
    r.c0 = id * chunk;
    if (r.c0 < n_items) {
        uint n = min(r.c0 + chunk, n_items) - r.c0;
        r.c1 = r.c0 + n / width * width;
        r.c2 = r.c1 + n % width;
    } else {
        r.c1 = 0;
        r.c2 = 0;
    }
    return r;
}

__kernel void
netsim(
    // Must be divisible by VECTOR_WIDTH
    uint n_neurons,
    uint n_synapses,
    uint n_tics,
    uchar t_ref_tics,
    uint rowstride,
    double rel_v_thr,
    double p11,
    double p21,
    double p22,
    double psn_w,
    __global double * restrict vm,

    // Should be zeroed
    __global uchar * restrict ref,

    // Should be zeroed
    __global double * restrict psc,

    __global uchar * restrict psn,
    __global synapse * restrict synapses,
    __global uint * restrict ret,
    // DC per neuron
    __global double * restrict dc,
    __global uint * restrict idx_src_del,
    __global float * restrict ibuf
) {
    uint gs = get_local_size(0);
    uint id = get_local_id(0);
    ranges r = slice_work(n_neurons, gs, id, VECTOR_WIDTH);
    uint n_spikes = 0;

    __local uint send_buf[N_GPU_WORK_ITEMS][SEND_BUF_SIZE];
    __local uint n_to_send[N_GPU_WORK_ITEMS];
    for (uint t = 0; t < n_tics; t++) {
        LOCAL_BARRIER;
        GLOBAL_BARRIER;
        uint psn_row = t * rowstride;
        uint ibuf_row = (t & I_BUF_ROW_MASK) * rowstride;
        n_to_send[id] = 0;
        for (uint i = r.c0; i < r.c1; i += VECTOR_WIDTH) {
            vdouble psc0 = VLOAD_AT(i, psc);
            vdouble vm0 = VLOAD_AT(i, vm);
            vdouble dc0 = VLOAD_AT(i, dc);

            vdouble x = vm0 * p22
                + dc0
                + psc0 * p21;

            vlong cnt = VLOAD_AT_AS_LONG(i, ref);

            vlong is_spike = x >= rel_v_thr;
            vlong is_send = !cnt && is_spike;

            vm0 = !cnt && !is_spike ? x : 0.0f;
            cnt = is_send ? t_ref_tics : (cnt != 0 ? cnt - 1 : 0);

            VSTORE_AT(vm0, i, vm);
            VSTORE_AT_AS_UCHAR(cnt, i, ref);
            if (any(is_send)) {
                for (uint j = 0; j < VECTOR_WIDTH; j++) {
                    if (is_send[j]) {
                        uint at = n_to_send[id];
                        send_buf[id][at] = i + j;
                        n_to_send[id]++;
                    }
                }
            }
            vdouble psn0 = VLOAD_AT_AS_DOUBLE(psn_row + i, psn);
            vdouble ibuf0 = VLOAD_AT_AS_DOUBLE(ibuf_row + i, ibuf);
            psc0 = psc0 * p11 + psn0 * psn_w + ibuf0;
            VSTORE_AT(psc0, i, psc);
            VSTORE_AT(0.0, ibuf_row + i, ibuf);
        }
        for (uint i = r.c1; i < r.c2; i++) {
            double psc0 = psc[i];
            double x = vm[i] * p22 + dc[i] + psc0 * p21;

            uchar cnt = ref[i];
            bool is_spike = x >= rel_v_thr;
            bool is_send = !cnt && is_spike;

            vm[i] = !cnt && !is_spike ? x : 0;
            ref[i] = is_send ? t_ref_tics : (cnt ? cnt - 1 : 0);

            if (is_send) {
                uint at = n_to_send[id];
                send_buf[id][at] = i;
                n_to_send[id]++;
            }
            psc[i] = psc0 * p11
                + psn[psn_row + i] * psn_w
                + ibuf[ibuf_row + i];
            ibuf[ibuf_row + i] = 0.0;
        }
        LOCAL_BARRIER;
        for (uint i = 0; i < N_GPU_WORK_ITEMS; i++) {
            for (uint j = 0; j < n_to_send[i]; j++) {
                uint src = send_buf[i][j];
                uint k0 = idx_src_del[2 * src];
                uint k1 = idx_src_del[2 * src + 1];
                for (uint k = k0 + id; k < k1; k += N_GPU_WORK_ITEMS) {
                    synapse s = synapses[k];
                    if (s.dst != 0xffffffff) {
                        uint dst = s.dst >> 8;
                        uint del = s.dst & 0xff;
                        uint row = (del + t) & I_BUF_ROW_MASK;
                        float w = s.weight;
                        ibuf[rowstride * row + dst] += w;
                    }
                }
            }
        }
        n_spikes += n_to_send[id];
    }
    ret[0] = work_group_reduce_add(n_spikes);
}
