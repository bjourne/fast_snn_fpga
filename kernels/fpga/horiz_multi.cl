// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
// Flags  : fpga/horiz/multi/{s,d}
#define IS_DEVICE       1
#include "src/shared.h"

#define N_SYNAPSE_QUEUE 2048
#define PSC_DEPTH       (N_FIXED_NEURONS / SYN_ALIGN)

typedef struct {
    float v[SYN_ALIGN];
} float_pkt;

typedef struct {
    uint v[N_TURNS + 1];
} index_pkt;

channel float_pkt to_main_frontier __attribute__((depth(512)));
channel uint to_frontier_buf __attribute__((depth(512)));
channel uint to_frontier_cnt __attribute__((depth(256)));

#define N_QUEUE             4096
#define QUEUE_ADDR(e)       ((e) & (N_QUEUE - 1))

__attribute__((uses_global_work_offset(0)))
__attribute__((max_global_work_dim(0)))
__kernel void
kern_frontier(uint n_tics,
              __global const volatile synapse * restrict gl_syns,
              __global volatile uint * restrict gl_spike_record,
              __global const volatile uint * restrict gl_idx) {

    ushort cnt[MAX_D] = {0};
    ushort addrs[MAX_D];
    index_pkt queue[N_QUEUE];
    ushort head = 0;

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

    uint n_spikes = 0;

    #if RECORD_SPIKES == 1
    uint rec_ptr = 1;
    #endif

    #pragma disable_loop_pipelining
    for (uint t = 0; t < n_tics; t++) {

        // Write current ibuf to main
        uint t_front = t & (N_FRONT - 1);
        for (uint i = 0; i < PSC_DEPTH; i++) {
            float_pkt pkt;
            #pragma unroll
            for (uint j = 0; j < SYN_ALIGN; j++) {
                pkt.v[j] = horizon[t_front][i][j];
                horizon[t_front][i][j] = 0;
            }
            write_channel_intel(to_main_frontier, pkt);
        }

        range arr[N_SYNAPSE_QUEUE];
        ushort arr_n = 0;

        #pragma ii 1
        #pragma ivdep
        for (uint i = 0; i < N_TURNS; i++) {
            uint t_from = (t - N_FRONT * i - 1) & (MAX_D - 1);
            uint c = cnt[t_from];
            uint from = addrs[t_from];
            #pragma ii 1
            #pragma ivdep
            for (uint j = 0; j < c; j++) {
                index_pkt pkt = queue[QUEUE_ADDR(from + j)];
                arr[arr_n++] = (range){pkt.v[i], pkt.v[i + 1]};
            }
        }

        for (uint i = 0; i < arr_n; i++) {
            uint o0 = arr[i].o0;
            uint o1 = arr[i].o1;

            #pragma ii 1
            #pragma ivdep
            for (uint j = o0; j < o1; j++) {
                #pragma ivdep
                #pragma unroll
                for (uint k = 0; k < SYN_ALIGN; k++) {
                    synapse s = gl_syns[SYN_ALIGN * j + k];
                    uint dst = s.dst & 0xffffff;
                    uint del = s.dst >> 24;
                    uint row = (del + t) & (N_FRONT - 1);
                    horizon[row][dst][k] += s.weight;
                }
            }
        }

        // Register incoming spikes
        uchar t_at = t & (MAX_D - 1);
        uint cnt_at = read_channel_intel(to_frontier_cnt);
        cnt[t_at] = cnt_at;
        addrs[t_at] = head;

        #if RECORD_SPIKES == 1
        gl_spike_record[rec_ptr++] = cnt_at;
        #endif
        for (uint i = 0; i < cnt_at; i++) {
            uint nid = read_channel_intel(to_frontier_buf);
            #if RECORD_SPIKES == 1
            gl_spike_record[rec_ptr++] = cnt_at;
            #endif
            uint base = N_TURNS * nid;
            uint a = QUEUE_ADDR(head + i);

            #pragma unroll
            for (uint j = 0; j < N_TURNS + 1; j++) {
                queue[a].v[j] = gl_idx[base + j];
            }
        }
        head = QUEUE_ADDR(head + cnt_at);
        n_spikes += cnt_at;
    }
    gl_spike_record[0] = n_spikes;
}

__attribute__((uses_global_work_offset(0)))
__attribute__((max_global_work_dim(0)))
__kernel void
kern_main(
    uint n_tics,
    char t_ref_tics,
    FTYPE rel_v_thr,
    FTYPE p11,
    FTYPE p21,
    FTYPE p22,
    FTYPE psn_w,
    __global const volatile FTYPE * restrict gl_vm,
    __global const volatile uchar * restrict gl_psn
) {
    FTYPE vm[N_FIXED_NEURONS];
    FTYPE psc[N_FIXED_NEURONS];
    char ref[N_FIXED_NEURONS];

    #pragma disable_loop_pipelining
    for (uint i = 0; i < N_FIXED_NEURONS; i++) {
        vm[i] = gl_vm[i];
        ref[i] = 0;
        psc[i] = 0;
    }

    #pragma disable_loop_pipelining
    for (uint t = 0; t < n_tics; t++) {

        // Receive frontier row
        for (uint i = 0; i < PSC_DEPTH; i++) {
            float_pkt pkt = read_channel_intel(to_main_frontier);
            #pragma unroll
            for (uint j = 0; j < SYN_ALIGN; j++) {
                psc[SYN_ALIGN * i + j] += (FTYPE)pkt.v[j];
            }
        }

        uchar psn_cache[N_FIXED_NEURONS];
        #pragma ivdep
        #pragma unroll 64
        #pragma ii 1
        for (uint i = 0; i < N_FIXED_NEURONS; i++) {
            psn_cache[i] = gl_psn[N_FIXED_NEURONS * t + i];
        }

        uint send_buf[64][SYN_ALIGN] __attribute__((numbanks(SYN_ALIGN), bankwidth(4)));
        uint n_send[SYN_ALIGN] = {0};

        #pragma ii 1
        for (uint i = 0; i < PSC_DEPTH; i++) {
            #pragma unroll
            for (uint j = 0; j < SYN_ALIGN; j++) {
                uint nid = SYN_ALIGN * i + j;
                FTYPE p = psc[nid];
                FTYPE x = vm[nid] * p22 + p * p21;

                char cnt = ref[nid];
                char is_spike = x >= rel_v_thr;
                char is_send = !cnt && is_spike;
                vm[nid] = !cnt && !is_spike ? x : 0;
                ref[nid] = is_send ? t_ref_tics : (cnt != 0 ? cnt - 1 : 0);

                FTYPE psn0 = (FTYPE)psn_cache[nid];
                psc[nid] = p * p11 + psn0 * psn_w;

                if (is_send) {
                    uint at = n_send[j];
                    send_buf[at][j] = nid;
                    n_send[j]++;
                }
            }
        }

        uint tot = 0;
        for (uint i = 0; i < SYN_ALIGN; i++) {
            tot += n_send[i];
        }
        write_channel_intel(to_frontier_cnt, tot);
        for (uint i = 0; i < SYN_ALIGN; i++) {
            for (uint j = 0; j < n_send[i]; j++) {
                write_channel_intel(to_frontier_buf, send_buf[j][i]);
            }
        }
    }
}
