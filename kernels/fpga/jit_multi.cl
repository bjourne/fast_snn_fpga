// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
//
// Flags            : jit/multi/{s,d}
#define IS_DEVICE
#include "src/shared.h"

#define N_SYNAPSE_QUEUE 2048
#define PSC_DEPTH  (N_FIXED_NEURONS / SYN_ALIGN)

typedef struct {
    float v[SYN_ALIGN];
} psc_pkt;

channel psc_pkt to_main __attribute__((depth(512)));
channel uint to_frontier __attribute__((depth(128)));

#define N_QUEUE             4096
#define QUEUE_ADDR(e)       ((e) & (N_QUEUE - 1))

static inline void
write_to_lanes(float lane[PSC_DEPTH][SYN_ALIGN][N_LANES],
               range arr[], ushort arr_n,
               __global const volatile synapse * restrict gl_syns) {

    ushort rem = arr_n & (N_LANES - 1);
    rem = rem ? N_LANES - rem : 0;
    for (ushort i = 0; i < rem; i++) {
        arr[arr_n++] = (range){0, 0};
    }
    for (ushort i = 0; i < arr_n; i += N_LANES) {
        #pragma ivdep
        for (uchar j = 0; j < N_LANES; j++) {
            uint o0 = arr[i + j].o0;
            uint o1 = arr[i + j].o1;
            #pragma ii 1
            #pragma ivdep
            for (uint k = o0; k < o1; k++) {
                #pragma ivdep
                #pragma unroll
                for (uint l = 0; l < SYN_ALIGN; l++) {
                    synapse s = gl_syns[SYN_ALIGN * k + l];
                    lane[s.dst][l][j] += s.weight;
                }
            }
        }
    }
}

__attribute__((uses_global_work_offset(0)))
__attribute__((max_global_work_dim(0)))
__kernel void
kern_frontier(uint n_tics,
              __global const volatile synapse * restrict gl_syns,
              __global volatile uint * restrict gl_spike_record,
              __global const volatile uint * restrict gl_idx) {

    float lane[PSC_DEPTH][SYN_ALIGN][N_LANES];
    ushort cnt[MAX_D] = {0};
    ushort addr[MAX_D];
    uint queue[N_QUEUE];
    uint head = 0;

    #pragma disable_loop_pipelining
    for (uint i = 0; i < PSC_DEPTH; i++) {
        for (uint j = 0; j < SYN_ALIGN; j++) {
            for (uint k = 0; k < N_LANES; k++) {
                lane[i][j][k] = 0;
            }
        }
    }
    uint n_spikes = 0;

    #pragma disable_loop_pipelining
    for (uint t = 0; t < n_tics; t++) {
        // Send current psc values
        #pragma ii 1
        for (uint i = 0; i < PSC_DEPTH; i++) {
            psc_pkt pkt;

            #pragma unroll
            for (uint j = 0; j < SYN_ALIGN; j++) {
                float tot = 0;
                #pragma unroll
                for (uint k = 0; k < N_LANES; k++) {
                    tot += lane[i][j][k];
                    lane[i][j][k] = 0;
                }
                pkt.v[j] = tot;
            }
            write_channel_intel(to_main, pkt);
        }

        // Update psc according to queue
        range arr[N_SYNAPSE_QUEUE];
        ushort arr_n = 0;
        for (uint i = 0; i < MAX_D; i++) {
            uint del = (t - i) & (MAX_D - 1);
            if (del >= 1 && del <= 62) {
                ushort c = cnt[i];
                ushort from = addr[i];
                for (ushort j = 0; j < c; j++) {
                    uint a = QUEUE_ADDR(from + j);
                    uint nid = queue[a];
                    uint o0 = gl_idx[MAX_D * nid + del];
                    uint o1 = gl_idx[MAX_D * nid + del + 1];
                    if (o0 != o1) {
                        arr[arr_n++] = (range){o0, o1};
                    }
                }
            }
        }
        write_to_lanes(lane, arr, arr_n, gl_syns);

        // Register spiking neurons
        uchar t_at = t & (MAX_D - 1);
        uint cnt_at = read_channel_intel(to_frontier);

        cnt[t_at] = cnt_at;
        addr[t_at] = head;

        #pragma ii 1
        for (uint i = 0; i < cnt_at; i++) {
            uint nid = read_channel_intel(to_frontier);
            uint a = QUEUE_ADDR(head + i);
            queue[a] = nid;
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
    // Initialize data
    FTYPE vm[N_FIXED_NEURONS];
    char ref[N_FIXED_NEURONS];
    FTYPE psc[N_FIXED_NEURONS];
    #pragma disable_loop_pipelining
    for (uint i = 0; i < N_FIXED_NEURONS; i++) {
        psc[i] = 0;
        vm[i] = gl_vm[i];
        ref[i] = 0;
    }

    #pragma disable_loop_pipelining
    for (uint t = 0; t < n_tics; t++) {
        for (uint i = 0; i < PSC_DEPTH; i++) {
            psc_pkt pkt = read_channel_intel(to_main);
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
                FTYPE v = vm[nid];
                FTYPE x = v * p22 + p * p21;

                char cnt = ref[nid];
                char is_spike = x >= rel_v_thr;
                char is_send = !cnt && is_spike;

                v = !cnt && !is_spike ? x : 0;
                cnt = is_send ? t_ref_tics : (cnt != 0 ? cnt - 1 : 0);

                vm[nid] = v;
                ref[nid] = cnt;

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
        write_channel_intel(to_frontier, tot);
        for (uint i = 0; i < SYN_ALIGN; i++) {
            for (uint j = 0; j < n_send[i]; j++) {
                write_channel_intel(to_frontier, send_buf[j][i]);
            }
        }
    }
}
