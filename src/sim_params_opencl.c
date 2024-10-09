// Copyright (C) 2023 Björn Lindqvist <bjourne@gmail.com>
#define IS_HOST
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#define CL_TARGET_OPENCL_VERSION 300
#include "common.h"
#include "opencl.h"

#include "sim_params.h"
#include "utils.h"
#include "var_vector.h"

#include "shared.h"

#define FMT_COMPILE_OPTS                                        \
    "-cl-std=CL2.0 "                                            \
    "-D USE_DOUBLES=%d "                                        \
    "-D RECORD_SPIKES=%d "                                      \
    "-D WIDTH=%d "                                              \
    "-D SYN_UNROLL=%d "                                         \
    "-D SYN_ALIGN=%d "                                          \
    "-D N_FRONT=%d "                                            \
    "-D N_LANES=%d"

typedef enum {
    BUF_VM = 0,
    BUF_DC_PRE,
    BUF_REF,
    BUF_PSC,
    BUF_PSN,
    BUF_SYNAPSE,
    BUF_INDEX,
    BUF_IBUF,
    BUF_SPIKE_RECORD,
} buffer_type;

char *JIT_MULTI_KERNELS[] = {
    "kern_main", "kern_frontier" // "kern_queue", "kern_coll", "kern_frontier"
};
char *HORIZ_MULTI_KERNELS[] = {
    "kern_main", "kern_frontier"
};
char *GMEM_MULTI_KERNELS[] = {"kern_spiker", "kern_main"};
char *DEFAULT_KERNEL[] = {"netsim"};

sim_result
sim_config_run_on_opencl(sim_config *me) {
    assert(me->n_sim_tics > 0);

    bool is_double = me->double_prec;
    size_t n_double = sizeof(double);
    size_t n_float = sizeof(float);
    size_t n_cl_uchar = sizeof(cl_uchar);

    size_t n_cl_uint = sizeof(cl_uint);

    size_t n_synapse = sizeof(synapse);
    size_t ftype_size = is_double ? n_double : n_float;

    uint32_t n_kernels = 0;
    char **kernel_names = NULL;
    if (me->multi_kernel) {
        if (me->algo == SIM_ALGO_JIT) {
            n_kernels = ARRAY_SIZE(JIT_MULTI_KERNELS);
            kernel_names = JIT_MULTI_KERNELS;
        } else if (me->algo == SIM_ALGO_HORIZON) {
            n_kernels = ARRAY_SIZE(HORIZ_MULTI_KERNELS);
            kernel_names = HORIZ_MULTI_KERNELS;
        }
        else {
            n_kernels = ARRAY_SIZE(GMEM_MULTI_KERNELS);
            kernel_names = GMEM_MULTI_KERNELS;
        }
    } else {
        n_kernels = 1;
        kernel_names = DEFAULT_KERNEL;
    }

    uint32_t n_orig_neurons = me->index->n_neurons;
    uint32_t n_padded_neurons = n_orig_neurons;
    if (me->device_type == OCL_DEVICE_TYPE_CPU) {
        n_padded_neurons = me->rowstride;
    } else if (me->device_type == OCL_DEVICE_TYPE_FPGA) {
        n_padded_neurons = me->rowstride;
        assert(n_padded_neurons == N_FIXED_NEURONS);
        if (me->algo == SIM_ALGO_HORIZON) {
            synapse_index *new = synapse_index_frontier_index(
                me->index, n_padded_neurons, N_FRONT, SYN_ALIGN);
            synapse_index_free(me->index);
            me->index = new;
        } else if (me->algo == SIM_ALGO_JIT) {
            synapse_index *new = synapse_index_align_delay_classes(
                me->index, n_padded_neurons, SYN_ALIGN);
            synapse_index_free(me->index);
            me->index = new;
        } else {
            synapse_index *new = synapse_index_merge_delays(me->index);
            synapse_index_free(me->index);
            me->index = new;
        }
    } else {
        assert(false);
    }

    // Retrieve count after padding index
    uint32_t n_synapses = me->index->n_synapses;

    ocl_ctx *ctx = ocl_ctx_init(me->platform_idx,
                                me->device_idx, true);

    OCL_CHECK_ERR(ctx->err);
    for (uint32_t i = 0; i < n_kernels; i++) {
        OCL_CHECK_ERR(ocl_ctx_add_queue(ctx));
    }
    char opts[512];
    sprintf(opts, FMT_COMPILE_OPTS,
            me->double_prec,
            RECORD_SPIKES,
            WIDTH,
            SYN_UNROLL,
            SYN_ALIGN,
            N_FRONT,
            N_LANES);
    OCL_CHECK_ERR(ocl_ctx_load_kernels(
                      ctx,
                      me->program_path, opts,
                      n_kernels, kernel_names));

    // Size of Poisson buffer must not be too large...
    size_t psn_rows = MIN(me->max_n_tics, me->n_sim_tics);

    // Add all buffers
    ocl_ctx_buf bufs[] = {
        // Membranes
        {0, n_padded_neurons * ftype_size, CL_MEM_READ_WRITE},

        // DC premultiplied
        {0, n_padded_neurons * ftype_size, CL_MEM_READ_ONLY},

        // Refractory
        {0, n_padded_neurons * n_cl_uchar, CL_MEM_READ_WRITE},

        // PSC
        {0, n_padded_neurons * ftype_size, CL_MEM_READ_WRITE},

        // Poisson
        {0, psn_rows * me->rowstride * n_cl_uchar, CL_MEM_READ_ONLY},

        // Synapses
        {0, n_synapses * n_synapse, CL_MEM_READ_ONLY},

        // Index. Note that we include one extra element!

        // Synapse index.
        {0, me->index->width * n_orig_neurons * n_cl_uint + n_cl_uint,
         CL_MEM_READ_ONLY},

        // Spike buffer, always floats.
        {0, me->rowstride * I_BUF_ROWS * n_float, CL_MEM_READ_WRITE},

        // Spike record
        {0, n_cl_uint * SPIKE_RECORD_ELEMENTS, CL_MEM_READ_WRITE}
    };

    for (size_t i = 0; i < ARRAY_SIZE(bufs); i++) {
        OCL_CHECK_ERR(ocl_ctx_add_buffer(ctx, bufs[i]));
    }

    // Write index
    OCL_CHECK_ERR(ocl_ctx_write_buffer(
                      ctx, 0, BUF_INDEX,
                      me->index->index));

    // Write synapses
    OCL_CHECK_ERR(ocl_ctx_write_buffer(
                      ctx, 0, BUF_SYNAPSE,
                      me->index->data));

    // Write initial volages relative to reset.
    void *v_m_init = calloc(n_padded_neurons, ftype_size);
    for (uint32_t i = 0; i < n_orig_neurons; i++) {
        double dbl_val = me->v_m_init[i] - me->v_r;
        mprec_write_double(v_m_init, i, dbl_val, is_double);
    }
    OCL_CHECK_ERR(ocl_ctx_write_buffer(
                      ctx, 0, BUF_VM,
                      v_m_init));
    free(v_m_init);

    void *dc_per_neuron = calloc(n_padded_neurons, ftype_size);
    uint32_t at = 0;
    for (uint32_t i = 0; i < me->n_layers; i++) {
        double dc = me->dc_per_layer[i];
        for (uint32_t j = 0; j < me->cnt_per_layer[i]; j++, at++) {
            mprec_write_double(dc_per_neuron, at, dc * me->p20, is_double);
        }
    }
    OCL_CHECK_ERR(ocl_ctx_write_buffer(
                      ctx, 0, BUF_DC_PRE,
                      dc_per_neuron));
    free(dc_per_neuron);

    OCL_CHECK_ERR(ocl_ctx_write_buffer(
                      ctx, 0, BUF_PSN,
                      me->psn));

    float zero = 0.0;
    OCL_CHECK_ERR(ocl_ctx_fill_buffer(
                      ctx, 0, BUF_IBUF,
                      &zero, n_float));

    size_t n_cl_mem = sizeof(cl_mem);
    size_t n_cl_char = sizeof(cl_char);
    assert(n_cl_char);

    // Buffer arguments to kernels
    void *buf_vm = &ctx->buffers[BUF_VM];
    void *buf_dc_pre = &ctx->buffers[BUF_DC_PRE];
    void *buf_ref = &ctx->buffers[BUF_REF];
    void *buf_psc = &ctx->buffers[BUF_PSC];
    void *buf_psn = &ctx->buffers[BUF_PSN];
    void *buf_synapse = &ctx->buffers[BUF_SYNAPSE];
    void *buf_index = &ctx->buffers[BUF_INDEX];
    void *buf_ibuf = &ctx->buffers[BUF_IBUF].ptr;
    void *buf_spike_record = &ctx->buffers[BUF_SPIKE_RECORD];

    // Double/single arguments
    double rel_v_thr = me->v_thr - me->v_r;

    float s_p11 = me->p11;
    float s_p21 = me->p21;
    float s_p22 = me->p22;
    float s_psn_w = me->psn_w;
    float s_rel_v_thr = rel_v_thr;

    void *a_p11 = &me->p11;
    void *a_p21 = &me->p21;
    void *a_p22 = &me->p22;
    void *a_psn_w = &me->psn_w;
    void *a_rel_v_thr = &rel_v_thr;
    if (!is_double) {
        a_p11 = &s_p11;
        a_p21 = &s_p21;
        a_p22 = &s_p22;
        a_psn_w = &s_psn_w;
        a_rel_v_thr = &s_rel_v_thr;
    }

    // Integer arguments
    void *a_n_tics = &me->n_sim_tics;
    void *a_t_ref_tics = &me->t_ref_tics;
    void *a_n_padded_neurons = &n_padded_neurons;
    void *a_n_synapses = &n_synapses;
    void *a_rowstride = &me->rowstride;


    if (me->device_type == OCL_DEVICE_TYPE_CPU) {
        OCL_CHECK_ERR(
            ocl_set_kernel_arguments(
                ctx->kernels[0], 19,
                n_cl_uint, a_n_padded_neurons,
                n_cl_uint, a_n_synapses,
                n_cl_uint, a_n_tics,
                n_cl_uchar, a_t_ref_tics,
                n_cl_uint, a_rowstride,
                ftype_size, a_rel_v_thr,
                ftype_size, a_p11,
                ftype_size, a_p21,
                ftype_size, a_p22,
                ftype_size, a_psn_w,
                n_cl_mem, buf_vm,
                n_cl_mem, buf_ref,
                n_cl_mem, buf_psc,
                n_cl_mem, buf_psn,
                n_cl_mem, buf_synapse,
                n_cl_mem, buf_spike_record,
                n_cl_mem, buf_dc_pre,
                n_cl_mem, buf_index,
                n_cl_mem, buf_ibuf));
    } else if (me->algo == SIM_ALGO_JIT) {
        if (me->multi_kernel) {
            // kern_main
            OCL_CHECK_ERR(
                ocl_set_kernel_arguments(
                    ctx->kernels[0], 9,
                    n_cl_uint, a_n_tics,
                    n_cl_uchar, a_t_ref_tics,
                    ftype_size, a_rel_v_thr,
                    ftype_size, a_p11,
                    ftype_size, a_p21,
                    ftype_size, a_p22,
                    ftype_size, a_psn_w,
                    n_cl_mem, buf_vm,
                    n_cl_mem, buf_psn));
            // kern_frontier
            OCL_CHECK_ERR(
                ocl_set_kernel_arguments(
                    ctx->kernels[1], 4,
                    n_cl_uint, a_n_tics,
                    n_cl_mem, buf_synapse,
                    n_cl_mem, buf_spike_record,
                    n_cl_mem, buf_index));
        } else {
            OCL_CHECK_ERR(
                ocl_set_kernel_arguments(
                    ctx->kernels[0], 12,
                    n_cl_uint, a_n_tics,
                    n_cl_uchar, a_t_ref_tics,
                    ftype_size, a_rel_v_thr,
                    ftype_size, a_p11,
                    ftype_size, a_p21,
                    ftype_size, a_p22,
                    ftype_size, a_psn_w,
                    n_cl_mem, buf_vm,
                    n_cl_mem, buf_psn,
                    n_cl_mem, buf_synapse,
                    n_cl_mem, buf_spike_record,
                    n_cl_mem, buf_index));
        }
    } else if (me->algo == SIM_ALGO_HORIZON) {
        if (me->multi_kernel) {
            // kern_main
            OCL_CHECK_ERR(
                ocl_set_kernel_arguments(
                    ctx->kernels[0], 9,
                    n_cl_uint, a_n_tics,
                    n_cl_uchar, a_t_ref_tics,
                    ftype_size, a_rel_v_thr,
                    ftype_size, a_p11,
                    ftype_size, a_p21,
                    ftype_size, a_p22,
                    ftype_size, a_psn_w,
                    n_cl_mem, buf_vm,
                    n_cl_mem, buf_psn));
            // kern_frontier
            OCL_CHECK_ERR(
                ocl_set_kernel_arguments(
                    ctx->kernels[1], 4,
                    n_cl_uint, a_n_tics,
                    n_cl_mem, buf_synapse,
                    n_cl_mem, buf_spike_record,
                    n_cl_mem, buf_index));
        } else {
            OCL_CHECK_ERR(
                ocl_set_kernel_arguments(
                    ctx->kernels[0], 12,
                    n_cl_uint, a_n_tics,
                    n_cl_uchar, a_t_ref_tics,
                    ftype_size, a_rel_v_thr,
                    ftype_size, a_p11,
                    ftype_size, a_p21,
                    ftype_size, a_p22,
                    ftype_size, a_psn_w,
                    n_cl_mem, buf_vm,
                    n_cl_mem, buf_psn,
                    n_cl_mem, buf_synapse,
                    n_cl_mem, buf_spike_record,
                    n_cl_mem, buf_index));
        }
    } else if (me->algo == SIM_ALGO_GMEM) {
        if (me->multi_kernel) {
            // kern_spiker
            OCL_CHECK_ERR(
                ocl_set_kernel_arguments(
                    ctx->kernels[0], 5,
                    n_cl_uint, a_n_tics,
                    n_cl_mem, buf_synapse,
                    n_cl_mem, buf_index,
                    n_cl_mem, buf_ibuf,
                    n_cl_mem, buf_spike_record));
            // kern_main
            OCL_CHECK_ERR(
                ocl_set_kernel_arguments(
                    ctx->kernels[1], 10,
                    n_cl_uint, a_n_tics,
                    n_cl_uchar, a_t_ref_tics,
                    ftype_size, a_rel_v_thr,
                    ftype_size, a_p11,
                    ftype_size, a_p21,
                    ftype_size, a_p22,
                    ftype_size, a_psn_w,
                    n_cl_mem, buf_vm,
                    n_cl_mem, buf_psn,
                    n_cl_mem, buf_ibuf));
        } else {
            OCL_CHECK_ERR(
                ocl_set_kernel_arguments(
                    ctx->kernels[0], 13,
                    n_cl_uint, a_n_tics,
                    n_cl_uchar, a_t_ref_tics,
                    ftype_size, a_rel_v_thr,
                    ftype_size, a_p11,
                    ftype_size, a_p21,
                    ftype_size, a_p22,
                    ftype_size, a_psn_w,
                    n_cl_mem, buf_vm,
                    n_cl_mem, buf_psn,
                    n_cl_mem, buf_synapse,
                    n_cl_mem, buf_index,
                    n_cl_mem, buf_ibuf,
                    n_cl_mem, buf_spike_record));
        }
    } else {
        assert(false);
    }
    for (uint32_t i = 0; i < n_kernels; i++) {
        cl_command_queue q = ctx->queues[i];
        assert(clFlush(q) == CL_SUCCESS);
        assert(clFinish(q) == CL_SUCCESS);
    }

    uint64_t start = nano_count();
    size_t wg_size[] = {N_GPU_WORK_ITEMS};
    for (uint32_t i = 0; i < n_kernels; i++) {
        OCL_CHECK_ERR(
            clEnqueueNDRangeKernel(
                ctx->queues[i], ctx->kernels[i],
                1, NULL, wg_size, wg_size,
                0, NULL, NULL));
    }
    for (uint32_t i = 0; i < n_kernels; i++) {
        clFinish(ctx->queues[i]);
    }
    uint64_t delta = nano_count() - start;
    uint32_t *rec = malloc(ctx->buffers[BUF_SPIKE_RECORD].n_bytes);
    OCL_CHECK_ERR(ocl_ctx_read_buffer(
                      ctx, 0, BUF_SPIKE_RECORD, rec));
    ocl_ctx_free(ctx);
    npy_arr *arr = npy_init('u', 4, 1, (int[]){SPIKE_RECORD_ELEMENTS}, rec, false);
    return (sim_result){rec[0], delta, arr};
}
