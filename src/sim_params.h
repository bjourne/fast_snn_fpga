// Copyright (C) 2023-2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef SIM_PARAMS_H
#define SIM_PARAMS_H

#include <stdint.h>
#include "npy.h"
#include "synapse_index.h"

// Right now kernel parameters are hardcoded like this. Should be
// fixed.
#define RECORD_SPIKES   1
#define WIDTH           8
#define SYN_UNROLL      2
#define SYN_ALIGN       8
#define N_FRONT         16
#define N_LANES         16

typedef enum {
    OCL_DEVICE_TYPE_CPU = 0,
    OCL_DEVICE_TYPE_GPU,
    OCL_DEVICE_TYPE_FPGA,
} ocl_device_type;

typedef enum {
    SIM_ALGO_GMEM = 0,
    SIM_ALGO_JIT,
    SIM_ALGO_HORIZON
} sim_algo;

typedef struct {
    // Number of tics to simulate
    uint32_t n_sim_tics;

    // Directory where network files are stored
    char *network_dir;

    // Running on cpu or not
    bool is_cpu;

    // OpenCL program
    char *program_path;

    // OpenCL platform
    uint32_t platform_idx;
    uint32_t device_idx;

    // Precision
    bool double_prec;

    // OpenCL device type
    ocl_device_type device_type;

    // FPGA config
    sim_algo algo;
    bool multi_kernel;

    // Network params
    double psn_w;
    double p11, p20, p21, p22;
    double v_r, v_thr;

    // Refractory period
    uint8_t t_ref_tics;

    // Rowstride for matrices with n_neurons columns
    uint32_t rowstride;

    // Initial neuron voltages
    double *v_m_init;

    // Layer config
    uint32_t n_layers;
    double *dc_per_layer;
    uint32_t *cnt_per_layer;

    // Psn data
    uint32_t max_n_tics;
    uint8_t *psn;

    // Index
    synapse_index *index;
} sim_config;

typedef struct {
    uint32_t n_spikes;
    uint64_t nanos;
    npy_arr *arr;
} sim_result;

sim_config *sim_config_init(int argc, char *argv[]);
void sim_config_print(sim_config *me);
void sim_config_print_banner(sim_config *me);
void sim_config_free(sim_config *me);
void sim_config_save_spikes(sim_config *me, npy_arr *arr);
sim_result sim_config_run_on_cpu(sim_config *me);
sim_result sim_config_run_on_opencl(sim_config *me);

#endif
