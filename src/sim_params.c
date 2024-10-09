// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "array.h"
#include "common.h"
#include "npy.h"
#include "paths.h"
#include "sim_params.h"
#include "synced_queue.h"
#include "utils.h"

#define FILE_LAYER_PARAMS "layer_params.npy"
#define FILE_NETWORK_PARAMS "network_params.npy"
#define FILE_NEURONS_CURRENT_INITIAL "neurons_current.npy"
#define FILE_PSN_SPIKES "poisson_spikes.npy"
#define FILE_SYNAPSE_INDEX "synapse_offset_delay.npy"
#define FILE_SYNAPSE_DST "synapse_dst.npy"
#define FILE_SYNAPSE_WEIGHT "synapse_weight.npy"
#define FILE_SYNAPSE_DELAY "synapse_delay.npy"

static void*
extract_data(npy_arr* arr) {
    void *ptr = arr->data;
    free(arr);
    return ptr;
}

static npy_arr*
load_arr(const char *dir, const char *fname) {
    char *path = paths_join(dir, fname);

    npy_arr *arr = npy_load(path);
    if (arr->error_code) {
        printf("Failed loading array %s.\n", path);
        free(path);
        npy_free(arr);
        return NULL;
    }
    char shape[256];
    npy_format_dims(arr, shape);
    printf("Loaded array %-30s %-10s\n", fname, shape);
    free(path);
    return arr;
}

#define PARSE_PARAM(x, y) \
    if (!strcmp(k, x)) {  \
    y = v;                \
    handled = true;       \
    }

#define PRINT_FLOAT_PARAM(x, y) printf("%-15s: %15.9f\n", x, y)
#define PRINT_INT_PARAM(x, y) printf("%-15s: %15d\n", x, y)
#define PRINT_LONG_PARAM(x, y) printf("%-15s: %15ld\n", x, y)
#define PRINT_STR_PARAM(x, y) printf("%-15s: %15s\n", x, y)
#define PRINT_BOOL_PARAM(x, y) printf("%-15s: %15s\n", x, (y) ? "yes" : "no");

static char *
device_type_name(ocl_device_type dev) {
    if (dev == OCL_DEVICE_TYPE_CPU) {
        return "cpu";
    } else if (dev == OCL_DEVICE_TYPE_GPU) {
        return "gpu";
    } else if (dev == OCL_DEVICE_TYPE_FPGA) {
        return "fpga";
    } else {
        assert(false);
    }
}

static char *
algo_name(sim_algo algo) {
    if (algo == SIM_ALGO_GMEM) {
        return "gmem";
    } else if (algo == SIM_ALGO_JIT) {
        return "jit";
    } else {
        return "horiz";
    }
}

static char *
multi_tag(bool is_multi) {
    return is_multi ? "multi" : "mono";
}

static void
usage(char *binary) {
    printf("Usage:\n");
    printf("\n");
    printf("  %s net_dir n_tics cpu cpu_flags\n", binary);
    printf("  %s net_dir n_tics opencl ocl_flags p_idx d_idx program\n", binary);
    printf("\n");
    printf("where\n\n");
    printf("  * net_dir is the network parameter directory;\n");
    printf("  * n_tics is the simulation duration;\n");
    printf("  * cpu_flags specifies precision (d or s);\n");
    printf("  * ocl_flags specifies OpenCL simulation flags;\n");
    printf("  * p-idx and d-idx are the OpenCL platform and device index; and \n");
    printf("  * program is the OpenCL program's path.\n");
}

void
sim_config_print(sim_config *me) {
    printf("\n== Simulation config ==\n");
    PRINT_INT_PARAM("n_sim_tics", me->n_sim_tics);
    PRINT_BOOL_PARAM("double_prec", me->double_prec);
    PRINT_BOOL_PARAM("is_cpu", me->is_cpu);
    if (!me->is_cpu) {
        PRINT_STR_PARAM("program_path", me->program_path);
        PRINT_INT_PARAM("platform_idx", me->platform_idx);
        PRINT_INT_PARAM("device_idx", me->device_idx);

        PRINT_STR_PARAM("device_type", device_type_name(me->device_type));
        PRINT_STR_PARAM("algo", algo_name(me->algo));
        PRINT_BOOL_PARAM("multi_kernel", me->multi_kernel);
    }
    PRINT_STR_PARAM("network_dir", me->network_dir);
    PRINT_FLOAT_PARAM("v_thr", me->v_thr);
    PRINT_FLOAT_PARAM("v_r", me->v_r);
    PRINT_FLOAT_PARAM("p11", me->p11);
    PRINT_FLOAT_PARAM("p20", me->p20);
    PRINT_FLOAT_PARAM("p21", me->p21);
    PRINT_FLOAT_PARAM("p22", me->p22);
    PRINT_INT_PARAM("t_ref_tics", me->t_ref_tics);
    PRINT_INT_PARAM("rowstride", me->rowstride);
    PRINT_INT_PARAM("n_layers", me->n_layers);
    PRINT_INT_PARAM("max_n_tics", me->max_n_tics);
    PRINT_INT_PARAM("n_neurons", me->index->n_neurons);
    PRINT_INT_PARAM("n_synapses", me->index->n_synapses);
    PRINT_INT_PARAM("index_width", me->index->width);
}

sim_config *
sim_config_init(int argc, char *argv[]) {
    sim_config *me = NULL;
    npy_arr *arr = NULL;
    if (argc < 5) {
        goto error;
    }

    char *end = NULL;
    me = calloc(1, sizeof(sim_config));
    me->network_dir = argv[1];
    me->n_sim_tics = strtol(argv[2], &end, 10);
    if (argv[2] == end) {
        goto error;
    }
    me->is_cpu = !strcmp(argv[3], "cpu");
    if (me->is_cpu) {
        if (argc != 5) {
            goto error;
        }
    } else {
        if (argc != 8) {
            goto error;
        }
        me->program_path = argv[7];
        me->platform_idx = strtol(argv[5], &end, 10);
        if (argv[6] == end) {
            goto error;
        }
        me->device_idx = strtol(argv[6], &end, 10);
        if (argv[7] == end) {
            goto error;
        }
    }

    // Flag parse
    char *copy = strdup(argv[4]);
    char *tok = strtok(copy, "/");
    while (tok) {
        if (!strcmp(tok, "gmem")) {
            me->algo = SIM_ALGO_GMEM;
        } else if (!strcmp(tok, "jit")) {
            me->algo = SIM_ALGO_JIT;
        } else if (!strcmp(tok, "horiz")) {
            me->algo = SIM_ALGO_HORIZON;
        } else if (!strcmp(tok, "d")) {
            me->double_prec = true;
        } else if (!strcmp(tok, "s")) {
            me->double_prec = false;
        } else if (!strcmp(tok, "multi")) {
            me->multi_kernel = true;
        } else if (!strcmp(tok, "mono")) {
            me->multi_kernel = false;
        } else if (!strcmp(tok, "cpu")) {
            me->device_type = OCL_DEVICE_TYPE_CPU;
        } else if (!strcmp(tok, "fpga")) {
            me->device_type = OCL_DEVICE_TYPE_FPGA;
        } else if (!strcmp(tok, "gpu")) {
            me->device_type = OCL_DEVICE_TYPE_GPU;
        } else {
            printf("Unrecognized flag: %s\n", tok);
            free(copy);
            return false;
        }
        tok = strtok(NULL, "/");
    }
    free(copy);

    arr = load_arr(me->network_dir, FILE_NETWORK_PARAMS);
    if (!arr) {
        goto error;
    }
    int el_size = arr->el_size;
    me->v_thr = me->v_r = me->p11
        = me->p20 = me->p21 = me->p22
        = me->psn_w = -1.0;
    me->t_ref_tics = 0;
    for (int i = 0; i < arr->dims[0]; i++) {
        char *k = arr->data + 2 * el_size * i;
        double v = strtod(k + el_size, NULL);
        bool handled = false;
        PARSE_PARAM("v_thr", me->v_thr);
        PARSE_PARAM("v_r", me->v_r);
        PARSE_PARAM("p11", me->p11);
        PARSE_PARAM("p20", me->p20);
        PARSE_PARAM("p21", me->p21);
        PARSE_PARAM("p22", me->p22);
        PARSE_PARAM("psn_w", me->psn_w);
        PARSE_PARAM("t_ref_tics", me->t_ref_tics);
        if (!handled) {
            // Unknown parameter
            goto error;
        }
    }
    if (me->v_thr == -1 || me->v_r == -1 || me->p11 == -1 ||
        me->p20 == -1 || me->p21 == -1 || me->p22 == -1 ||
        me->psn_w == -1 || me->t_ref_tics == 0) {
        goto error;
    }
    npy_free(arr);

    // Initial voltages
    arr = load_arr(me->network_dir, FILE_NEURONS_CURRENT_INITIAL);
    if (!arr) {
        goto error;
    }
    uint32_t n_neurons = arr->dims[0];
    me->rowstride = PAD_NEURON_COUNT(n_neurons);
    me->v_m_init = extract_data(arr);

    // Load counts and constant current per layer
    arr = load_arr(me->network_dir, FILE_LAYER_PARAMS);
    if (!arr) {
        goto error;
    }
    me->n_layers = arr->dims[1];

    me->dc_per_layer = malloc(me->n_layers * sizeof(double));
    me->cnt_per_layer = malloc(me->n_layers * sizeof(uint32_t));
    for (uint32_t i = 0; i < me->n_layers; i++) {
        me->cnt_per_layer[i] = npy_value_at_as_double(arr, i);
        me->dc_per_layer[i] =
            npy_value_at_as_double(arr, me->n_layers + i);
    }
    npy_free(arr);

    // Load Poisson
    arr = load_arr(me->network_dir, FILE_PSN_SPIKES);
    if (!arr) {
        goto error;
    }
    me->max_n_tics = arr->dims[0];
    assert((uint32_t)arr->dims[1] == n_neurons);
    uint8_t *psn = extract_data(arr);

    uint8_t *psn_strided = aligned_calloc(me->rowstride * me->max_n_tics,
                                          sizeof(uint8_t));
    assert(psn_strided);
    for (uint32_t t = 0; t < me->max_n_tics; t++) {
        memcpy(&psn_strided[t * me->rowstride],
               &psn[t * n_neurons],
               n_neurons);

    }
    me->psn = psn_strided;
    free(psn);

    // Load synapse delay offsets
    arr = load_arr(me->network_dir, FILE_SYNAPSE_INDEX);
    if (!arr) {
        goto error;
    }
    uint32_t *index = extract_data(arr);
    uint32_t n_synapses = index[64 * n_neurons - 1];

    // Syn weight
    arr = load_arr(me->network_dir, FILE_SYNAPSE_WEIGHT);
    if (!arr) {
        goto error;
    }
    float *syn_weight = extract_data(arr);

    // Syn delay
    arr = load_arr(me->network_dir, FILE_SYNAPSE_DELAY);
    if (!arr) {
        goto error;
    }
    uint8_t *syn_delay = extract_data(arr);

    arr = load_arr(me->network_dir, FILE_SYNAPSE_DST);
    if (!arr) {
        goto error;
    }
    uint32_t *syn_dst = extract_data(arr);

    synapse *synapses = aligned_calloc(n_synapses, sizeof(synapse));
    for (uint32_t i = 0; i < n_synapses; i++) {
        synapses[i].dst = syn_dst[i];
        synapses[i].weight = syn_weight[i];
    }

    me->index = synapse_index_init(synapses, index, 64,
                                   n_neurons, n_synapses);
    free(syn_weight);
    free(syn_delay);
    return me;
 error:
    usage(argv[0]);
    if (me) {
        sim_config_free(me);
    }
    if (arr) {
        free(arr);
    }
    return NULL;
}

void
sim_config_print_banner(sim_config *me) {
    char *prec = me->double_prec ? "double" : "single";
    if (me->is_cpu) {
        printf("\n== CPU: %s precision ==\n", prec);
    } else {
        char *backend = device_type_name(me->device_type);
        char *algo = algo_name(me->algo);
        printf("\n== OpenCL: %s backend, "
               "%d work item(s), "
               "%s algorithm, "
               "%s precision, "
               "%s-kernel ==\n",
               backend, N_GPU_WORK_ITEMS, algo, prec,
               me->multi_kernel ? "multi" : "mono");
    }
}

void
sim_config_save_spikes(sim_config *me, npy_arr *arr) {
    char buf[256];
    char *prec_tag = me->double_prec ? "d" : "s";
    uint32_t n_tics = me->n_sim_tics;
    if (me->is_cpu) {
        sprintf(buf, "cpu_%s_%09d.npy", prec_tag, n_tics);
    } else {
        sprintf(buf, "%s_%s_%s_%s_%09d.npy",
                device_type_name(me->device_type),
                algo_name(me->algo),
                multi_tag(me->multi_kernel),
                prec_tag,
                n_tics);
    }
    printf("Saving spikes to %s\n", buf);
    assert(npy_save(arr, buf) == NPY_ERR_NONE);
}


void
sim_config_free(sim_config *me) {
    if (me->v_m_init) {
        free(me->v_m_init);
    }
    if (me->dc_per_layer) {
        free(me->dc_per_layer);
    }
    if (me->cnt_per_layer) {
        free(me->cnt_per_layer);
    }
    if (me->psn) {
        free(me->psn);
    }
    if (me->index) {
        synapse_index_free(me->index);
    }
    free(me);
}
