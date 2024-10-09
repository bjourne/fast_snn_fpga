// Copyright (C) 2023 Björn Lindqvist <bjourne@gmail.com>
//
// Variable names:
//
//  - ibuf: spike propagation buffer
//  - rec: record of all spikes
//  - cls: spike sender class
//
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "opencl.h"
#include "paths.h"
#include "sim_params.h"
#include "utils.h"

int
main(int argc, char *argv[]) {
    sim_config *sc = sim_config_init(argc, argv);
    if (!sc) {
        return 1;
    }
    sim_config_print(sc);

    printf("\n== Macros ==\n"
           MACRO_KEY_VAL(WIDTH) ", "
           MACRO_KEY_VAL(SYN_UNROLL) ", "
           MACRO_KEY_VAL(N_FRONT) ", "
           MACRO_KEY_VAL(N_LANES) ", "
           MACRO_KEY_VAL(SYN_ALIGN) ", "
           MACRO_KEY_VAL(RECORD_SPIKES)
           "\n");
    uint32_t n_tics = sc->n_sim_tics;
    if (n_tics > sc->max_n_tics) {
        printf("Note: requested %d tics > %d max tics\n",
               n_tics, sc->max_n_tics);
    }

    sim_config_print_banner(sc);

    sim_result res;
    if (sc->is_cpu) {
        res = sim_config_run_on_cpu(sc);
    } else {
        res = sim_config_run_on_opencl(sc);
    }
    double secs = nanos_to_secs(res.nanos);
    double rt_factor = secs / ((double)n_tics / 10000.0);
    printf("%u spikes in %u tics (%.3f s, %.2f slower than realtime)\n",
           res.n_spikes, n_tics, secs, rt_factor);

    sim_config_save_spikes(sc, res.arr);

    assert(res.arr);
    npy_free(res.arr);
    sim_config_free(sc);
    return 0;
}
