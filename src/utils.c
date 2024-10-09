#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "utils.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

bool
files_read(const char *path, char **data, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    size_t n = (size_t)ftell(f);
    rewind(f);
    if (size) {
        *size = n;
    }
    if (!data) {
        goto ok;
    }
    *data = (char*)malloc(sizeof(char)*(n + 1));
    if (fread(*data, 1, n, f) != n) {
        free(*data);
        return false;
    }
    (*data)[n] = '\0';
 ok:
    fclose(f);
    return true;
}

uint64_t
nano_count() {
    struct timespec t;
    assert(!clock_gettime(CLOCK_MONOTONIC, &t));
    return (uint64_t)t.tv_sec * 1000000000 + t.tv_nsec;
}

double
nanos_to_secs(uint64_t nanos) {
    return (double)nanos  / (1000 * 1000 * 1000);
}

void*
aligned_calloc(size_t nmemb, size_t size) {
    size_t n_bytes = nmemb * size;
    void *ptr = aligned_alloc(64, n_bytes);
    memset(ptr, 0, n_bytes);
    return ptr;
}

void
mean_std64(double *arr, uint32_t n, double *mean, double *std) {
    double sum = 0.0, fn = (double)n;
    for (uint32_t i = 0; i < n; i++) {
        sum += arr[i];
    }
    *mean = sum / fn;

    *std = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        double diff = arr[i] - *mean;
        *std += diff*diff;
    }
    *std = sqrt(*std / fn);
}

void
sleep_cp(unsigned int millis) {
    #ifdef _WIN32
        Sleep(millis);
    #else
        usleep(millis * 1000);
    #endif // _WIN32
}

void
mprec_write_double(void *arr, size_t i, double value, bool is_double) {
    if (is_double) {
        ((double *)arr)[i] = value;
    } else {
        ((float *)arr)[i] = (float)value;
    }
}
