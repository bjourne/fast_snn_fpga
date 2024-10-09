// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

uint64_t nano_count();
double nanos_to_secs(uint64_t nanos);
void* aligned_calloc(size_t nmemb, size_t size);
bool files_read(const char *path, char **data, size_t *size);
void sleep_cp(unsigned int millis);

void mprec_write_double(void *arr, size_t i, double value, bool is_double);

#endif
