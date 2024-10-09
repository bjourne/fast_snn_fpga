// Copyright (C) 2022-2024 Björn A. Lindqvist <bjourne@gmail.com>
//
// All functions should return a cl_int, indicating whether an error
// occurred.
#ifndef OPENCL_H
#define OPENCL_H

#include <stdbool.h>
#include "pretty.h"

#define CL_TARGET_OPENCL_VERSION 300

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Some more error codes. All OpenCL's error codes are negative we can
// freely use positive values.
#define OCL_FILE_NOT_FOUND      1
#define OCL_BAD_PLATFORM_IDX    2
#define OCL_BAD_DEVICE_IDX      3

// Print functions
void
ocl_print_platform_details(cl_platform_id plat);

void
ocl_print_device_details(cl_device_id dev, pretty_printer *pp);

// Platform and device functions
cl_int
ocl_get_platforms(cl_uint *n_platforms, cl_platform_id **platforms);
cl_int
ocl_get_devices(cl_platform_id platform,
                cl_uint *n_devices, cl_device_id **devices);
void *
ocl_get_platform_info(cl_platform_id platform,
                      cl_platform_info info);

// Error handling
void ocl_check_err(cl_int err, char *file, int line);
#define OCL_CHECK_ERR(err) ocl_check_err(err, __FILE__, __LINE__)

// Loading and running kernels
cl_int
ocl_load_kernels(cl_context ctx, cl_device_id dev,
                 const char *path, const char *options,
                 size_t n_kernels, char *names[],
                 cl_program *program, cl_kernel *kernels);

cl_int
ocl_set_kernel_arguments(cl_kernel kernel, size_t n_args, ...);

cl_int
ocl_run_nd_kernel(cl_command_queue queue, cl_kernel kernel,
                  cl_uint work_dim,
                  const size_t *global,
                  const size_t *local,
                  size_t n_args, ...);


// Buffer creation functions
cl_int
ocl_create_and_write_buffer(cl_context ctx, cl_mem_flags flags,
                            cl_command_queue queue,
                            void *src, size_t n_bytes,
                            cl_mem *mem);
cl_int
ocl_create_and_fill_buffer(cl_context ctx, cl_mem_flags flags,
                           cl_command_queue queue,
                           void *pattern, size_t pattern_size,
                           size_t n_bytes, cl_mem *mem);
cl_int
ocl_create_empty_buffer(cl_context ctx, cl_mem_flags flags,
                        size_t n_bytes, cl_mem *mem);

// Buffer write functions
cl_int
ocl_write_buffer(cl_command_queue queue, void *src,
                 size_t n_bytes, cl_mem mem);
cl_int
ocl_fill_buffer(cl_command_queue queue,
                void *pattern, size_t pattern_size,
                size_t n_bytes, cl_mem mem);

// Buffer read functions
cl_int
ocl_read_buffer(cl_command_queue queue, void *dst,
                size_t n_bytes, cl_mem mem);

// Pipe creation
cl_int
ocl_create_pipe(cl_context ctx,
                cl_uint packet_size, cl_uint n_packets,
                cl_mem *mem);

// Event handling
cl_int
ocl_poll_event_until(cl_event event, cl_int exec_status, cl_uint millis);

// Convenience functions
cl_int
ocl_basic_setup(
    cl_uint plat_idx, cl_uint dev_idx,
    cl_platform_id *platform,
    cl_device_id *device,
    cl_context *ctx,
    size_t n_queues,
    cl_command_queue *queues
);

// "Object-oriented" interface. Idea is to make an interface to save
// lots of boilerplate code.
typedef struct {
    cl_mem ptr;
    size_t n_bytes;
    cl_mem_flags flags;
} ocl_ctx_buf;

#define OCL_CTX_MAX_BUFFERS     20
#define OCL_CTX_MAX_QUEUES      20
#define OCL_CTX_MAX_KERNELS     20

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    size_t n_buffers;

    ocl_ctx_buf buffers[OCL_CTX_MAX_BUFFERS];

    size_t n_queues;
    cl_command_queue *queues;
    size_t n_kernels;
    cl_kernel *kernels;
    cl_program program;
    cl_int err;
} ocl_ctx;

ocl_ctx *ocl_ctx_init(cl_uint plat_idx, cl_uint dev_idx, bool print);
void ocl_ctx_free(ocl_ctx *me);

cl_int ocl_ctx_load_kernels(ocl_ctx *me,
                            const char *path, const char *options,
                            size_t n_kernels, char *names[]);

cl_int ocl_ctx_add_queue(ocl_ctx *me);

cl_int ocl_ctx_add_buffer(ocl_ctx *me, ocl_ctx_buf buf);

cl_int ocl_ctx_write_buffer(ocl_ctx *me,
                            size_t queue_idx, size_t buf_idx,
                            void *arr);
cl_int ocl_ctx_fill_buffer(ocl_ctx *me,
                           size_t queue_idx, size_t buf_idx,
                           void *pattern, size_t pattern_size);

cl_int ocl_ctx_read_buffer(ocl_ctx *me,
                           size_t queue_idx, size_t buf_idx,
                           void *arr);

cl_int ocl_ctx_run_kernel(ocl_ctx *me,
                          size_t queue_idx, size_t kernel_idx,
                          size_t work_dim,
                          const size_t *global, const size_t *local,
                          size_t n_args, ...);


#endif
