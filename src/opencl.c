// Copyright (C) 2022-2024 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "common.h"
#include "opencl.h"
#include "paths.h"
#include "pretty.h"
#include "utils.h"

#define OCL_DEBUG   1

#define BOOL_TO_YES_NO(x) ((x) ? "yes" : "no")

#define ERR_RETURN_STRING(x) case x: return #x;

const char *
err_str(cl_int err) {
    switch (err) {
        ERR_RETURN_STRING(OCL_FILE_NOT_FOUND                )
        ERR_RETURN_STRING(OCL_BAD_PLATFORM_IDX              )
        ERR_RETURN_STRING(OCL_BAD_DEVICE_IDX                )
        ERR_RETURN_STRING(CL_SUCCESS                        )
        ERR_RETURN_STRING(CL_DEVICE_NOT_FOUND               )
        ERR_RETURN_STRING(CL_DEVICE_NOT_AVAILABLE           )
        ERR_RETURN_STRING(CL_COMPILER_NOT_AVAILABLE         )
        ERR_RETURN_STRING(CL_MEM_OBJECT_ALLOCATION_FAILURE  )
        ERR_RETURN_STRING(CL_OUT_OF_RESOURCES               )
        ERR_RETURN_STRING(CL_OUT_OF_HOST_MEMORY             )
        ERR_RETURN_STRING(CL_PROFILING_INFO_NOT_AVAILABLE   )
        ERR_RETURN_STRING(CL_MEM_COPY_OVERLAP               )
        ERR_RETURN_STRING(CL_IMAGE_FORMAT_MISMATCH          )
        ERR_RETURN_STRING(CL_IMAGE_FORMAT_NOT_SUPPORTED     )
        ERR_RETURN_STRING(CL_BUILD_PROGRAM_FAILURE          )
        ERR_RETURN_STRING(CL_MAP_FAILURE                    )
        ERR_RETURN_STRING(CL_MISALIGNED_SUB_BUFFER_OFFSET   )
        ERR_RETURN_STRING(CL_COMPILE_PROGRAM_FAILURE        )
        ERR_RETURN_STRING(CL_LINKER_NOT_AVAILABLE           )
        ERR_RETURN_STRING(CL_LINK_PROGRAM_FAILURE           )
        ERR_RETURN_STRING(CL_DEVICE_PARTITION_FAILED        )
        ERR_RETURN_STRING(CL_KERNEL_ARG_INFO_NOT_AVAILABLE  )
        ERR_RETURN_STRING(CL_INVALID_VALUE                  )
        ERR_RETURN_STRING(CL_INVALID_DEVICE_TYPE            )
        ERR_RETURN_STRING(CL_INVALID_PLATFORM               )
        ERR_RETURN_STRING(CL_INVALID_DEVICE                 )
        ERR_RETURN_STRING(CL_INVALID_CONTEXT                )
        ERR_RETURN_STRING(CL_INVALID_QUEUE_PROPERTIES       )
        ERR_RETURN_STRING(CL_INVALID_COMMAND_QUEUE          )
        ERR_RETURN_STRING(CL_INVALID_HOST_PTR               )
        ERR_RETURN_STRING(CL_INVALID_MEM_OBJECT             )
        ERR_RETURN_STRING(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        ERR_RETURN_STRING(CL_INVALID_IMAGE_SIZE             )
        ERR_RETURN_STRING(CL_INVALID_SAMPLER                )
        ERR_RETURN_STRING(CL_INVALID_BINARY                 )
        ERR_RETURN_STRING(CL_INVALID_BUILD_OPTIONS          )
        ERR_RETURN_STRING(CL_INVALID_PROGRAM                )
        ERR_RETURN_STRING(CL_INVALID_PROGRAM_EXECUTABLE     )
        ERR_RETURN_STRING(CL_INVALID_KERNEL_NAME            )
        ERR_RETURN_STRING(CL_INVALID_KERNEL_DEFINITION      )
        ERR_RETURN_STRING(CL_INVALID_KERNEL                 )
        ERR_RETURN_STRING(CL_INVALID_ARG_INDEX              )
        ERR_RETURN_STRING(CL_INVALID_ARG_VALUE              )
        ERR_RETURN_STRING(CL_INVALID_ARG_SIZE               )
        ERR_RETURN_STRING(CL_INVALID_KERNEL_ARGS            )
        ERR_RETURN_STRING(CL_INVALID_WORK_DIMENSION         )
        ERR_RETURN_STRING(CL_INVALID_WORK_GROUP_SIZE        )
        ERR_RETURN_STRING(CL_INVALID_WORK_ITEM_SIZE         )
        ERR_RETURN_STRING(CL_INVALID_GLOBAL_OFFSET          )
        ERR_RETURN_STRING(CL_INVALID_EVENT_WAIT_LIST        )
        ERR_RETURN_STRING(CL_INVALID_EVENT                  )
        ERR_RETURN_STRING(CL_INVALID_OPERATION              )
        ERR_RETURN_STRING(CL_INVALID_GL_OBJECT              )
        ERR_RETURN_STRING(CL_INVALID_BUFFER_SIZE            )
        ERR_RETURN_STRING(CL_INVALID_MIP_LEVEL              )
        ERR_RETURN_STRING(CL_INVALID_GLOBAL_WORK_SIZE       )
        ERR_RETURN_STRING(CL_INVALID_PROPERTY               )
        ERR_RETURN_STRING(CL_INVALID_IMAGE_DESCRIPTOR       )
        ERR_RETURN_STRING(CL_INVALID_COMPILER_OPTIONS       )
        ERR_RETURN_STRING(CL_INVALID_LINKER_OPTIONS         )
        ERR_RETURN_STRING(CL_INVALID_DEVICE_PARTITION_COUNT )
    default:
        return NULL;
    }
}

void
ocl_check_err(cl_int err, char *file, int line) {
    if (err == CL_SUCCESS)  {
        return;
    }
    const char *s = err_str(err);
    const char *key1 = "OpenCL error";
    const char *key2 = "Caused by";
    if (s) {
        printf("%-12s: %s\n", key1, s);
    } else {
        printf("%-12s: %d\n", key1, err);
    }
    printf("%-12s: %s:%d\n", key2, file, line);
    exit(2);
}

void *
ocl_get_platform_info(cl_platform_id platform,
                      cl_platform_info attr) {
    size_t n_bytes;
    OCL_CHECK_ERR(clGetPlatformInfo(platform, attr, 0, NULL, &n_bytes));
    void *bytes = (void *)malloc(n_bytes);
    OCL_CHECK_ERR(clGetPlatformInfo(platform, attr,
                                    n_bytes, bytes, NULL));
    return bytes;
}

cl_int
ocl_get_platforms(cl_uint *n_platforms, cl_platform_id **platforms) {
    cl_int err;

    err = clGetPlatformIDs(0, NULL, n_platforms);
    if (err != CL_SUCCESS) {
        return err;
    }

    *platforms = (cl_platform_id *)malloc(
        sizeof(cl_platform_id) * *n_platforms);

    return clGetPlatformIDs(*n_platforms, *platforms, NULL);
}

cl_int
ocl_get_devices(cl_platform_id platform,
                cl_uint *n_devices, cl_device_id **devices) {

    cl_int err;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, n_devices);
    if (err != CL_SUCCESS) {
        return err;
    }

    *devices = (cl_device_id *)malloc(
        sizeof(cl_device_id) * *n_devices);

    return clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                          *n_devices, *devices, NULL);
}

static void
print_device_info_str(cl_device_id dev, pretty_printer *pp,
                      cl_device_info attr, char *attr_name) {
    size_t n_bytes;
    clGetDeviceInfo(dev, attr, 0, NULL, &n_bytes);
    char *bytes = (char *)malloc(n_bytes);
    clGetDeviceInfo(dev, attr, n_bytes, bytes, NULL);
    pp_print_key_value(pp, attr_name, "%s", bytes);
    free(bytes);
}

void
ocl_print_device_details(cl_device_id dev, pretty_printer *pp) {
    pretty_printer *use_pp;
    if (!pp) {
        use_pp = pp_init();
    } else {
        use_pp = pp;
    }

    char *attr_names[] = {
        "Name", "Version", "Driver", "C Version"
    };
    int attr_types[] = {
        CL_DEVICE_NAME,
        CL_DEVICE_VERSION, CL_DRIVER_VERSION,
        CL_DEVICE_OPENCL_C_VERSION
    };
    for (size_t i = 0; i < ARRAY_SIZE(attr_types); i++) {
        print_device_info_str(dev, use_pp, attr_types[i], attr_names[i]);
    }

    char *keys[] = {
        "Compute units",
        "Global memory",
        "Max allocation",
        "Max wg. size",
        "Local mem. size"
    };
    cl_device_info params[] = {
        CL_DEVICE_MAX_COMPUTE_UNITS,
        CL_DEVICE_GLOBAL_MEM_SIZE,
        CL_DEVICE_MAX_MEM_ALLOC_SIZE,
        CL_DEVICE_MAX_WORK_GROUP_SIZE,
        CL_DEVICE_LOCAL_MEM_SIZE
    };
    size_t sizes[] = {
        sizeof(cl_uint),
        sizeof(cl_ulong),
        sizeof(cl_ulong),
        sizeof(cl_ulong),
        sizeof(cl_ulong)
    };
    char *suffixes[] = {"", "B", "B", "", "B"};
    for (size_t i = 0; i < 5; i++) {
        char *suf = suffixes[i];
        char *key = keys[i];
        uint64_t val;
        OCL_CHECK_ERR(clGetDeviceInfo(dev, params[i], sizes[i], &val, NULL));
        use_pp->n_decimals = !strcmp(suf, "") ? 0 : 2;
        pp_print_key_value_with_unit(use_pp, key, (double)val, suf);
    }

    //print_prefix(ind);
    size_t n_bytes;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                    0, NULL, &n_bytes);
    size_t *d = (size_t *)malloc(n_bytes);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, n_bytes, d, NULL);

    pp_print_key_value(use_pp,
                       "Max work items",
                       "%ld:%ld:%ld", d[0], d[1], d[2]);
    free(d);

    cl_device_info flags[] = {
        CL_DEVICE_PIPE_SUPPORT,
        CL_DEVICE_COMPILER_AVAILABLE,
        CL_DEVICE_IMAGE_SUPPORT
    };
    char *names[] = {
        "Pipe support",
        "Compiler usable",
        "Image support"
    };

    for (size_t i = 0; i < ARRAY_SIZE(flags); i++) {
        cl_bool val;
        cl_int err = clGetDeviceInfo(dev, flags[i],
                                     sizeof(cl_bool), &val, NULL);
        pp_print_key_value(use_pp, names[i], "%s",
                           BOOL_TO_YES_NO(err == CL_SUCCESS && val));
    }
    if (!pp) {
        pp_free(use_pp);
    }
}

void
ocl_print_platform_details(cl_platform_id plat) {
    cl_platform_info attr_types[] = {
        CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
        CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS
    };
    char *attr_names[] = {
        "Name", "Vendor",
        "Version", "Profile", "Extensions"
    };

    pretty_printer *pp = pp_init();
    for (size_t i = 0; i < ARRAY_SIZE(attr_names); i++) {
        char *info = (char *)ocl_get_platform_info(plat,
                                                   attr_types[i]);
        pp_print_key_value(pp, attr_names[i], "%s", info);
        free(info);
    }
    cl_uint n_devices;
    cl_device_id *devices;

    OCL_CHECK_ERR(ocl_get_devices(plat, &n_devices, &devices));
    pp_print_key_value(pp, "Devices", "%d", n_devices);
    printf("\n");
    pp->indent++;
    for (cl_uint i = 0; i < n_devices; i++) {
        ocl_print_device_details(devices[i], pp);
    }
    pp->indent--;
    printf("\n");
    free(devices);
    pp_free(pp);
}

cl_int
ocl_load_kernels(cl_context ctx, cl_device_id dev,
                 const char *path, const char *options,
                 size_t n_kernels, char *names[],
                 cl_program *program, cl_kernel *kernels) {
    char *data;
    size_t n_data;

    if (!files_read(path, &data, &n_data)) {
        return OCL_FILE_NOT_FOUND;
    }

    // Check file extension
    cl_int err;
    if (!strcmp(paths_ext(path), "cl")) {
        *program = clCreateProgramWithSource(
            ctx, 1,
            (const char **)&data,
            (const size_t *)&n_data, &err);

    } else {
        *program = clCreateProgramWithBinary(
            ctx, 1, &dev,
            &n_data, (const unsigned char **)&data,
            &err, NULL);
    }
    free(data);
    if (err != CL_SUCCESS) {
        return err;
    }

    err = clBuildProgram(*program, 1, &dev, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Build failed: %s\n", err_str(err));
        size_t n_log;
        OCL_CHECK_ERR(clGetProgramBuildInfo(*program, dev, CL_PROGRAM_BUILD_LOG,
                                            0, NULL, &n_log));
        assert(n_log > 0);

        // Acquire, print, and free the log.
        char * log = (char *)malloc(n_log);
        clGetProgramBuildInfo(*program, dev,
                              CL_PROGRAM_BUILD_LOG, n_log, log, NULL);
        printf("%s\n", log);
        free(log);
        return err;
    }

    for (size_t i = 0; i < n_kernels; i++) {
        kernels[i] = clCreateKernel(*program, names[i], &err);
        if (err != CL_SUCCESS) {
            return err;
        }
    }
    return CL_SUCCESS;
}


static cl_int
set_kernel_arguments(cl_kernel kernel, int n_args, va_list ap) {
    for (int i = 0; i < n_args; i++) {
        size_t arg_size = va_arg(ap, size_t);
        void *arg_value = va_arg(ap, void *);
        cl_int err = clSetKernelArg(kernel, i, arg_size, arg_value);
        if (err != CL_SUCCESS) {
            return err;
        }
    }
    return CL_SUCCESS;
}

cl_int
ocl_set_kernel_arguments(cl_kernel kernel, size_t n_args, ...) {
    va_list ap;
    va_start(ap, n_args);
    cl_int err = set_kernel_arguments(kernel, n_args, ap);
    va_end(ap);
    return err;
}

cl_int
ocl_run_nd_kernel(cl_command_queue queue, cl_kernel kernel,
                  cl_uint work_dim,
                  const size_t *global,
                  const size_t *local,
                  size_t n_args, ...) {
    va_list ap;
    va_start(ap, n_args);
    cl_int err = set_kernel_arguments(kernel, n_args, ap);
    va_end(ap);
    if (err != CL_SUCCESS) {
        return err;
    }

    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL,
                                 global, local, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        return err;
    }
    return ocl_poll_event_until(event, CL_COMPLETE, 10);
}

// Buffer creation functions
cl_int
ocl_create_and_write_buffer(cl_context ctx, cl_mem_flags flags,
                            cl_command_queue queue, void *src,
                            size_t n_bytes, cl_mem *mem) {
    cl_int err;
    *mem = clCreateBuffer(
        ctx, flags,
        n_bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        return err;
    }
    err = ocl_write_buffer(queue, src, n_bytes, *mem);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(*mem);
    }
    return err;
}


cl_int
ocl_create_and_fill_buffer(cl_context ctx, cl_mem_flags flags,
                           cl_command_queue queue,
                           void *pattern, size_t pattern_size,
                           size_t n_bytes, cl_mem *mem) {
    cl_int err;
    *mem = clCreateBuffer(
        ctx, flags,
        n_bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        return err;
    }
    err = ocl_fill_buffer(queue, pattern, pattern_size, n_bytes, *mem);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(*mem);
    }
    return err;
}


cl_int
ocl_create_empty_buffer(cl_context ctx, cl_mem_flags flags,
                        size_t n_bytes, cl_mem *mem) {
    cl_int err;
    *mem = clCreateBuffer(
        ctx, flags,
        n_bytes, NULL, &err);
    return err;
}

// Buffer write functions
cl_int
ocl_write_buffer(cl_command_queue queue, void *src,
                 size_t n_bytes, cl_mem mem) {
    #if OCL_DEBUG == 1
    printf("%-8s %10ld bytes to buffer %p from %p.\n",
           "Writing", n_bytes, mem, src);
    #endif
    cl_int err = clEnqueueWriteBuffer(queue, mem, CL_TRUE,
                                      0, n_bytes, src,
                                      0, NULL, NULL);
    if (err != CL_SUCCESS) {
        return err;
    }
    return clFinish(queue);
}

cl_int
ocl_fill_buffer(cl_command_queue queue,
                void *pattern, size_t pattern_size,
                size_t n_bytes, cl_mem mem) {
    #if OCL_DEBUG == 1
    printf("%-8s %10ld bytes to buffer %p.\n",
           "Filling", n_bytes, mem);
    #endif
    cl_int err = clEnqueueFillBuffer(queue, mem,
                                     pattern, pattern_size,
                                     0, n_bytes,
                                     0, NULL, NULL);
    if (err != CL_SUCCESS) {
        return err;
    }
    return clFinish(queue);
}



// Buffer read functions
cl_int
ocl_read_buffer(cl_command_queue queue, void *dst,
                size_t n_bytes, cl_mem mem) {
    return clEnqueueReadBuffer(
        queue, mem, CL_TRUE,
        0, n_bytes, dst,
        0, NULL, NULL);
}


// If queue is NULL the queue is initialized.
cl_int
ocl_basic_setup(cl_uint plat_idx, cl_uint dev_idx,
                cl_platform_id *platform,
                cl_device_id *device,
                cl_context *ctx,
                size_t n_queues,
                cl_command_queue *queues) {
    cl_uint n_platforms;
    cl_platform_id *platforms;
    cl_int err = ocl_get_platforms(&n_platforms, &platforms);
    if (err != CL_SUCCESS) {
        return err;
    }
    if (plat_idx >= n_platforms) {
        return OCL_BAD_PLATFORM_IDX;
    }
    *platform = platforms[plat_idx];

    cl_uint n_devices;
    cl_device_id *devices;
    err = ocl_get_devices(*platform, &n_devices, &devices);
    if (err != CL_SUCCESS) {
        return err;
    }
    if (dev_idx >= n_devices) {
        return OCL_BAD_DEVICE_IDX;
    }
    *device = devices[dev_idx];

    *ctx = clCreateContext(NULL, 1, device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        goto done;
    }
    for (size_t i = 0; i < n_queues; i++) {
        queues[i] = clCreateCommandQueueWithProperties(
            *ctx, *device, NULL, &err);
        if (err != CL_SUCCESS) {
            for (size_t j = 0; j < i; j++) {
                clReleaseCommandQueue(queues[j]);
            }
            clReleaseContext(*ctx);
            goto done;
        }
    }
 done:
    free(devices);
    free(platforms);
    return err;
}

cl_int
ocl_create_pipe(cl_context ctx,
                cl_uint packet_size, cl_uint n_packets,
                cl_mem *mem) {
    cl_int err;
    *mem = clCreatePipe(
        ctx, CL_MEM_READ_WRITE,
        packet_size, n_packets,
        NULL, &err
    );
    return err;
}

// Event handling
cl_int
ocl_poll_event_until(cl_event event,
                     cl_int exec_status,
                     cl_uint millis) {
    while (true) {
        cl_int value;
        cl_int err = clGetEventInfo(event,
                                    CL_EVENT_COMMAND_EXECUTION_STATUS,
                                    sizeof(cl_int),
                                    &value,
                                    NULL);
        if (err != CL_SUCCESS) {
            return err;
        }
        if (value == exec_status) {
            return CL_SUCCESS;
        }
        sleep_cp(millis);
    }
}

////////////////////////////////////////////////////////////////////////
// Object-oriented interface
////////////////////////////////////////////////////////////////////////

ocl_ctx *
ocl_ctx_init(cl_uint plat_idx, cl_uint dev_idx, bool print) {
    ocl_ctx *me = malloc(sizeof(ocl_ctx));
    me->n_queues = 0;
    me->queues = malloc(sizeof(cl_command_queue) * OCL_CTX_MAX_QUEUES);
    me->n_kernels = 0;
    me->kernels = malloc(sizeof(cl_kernel) * OCL_CTX_MAX_KERNELS);
    me->n_buffers = 0;

    me->err = ocl_basic_setup(plat_idx, dev_idx,
                              &me->platform, &me->device,
                              &me->context, 0, NULL);
    if (me->err == CL_SUCCESS && print) {
        ocl_print_device_details(me->device, 0);
        printf("\n");
    }
    return me;
}

void
ocl_ctx_free(ocl_ctx *me) {
    for (size_t i = 0; i < me->n_queues; i++) {
        OCL_CHECK_ERR(clFlush(me->queues[i]));
        OCL_CHECK_ERR(clFinish(me->queues[i]));
        clReleaseCommandQueue(me->queues[i]);
    }
    for (size_t i = 0; i < me->n_kernels; i++) {
        clReleaseKernel(me->kernels[i]);
    }
    for (size_t i = 0; i < me->n_buffers; i++) {
        clReleaseMemObject(me->buffers[i].ptr);
    }
    clReleaseProgram(me->program);
    clReleaseContext(me->context);
    free(me->kernels);
    free(me->queues);
    free(me);
}

cl_int
ocl_ctx_load_kernels(ocl_ctx *me,
                     const char *path,
                     const char *options,
                     size_t n_kernels,
                     char *names[]) {
    assert(me->n_kernels == 0 && n_kernels < OCL_CTX_MAX_KERNELS);
    me->n_kernels = n_kernels;
    return ocl_load_kernels(me->context, me->device,
                            path, options,
                            n_kernels, names,
                            &me->program, me->kernels);
}

cl_int
ocl_ctx_add_buffer(ocl_ctx *me, ocl_ctx_buf buf) {
    cl_int err;
    size_t idx = me->n_buffers;
    size_t n_bytes = buf.n_bytes;

    buf.ptr = clCreateBuffer(me->context, buf.flags, n_bytes, NULL, &err);
    me->buffers[idx] = buf;
    if (err != CL_SUCCESS) {
        return err;
    }
    #if OCL_DEBUG == 1
    printf("%-8s %10ld bytes as buffer %2ld (%p).\n",
           "Created", n_bytes, idx, buf.ptr);
    #endif
    me->n_buffers++;
    return CL_SUCCESS;
}

cl_int
ocl_ctx_add_queue(ocl_ctx *me) {
    cl_int err;
    me->queues[me->n_queues] = clCreateCommandQueueWithProperties(
        me->context, me->device, NULL, &err
    );
    if (err != CL_SUCCESS) {
        return err;
    }
    me->n_queues++;
    return CL_SUCCESS;
}

cl_int
ocl_ctx_write_buffer(ocl_ctx *me,
                     size_t queue_idx,
                     size_t buf_idx,
                     void *arr) {
    assert(buf_idx < me->n_buffers && queue_idx < me->n_queues);
    ocl_ctx_buf buf = me->buffers[buf_idx];
    cl_command_queue q = me->queues[queue_idx];
    assert(buf.ptr);
    return ocl_write_buffer(q, arr, buf.n_bytes, buf.ptr);
}

cl_int
ocl_ctx_fill_buffer(ocl_ctx *me,
                    size_t queue_idx, size_t buf_idx,
                    void *pat, size_t pat_size) {
    assert(buf_idx < me->n_buffers && queue_idx < me->n_queues);
    ocl_ctx_buf buf = me->buffers[buf_idx];
    assert(buf.ptr);
    cl_command_queue q = me->queues[queue_idx];
    return ocl_fill_buffer(q, pat, pat_size, buf.n_bytes, buf.ptr);
}


cl_int
ocl_ctx_read_buffer(ocl_ctx *me,
                    size_t queue_idx, size_t buf_idx,
                    void *arr) {
    ocl_ctx_buf buf = me->buffers[buf_idx];
    return ocl_read_buffer(
        me->queues[queue_idx],
        arr, buf.n_bytes,
        buf.ptr
    );
}

cl_int
ocl_ctx_run_kernel(
    ocl_ctx *me,
    size_t queue_idx,
    size_t kernel_idx,
    size_t work_dim,
    const size_t *global,
    const size_t *local,
    size_t n_args, ...
) {
    cl_kernel kernel = me->kernels[kernel_idx];
    cl_command_queue queue = me->queues[queue_idx];
    va_list ap;
    va_start(ap, n_args);
    cl_int err = set_kernel_arguments(kernel, n_args, ap);
    va_end(ap);
    if (err != CL_SUCCESS) {
        return err;
    }
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL,
                                 global, local, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        return err;
    }
    return ocl_poll_event_until(event, CL_COMPLETE, 10);
}
