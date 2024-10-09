// Copyright (C) 2024 Björn Lindqvist <bjourne@gmail.com>
#ifndef SHARED_H
#define SHARED_H

// Maximum synapse delay
#define MAX_D                   64

// Max PSN row
#define MAX_PSN_T               15000

// Maximum number of spiking neurons per tic
#define MAX_SPIKES_PER_TIC      256

#define SPIKE_RECORD_ELEMENTS   (50 * 1000 * 1000)

#define DEPTH               (N_FIXED_NEURONS / WIDTH)
#define N_TURNS             (MAX_D / N_FRONT)

// Shared with OpenCL
#define N_GPU_WORK_ITEMS_POW    0
#define N_GPU_WORK_ITEMS        (1 << N_GPU_WORK_ITEMS_POW)
#define N_GPU_WORK_ITEMS_MASK   (N_GPU_WORK_ITEMS - 1)

// Size of the spike buffer
#define I_BUF_ROWS              (1 << 6)
#define I_BUF_ROW_MASK          (I_BUF_ROWS - 1)

// Whether to use real barriers
#if N_GPU_WORK_ITEMS > 1
#define GLOBAL_BARRIER barrier(CLK_GLOBAL_MEM_FENCE)
#define LOCAL_BARRIER barrier(CLK_LOCAL_MEM_FENCE)
#else
#define GLOBAL_BARRIER
#define LOCAL_BARRIER
#endif

// Alignment of 2d buffers
#define NEURON_ALIGN        64

#define ALIGN_BUFFER_SIZE(x)     (((x) + NEURON_ALIGN - 1) / NEURON_ALIGN * NEURON_ALIGN)
#define PAD_NEURON_COUNT(x)      ALIGN_BUFFER_SIZE((x) + 64)

// Fixed neuron count for fpga
#define N_BASE_NEURONS      77169

// Should always evaluate to 77248. We add 64 since we need space for
// idempotent receiver neurons.
#define N_FIXED_NEURONS     PAD_NEURON_COUNT(N_BASE_NEURONS)

#if defined(IS_HOST)

#include <stdint.h>

typedef struct {
    uint32_t dst;
    float weight;
} synapse;

#elif defined(IS_DEVICE)

typedef struct {
    uint o0;
    uint o1;
} range;

typedef struct {
    uint dst;
    float weight;
} synapse;

// Extensions
#pragma OPENCL EXTENSION cl_intel_channels : enable


#if VECTOR_WIDTH==2
#define VLOAD(o, a) vload2((o), (a))
#define VLOAD_AT(o, a) vload2((o) / VECTOR_WIDTH, (a))
#define VSTORE(item, ofs, arr) vstore2((item), (ofs), (arr))
#define VSTORE_AT(e, o, a) vstore2((e), (o) / VECTOR_WIDTH, (a))
#define CONVERT_VUCHAR  convert_uchar2
#define CONVERT_VLONG  convert_long2
#define CONVERT_VINT  convert_int2
#define CONVERT_VUINT  convert_uint2
#define CONVERT_VDOUBLE convert_double2
#define CONVERT_VFLOAT convert_float2
typedef uchar2 vuchar;
typedef int2 vint;
typedef uint2 vuint;
typedef long2 vlong;
typedef float2 vfloat;
typedef double2 vdouble;
#elif VECTOR_WIDTH==4
#define VLOAD(o, a) vload4((o), (a))
#define VLOAD_AT(o, a) vload4((o) / VECTOR_WIDTH, (a))
#define VSTORE(item, ofs, arr) vstore4((item), (ofs), (arr))
#define VSTORE_AT(e, o, a) vstore4((e), (o) / VECTOR_WIDTH, (a))
#define CONVERT_VCHAR  convert_char4
#define CONVERT_VUCHAR  convert_uchar4
#define CONVERT_VLONG  convert_long4
#define CONVERT_VINT  convert_int4
#define CONVERT_VUINT  convert_uint4
#define CONVERT_VDOUBLE convert_double4
#define CONVERT_VFLOAT convert_float4
typedef uchar4 vuchar;
typedef int4 vint;
typedef uint4 vuint;
typedef long4 vlong;
typedef float4 vfloat;
typedef double4 vdouble;
#elif VECTOR_WIDTH==8
#define VLOAD(o, a) vload8((o), (a))
#define VLOAD_AT(o, a) vload8((o) / VECTOR_WIDTH, (a))
#define VSTORE(item, ofs, arr) vstore8((item), (ofs), (arr))
#define VSTORE_AT(e, o, a) vstore8((e), (o) / VECTOR_WIDTH, (a))
#define CONVERT_VCHAR  convert_char8
#define CONVERT_VUCHAR  convert_uchar8
#define CONVERT_VLONG  convert_long8
#define CONVERT_VINT  convert_int8
#define CONVERT_VUINT  convert_uint8
#define CONVERT_VDOUBLE convert_double8
#define CONVERT_VFLOAT convert_float8

typedef uchar8 vuchar;
typedef int8 vint;
typedef uint8 vuint;
typedef long8 vlong;
typedef float8 vfloat;
typedef double8 vdouble;

#elif VECTOR_WIDTH==16
#define VLOAD(o, a) vload16((o), (a))
#define VLOAD_AT(o, a) vload16((o) / VECTOR_WIDTH, (a))
#define VSTORE(item, ofs, arr) vstore16((item), (ofs), (arr))
#define VSTORE_AT(e, o, a) vstore16((e), (o) / VECTOR_WIDTH, (a))
#define CONVERT_VCHAR  convert_char16
#define CONVERT_VUCHAR  convert_uchar16
#define CONVERT_VLONG  convert_long16
#define CONVERT_VINT  convert_int16
#define CONVERT_VUINT  convert_uint16
#define CONVERT_VDOUBLE convert_double16
#define CONVERT_VFLOAT convert_float16
typedef uchar16 vuchar;
typedef int16 vint;
typedef uint16 vuint;
typedef long16 vlong;
typedef float16 vfloat;
typedef double16 vdouble;
#elif VECTOR_WIDTH==32
#else
#endif

#if defined(VECTOR_WIDTH)
#define VLOAD_AS_INT(o, a) CONVERT_VINT(VLOAD(o, a))
#define VLOAD_AS_FLOAT(o, a) CONVERT_VFLOAT(VLOAD(o, a))

#define VLOAD_AT_AS_INT(o, a) CONVERT_VINT(VLOAD_AT(o, a))
#define VLOAD_AT_AS_LONG(o, a) CONVERT_VLONG(VLOAD_AT(o, a))
#define VLOAD_AT_AS_FLOAT(o, a) CONVERT_VFLOAT(VLOAD_AT(o, a))
#define VLOAD_AT_AS_DOUBLE(o, a) CONVERT_VDOUBLE(VLOAD_AT(o, a))

#define VSTORE_AS_CHAR(e, o, a) VSTORE(CONVERT_VCHAR(e), o, a)
#define VSTORE_AS_UCHAR(e, o, a) VSTORE(CONVERT_VUCHAR(e), o, a)

#define VSTORE_AT_AS_UINT(e, o, a) VSTORE_AT(CONVERT_VUINT(e), o, a)
#define VSTORE_AT_AS_CHAR(e, o, a) VSTORE_AT(CONVERT_VCHAR(e), o, a)
#define VSTORE_AT_AS_UCHAR(e, o, a) VSTORE_AT(CONVERT_VUCHAR(e), o, a)

#define VLOAD_AS_LONG(o, a) CONVERT_VLONG(VLOAD(o, a))
#define VLOAD_AS_DOUBLE(o, a) CONVERT_VDOUBLE(VLOAD(o, a))

#endif

#if !defined(USE_DOUBLES)
#error "Specify USE_DOUBLES"
#endif

#if USE_DOUBLES==0

#define FTYPE       float
#define VFTYPE      vfloat
#define VITYPE      vint
#define FTYPE_SIZE  4
#define VLOAD_AT_AS_FTYPE   VLOAD_AT_AS_FLOAT
#define VLOAD_AT_AS_ITYPE   VLOAD_AT_AS_INT

#else

#define FTYPE               double
#define VFTYPE              vdouble
#define VITYPE              vlong
#define FTYPE_SIZE          8
#define VLOAD_AT_AS_FTYPE   VLOAD_AT_AS_DOUBLE
#define VLOAD_AT_AS_ITYPE   VLOAD_AT_AS_LONG

#endif



#else

#error "Define IS_HOST or IS_DEVICE!"

#endif


#endif
