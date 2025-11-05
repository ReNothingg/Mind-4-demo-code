#pragma once

#include <stdint.h>

#include <internal/macros.h>

typedef struct mindfour_DENSELY_PACKED_STRUCTURE
{
    mindfour_ALIGN(2) uint16_t bits;
} mindfour_bfloat16;
static_assert(sizeof(mindfour_bfloat16) == 2, "bfloat16 size is not 2 bytes");

typedef struct mindfour_DENSELY_PACKED_STRUCTURE
{
    mindfour_ALIGN(2) uint16_t bits;
} mindfour_float16;
static_assert(sizeof(mindfour_float16) == 2, "float16 size is not 2 bytes");

typedef struct mindfour_DENSELY_PACKED_STRUCTURE
{
    mindfour_ALIGN(1) uint8_t bits;
} mindfour_float8ue8m0;
static_assert(sizeof(mindfour_float8ue8m0) == 1, "mindfour_float8ue8m0 size is not 1 bytes");

typedef struct mindfour_DENSELY_PACKED_STRUCTURE
{
    mindfour_ALIGN(1) uint8_t bits;
} mindfour_float8e5m2;
static_assert(sizeof(mindfour_float8e5m2) == 1, "float8e5m2 size is not 1 bytes");

typedef struct mindfour_DENSELY_PACKED_STRUCTURE
{
    mindfour_ALIGN(1) uint8_t bits;
} mindfour_float8e4m3;
static_assert(sizeof(mindfour_float8e4m3) == 1, "mindfour_float8e4m3 size is not 1 bytes");

typedef struct mindfour_DENSELY_PACKED_STRUCTURE
{
    mindfour_ALIGN(1) uint8_t bits;
} mindfour_float4e2m1x2;
static_assert(sizeof(mindfour_float4e2m1x2) == 1, "mindfour_float4e2m1x2 size is not 1 bytes");
