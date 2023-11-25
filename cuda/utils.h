#pragma once
#include <stdio.h>
#include <cuda.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define ABS(a) ((a) < 0 ? -(a) : (a))
#define TO_IDX_2(y, x, sx) ((y) * (sx) + (x))
#define TO_IDX_3(z, y, x, sy, sx) (((z) * (sy) + (y)) * (sx) + (x))

__host__ void catch_error(cudaError_t err);