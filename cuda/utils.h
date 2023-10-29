#pragma once
#include <stdio.h>
#include <cuda.h>

#define TO_IDX_2(y, x, sx) ((y) * (sx) + (x))
#define TO_IDX_3(z, y, x, sy, sx) (((z) * (sy) + (y)) * (sx) + (x))

__host__ void catch_error(cudaError_t err);