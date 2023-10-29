#pragma once
#include <cuda.h>

#include "utils.h"

__host__ double *to_cwh(unsigned char *input, int width, int height, int channels);
__host__ unsigned char *to_whc(double *input, int width, int height, int channels);

__global__ void conv2D(double *input, double *output,
                       size_t width, size_t height, size_t channels,
                       double *kernel, size_t kernel_dim);