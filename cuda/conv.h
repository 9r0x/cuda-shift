#pragma once
#include <cuda.h>

__global__ void conv2D(float *input, float *output,
                       size_t width, size_t height, size_t channels,
                       double *kernel, size_t kernel_dim);