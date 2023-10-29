#pragma once
#include <cuda.h>

#include "utils.h"

__host__ double *to_cwh(unsigned char *input, int width, int height, int channels);
__host__ unsigned char *to_whc(double *input, int width, int height, int channels);

__global__ void gen_rectangle_mask(size_t width, size_t height,
                                   size_t left, size_t right,
                                   size_t top, size_t bottom,
                                   double alpha, double *__restrict__ mask);
__global__ void apply_mask(double *__restrict__ input,
                           double *__restrict__ mask,
                           double *__restrict__ output,
                           size_t width, size_t height, size_t channels);
__global__ void conv2D(double *__restrict__ input, double *__restrict__ output,
                       size_t in_width, size_t in_height, size_t channels,
                       double *__restrict__ kernel, size_t kernel_radius);