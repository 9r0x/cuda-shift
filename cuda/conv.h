#pragma once
#include <cuda.h>

#include "utils.h"

__host__ double *to_chw(unsigned char *input, int height, int width, int channels);
__host__ unsigned char *to_hwc(double *input, int height, int width, int channels);

__global__ void sum_masks(size_t height, size_t width, double *__restrict__ m1, double *__restrict__ m2, double *__restrict__ m3);

__global__ void gen_rectangle_mask(size_t height, size_t width,
                                   size_t top, size_t bottom,
                                   size_t left, size_t right,
                                   double *__restrict__ mask, size_t feather_distance);
__global__ void conv2D(double *__restrict__ input,
                       double *__restrict__ mask,
                       double *__restrict__ output,
                       size_t height, size_t width,
                       double *__restrict__ kernel, size_t kernel_radius);
__global__ void blend(double **__restrict__ input,
                      double *__restrict__ output,
                      size_t height, size_t width, size_t num);