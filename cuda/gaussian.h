#pragma once
#include <stdio.h>
#include <omp.h>

__host__ double *gen_gaussian_kernel(size_t r, double sigma);
__host__ void mirror_gaussian_kernel(size_t r, double *gaussian_kernel_h);
__host__ void normalize_gaussian_kernel(size_t r, double *gaussian_kernel_h);