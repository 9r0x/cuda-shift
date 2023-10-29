#pragma once
#include <stdio.h>
#include <stdlib.h>

__host__ unsigned char *load_image(const char *__restrict__ filename,
                                   size_t *__restrict__ width,
                                   size_t *__restrict__ height,
                                   size_t *__restrict__ channels);
__host__ int save_image(const char *__restrict__ filename,
                        unsigned char *__restrict__ img,
                        size_t width,
                        size_t height,
                        size_t channels);
__host__ void free_image(unsigned char *image);