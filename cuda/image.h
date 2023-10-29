#pragma once
#include <stdio.h>
#include <stdlib.h>

__host__ unsigned char *load_image(const char *filename,
                                   size_t *width,
                                   size_t *height,
                                   size_t *channels);
__host__ void free_image(unsigned char *image);
__host__ int save_image(const char *filename,
                        unsigned char *img,
                        size_t width,
                        size_t height,
                        size_t channels);