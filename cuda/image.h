#pragma once
#include <stdio.h>
#include <stdlib.h>

unsigned char *load_image(const char *filename,
                          size_t *width,
                          size_t *height,
                          size_t *channels);
void free_image(unsigned char *image);
int save_image(const char *filename,
               unsigned char *img,
               size_t width,
               size_t height,
               size_t channels);
double *to_cwh(unsigned char *input, int width, int height, int channels);
unsigned char *to_whc(double *input, int width, int height, int channels);