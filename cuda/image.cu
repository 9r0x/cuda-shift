#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TO_LOWER(c) ((c) >= 'A' && (c) <= 'Z') ? (c) + 32 : (c)

char *getFileExtension(const char *path)
{
    const char *lastPeriod = strrchr(path, '.');
    if (!lastPeriod || lastPeriod == path)
        return NULL;

    char *extension = (char *)malloc(strlen(lastPeriod));
    if (!extension)
        return NULL;

    for (size_t i = 1; i < strlen(lastPeriod); i++)
        extension[i - 1] = TO_LOWER(lastPeriod[i]);
    extension[strlen(lastPeriod) - 1] = '\0';
    return extension;
}

unsigned char *load_image(const char *filename,
                          size_t *width,
                          size_t *height,
                          size_t *channels)
{
    int w, h, c;
    unsigned char *img = stbi_load(filename, &w, &h, &c, 0);
    if (img == NULL)
    {
        fprintf(stderr, "[!] Error in loading the image %s\n", filename);
        return NULL;
    }
    *width = (size_t)w;
    *height = (size_t)h;
    *channels = (size_t)c;
    return img;
}

int save_image(const char *filename,
               unsigned char *img,
               size_t width,
               size_t height,
               size_t channels)
{
    char *ext = getFileExtension(filename);
    if (!ext)
    {
        // No need to free ext here as we returned \0
        fprintf(stderr, "[!] Invalid path.\n");
        return 0;
    }
    else if (strcmp(ext, "jpg") == 0 || strcmp(ext, "jpeg") == 0)
    {
        free(ext);
        return stbi_write_jpg(filename, (int)width,
                              (int)height, (int)channels, img, 100);
    }
    else if (strcmp(ext, "png") == 0)
    {
        free(ext);
        return stbi_write_png(filename, (int)width,
                              (int)height, (int)channels, img, width * channels);
    }
    else
    {
        free(ext);
        fprintf(stderr, "[!] Unsupported file type.\n");
        return 0;
    }
}

void free_image(unsigned char *img)
{
    stbi_image_free(img);
}

double *to_cwh(unsigned char *input, int width, int height, int channels)
{
    double *output = new double[width * height * channels];
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                int oldIdx = (y * width + x) * channels + ch;
                int newIdx = ch * width * height + y * width + x;
                output[newIdx] = (double)(input[oldIdx]) / 255.0;
            }
        }
    }
    return output;
}

unsigned char *to_whc(double *input, int width, int height, int channels)
{
    unsigned char *output = new unsigned char[width * height * channels];
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                int oldIdx = ch * width * height + y * width + x;
                int newIdx = (y * width + x) * channels + ch;
                output[newIdx] = (unsigned char)(fmin(fmax(input[oldIdx] * 255.0, 0.0), 255.0) + 0.5);
            }
        }
    }
    return output;
}
