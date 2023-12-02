#include <cuda.h>
#include <stdio.h>

#include "kernel.h"
#include "image.h"
#include "conv.h"
#include "utils.h"

#define FULL_MASK 0xffffffff
#define __MAXWARPS_PER_BLOCK__ (__MAXLANES_PER_BLOCK__ / __LANES_PER_WARP__)

#define BLOCK_DIM 32
#define __MAXLANES_PER_BLOCK__ (BLOCK_DIM * BLOCK_DIM)
#define __LANES_PER_WARP__ 32 // Nvidia defines this as 32

__host__ void printUsage(const char *programName)
{
    fprintf(stderr, "Usage: %s <input_path> <output_path>\n", programName);
    fprintf(stderr, "Arguments:\n");
    fprintf(stderr, "  <input_path>    Path to the input image. Supported formats are PNG and JPEG(JPG).\n");
    fprintf(stderr, "  <output_path>   Path to the output image. Supported formats are PNG and JPEG(JPG).\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --help         -h         Print this help message.\n");
    fprintf(stderr, "  --radius       -r = 10    Dimension of the Gaussian kernel radius, center pixel included.\n");
    fprintf(stderr, "  --sigma        -s = 3.0   Standard deviation of the Gaussian kernel. Must be a positive number.\n");
    fprintf(stderr, "  --far          -f = 0.2   Lower end of the far mask in height ratio.\n");
    fprintf(stderr, "  --near         -n = 0.8   Upper end of the near mask in height ratio.\n");
    fprintf(stderr, "  --feather         = 0.1   Feather size in height ratio.\n");
    fprintf(stderr, "\nExample: %s input.png -r 5 - s 1.2 output.png\n", programName);
}

int main(int argc, char **argv)
{

    size_t width, height;
    size_t channels, kernel_dim, near, far, feather;
    size_t kernel_radius = 10;
    double near_ratio = 0.8, far_ratio = 0.2, sigma = 3.0, feather_ratio = 0.1;
    // No. of total bytes on the device
    size_t kernel_bytes, image_bytes, mask_bytes;
    const char *input_path = NULL, *output_path = NULL;
    unsigned char *image;
    double *kernel_h, *in_h, *out_h;
    double *kernel_d, *in_d, *out_d;

    double *near_mask_d, *mid_mask_d, *far_mask_d;
    double *near_out_d, *mid_out_d, *far_out_d;

    cudaStream_t near_stream, mid_stream, far_stream;

    // Although this is a on-host array
    // its elements are pointers to device memory
    double *in_blend_h[3];
    double **in_blend_d;

    if (__glibc_unlikely(argc < 3))
    {
        printUsage(argv[0]);
        return 1;
    }
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            printUsage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--radius") == 0 || strcmp(argv[i], "-r") == 0)
        {
            if (__glibc_likely(i + 1 < argc))
            {
                kernel_radius = atoi(argv[++i]);
                if (__glibc_unlikely(kernel_radius < 1))
                {
                    fprintf(stderr, "[!] Kernel radius must be a positive integer.\n");
                    return 1;
                }
            }
            else
            {
                fprintf(stderr, "[!] Error: Missing value for --radius/-r\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--sigma") == 0 || strcmp(argv[i], "-s") == 0)
        {
            if (__glibc_likely(i + 1 < argc))
            {
                sigma = atof(argv[++i]);
                if (__glibc_unlikely(sigma <= 0))
                {
                    fprintf(stderr, "[!] Sigma must be a positive number.\n");
                    return 1;
                }
            }
            else
            {
                fprintf(stderr, "[!] Error: Missing value for --sigma/-s\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--far") == 0 || strcmp(argv[i], "-f") == 0)
        {
            if (__glibc_likely(i + 1 < argc))
            {
                far_ratio = atof(argv[++i]);
                if (__glibc_unlikely(far_ratio < 0 || far_ratio > 1))
                {
                    fprintf(stderr, "[!] Far ratio must be between 0 and 1.\n");
                    return 1;
                }
            }
            else
            {
                fprintf(stderr, "[!] Error: Missing value for --far/-f\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--near") == 0 || strcmp(argv[i], "-n") == 0)
        {
            if (__glibc_likely(i + 1 < argc))
            {
                near_ratio = atof(argv[++i]);
                if (__glibc_unlikely(near_ratio < 0 || near_ratio > 1))
                {
                    fprintf(stderr, "[!] Near must be between 0 and 1.\n");
                    return 1;
                }
            }
            else
            {
                fprintf(stderr, "[!] Error: Missing value for --near/-n\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--feather") == 0)
        {
            if (__glibc_likely(i + 1 < argc))
            {
                feather_ratio = atof(argv[++i]);
                if (__glibc_unlikely(feather_ratio < 0 || feather_ratio > 1))
                {
                    fprintf(stderr, "[!] Feather size must be between 0 and 1.\n");
                    return 1;
                }
            }
            else
            {
                fprintf(stderr, "[!] Error: Missing value for --feather\n");
                return 1;
            }
        }
        else
        {
            // Handle positional arguments
            if (input_path == NULL)
                input_path = argv[i];
            else if (output_path == NULL)
                output_path = argv[i];
            else
            {
                fprintf(stderr, "[!] Error: Unexpected argument: %s\n", argv[i]);
                return 1;
            }
        }
    }

    // Check if positional arguments are provided
    if (__glibc_unlikely(input_path == NULL || output_path == NULL))
    {
        fprintf(stderr, "[!] Error: Input and output paths are required.\n");
        printUsage(argv[0]);
        return 1;
    }
    if (__glibc_unlikely(near_ratio < far_ratio))
    {
        fprintf(stderr, "[!] Near must be greater than far.\n");
        return 1;
    }

    kernel_dim = (kernel_radius << 1) - 1;
    kernel_bytes = kernel_dim * kernel_dim * sizeof(double);

    printf("\n[ ] Loading image ...\n");
    image = load_image(input_path, &height, &width, &channels);
    if (__glibc_unlikely(!image))
    {
        fprintf(stderr, "[!] Error loading image.\n");
        return 1;
    }
    printf("[+] Image loaded from %s\n", input_path);

    mask_bytes = width * height * sizeof(double);
    image_bytes = mask_bytes * channels;

    far = far_ratio * height;
    near = near_ratio * height;
    feather = feather_ratio * height;

    printf("[*] Kernel dimensions: %d x %d \n", kernel_dim, kernel_dim);
    printf("[*] Sigma: %f\n", sigma);
    printf("[*] far: %d, near: %d\n", far, near);
    printf("[*] Feather radius: %d\n", feather);
    printf("[*] Image dimensions(HWC): %d x %d x %d\n", height, width, channels);

    printf("\n[ ] Generating Gaussian kernel ...\n");
    kernel_h = gen_gaussian_kernel(kernel_radius, sigma);
    mirror_gaussian_kernel(kernel_radius, kernel_h);
    normalize_gaussian_kernel(kernel_radius, kernel_h);

#ifdef DEBUG
    for (auto i = 0; i < kernel_dim; i++)
    {
        for (auto j = 0; j < kernel_dim; j++)
            printf("%f ", kernel_h[i * kernel_dim + j]);
        printf("\n");
    }
#endif
    printf("[+] Kernel generated.\n");

    printf("\n[ ] Converting image to cwh format ...\n");
    in_h = to_chw(image, height, width, channels);
    if (__glibc_unlikely(!in_h))
    {
        free_image(image);
        fprintf(stderr, "[!] Error converting image to cwh format.\n");
        return 1;
    }
    free_image(image);
    printf("[+] Image converted.\n");

    printf("\n[ ] Allocating memory on device ...\n");
    dim3 blockSize(32, 32);
    dim3 gridSize(channels,
                  (height + blockSize.x - 1) / blockSize.x,
                  (width + blockSize.y - 1) / blockSize.y);

    printf("[*] Grid size: %d, %d, %d\n", gridSize.x, gridSize.y, gridSize.z);
    printf("[*] Block size: %d, %d\n", blockSize.x, blockSize.y);

    catch_error(cudaStreamCreate(&near_stream));
    catch_error(cudaStreamCreate(&mid_stream));
    catch_error(cudaStreamCreate(&far_stream));

    catch_error(cudaMallocHost((void **)&out_h, image_bytes));

    catch_error(cudaMalloc((void **)&kernel_d, kernel_bytes));
    catch_error(cudaMalloc((void **)&in_d, image_bytes));
    catch_error(cudaMalloc((void **)&out_d, image_bytes));

    catch_error(cudaMemcpy(kernel_d, kernel_h, kernel_bytes, cudaMemcpyHostToDevice));
    catch_error(cudaMemcpy(in_d, in_h, image_bytes, cudaMemcpyHostToDevice));

    catch_error(cudaMalloc((void **)&near_mask_d, mask_bytes));
    catch_error(cudaMalloc((void **)&mid_mask_d, mask_bytes));
    catch_error(cudaMalloc((void **)&far_mask_d, mask_bytes));

    catch_error(cudaMalloc((void **)&near_out_d, image_bytes));
    catch_error(cudaMalloc((void **)&mid_out_d, image_bytes));
    catch_error(cudaMalloc((void **)&far_out_d, image_bytes));
    printf("[+] Memory allocated on device.\n");

    printf("\n[ ] Generating masks ...\n");
    gen_rectangle_mask<<<gridSize, blockSize, 0, far_stream>>>(height, width,
                                                               0, far, 0, 0,
                                                               far_mask_d, feather);
    gen_rectangle_mask<<<gridSize, blockSize, 0, mid_stream>>>(height, width,
                                                               far + feather, near, 0, 0,
                                                               mid_mask_d, feather);
    gen_rectangle_mask<<<gridSize, blockSize, 0, near_stream>>>(height, width,
                                                                near + feather, 0, 0, 0,
                                                                near_mask_d, feather);

#ifdef DEBUG
    // sum masks and all should be 1
    catch_error(cudaDeviceSynchronize());
    sum_masks<<<gridSize, blockSize>>>(height, width,
                                       near_mask_d, mid_mask_d, far_mask_d);
    catch_error(cudaDeviceSynchronize());
#endif

    printf("\n[ ] Applying convolution ...\n");
    conv2D<<<gridSize, blockSize, 0, far_stream>>>(in_d, far_mask_d, far_out_d,
                                                   height, width,
                                                   kernel_d, kernel_radius);
    conv2D<<<gridSize, blockSize, 0, mid_stream>>>(in_d, mid_mask_d, mid_out_d,
                                                   height, width,
                                                   NULL, 1);
    conv2D<<<gridSize, blockSize, 0, near_stream>>>(in_d, near_mask_d, near_out_d,
                                                    height, width,
                                                    kernel_d, kernel_radius);
    catch_error(cudaDeviceSynchronize());
    printf("[+] Masks generated.\n");
    printf("[+] Convolution applied.\n");

    printf("\n[ ] Combining images ...\n");
    in_blend_h[0] = near_out_d;
    in_blend_h[1] = mid_out_d;
    in_blend_h[2] = far_out_d;

    catch_error(cudaMalloc((void **)&in_blend_d, 3 * sizeof(double *)));
    catch_error(cudaMemcpy(in_blend_d, in_blend_h,
                           3 * sizeof(double *), cudaMemcpyHostToDevice));

    blend<<<gridSize, blockSize>>>(in_blend_d, out_d, height, width, 3);

    catch_error(cudaDeviceSynchronize());
    catch_error(cudaMemcpy(out_h, out_d, image_bytes, cudaMemcpyDeviceToHost));
    printf("[+] Images combined.\n");

    printf("\n[ ] Saving image ...\n");
    // Reusing freed image pointer
    image = to_hwc(out_h, height, width, channels);
    // stb returns 0 on failure
    if (__glibc_unlikely(!save_image(output_path, image, height, width, channels)))
    {
        free_image(image);
        fprintf(stderr, "[!] Error saving image.\n");
        return 1;
    }
    else
    {
        free_image(image);
        printf("[+] Image saved to %s\n", output_path);
    }

    catch_error(cudaFreeHost(kernel_h));
    catch_error(cudaFreeHost(in_h));
    catch_error(cudaFreeHost(out_h));

    catch_error(cudaFree(kernel_d));
    catch_error(cudaFree(in_blend_d));
    catch_error(cudaFree(near_mask_d));
    catch_error(cudaFree(mid_mask_d));
    catch_error(cudaFree(far_mask_d));
    catch_error(cudaFree(near_out_d));
    catch_error(cudaFree(mid_out_d));
    catch_error(cudaFree(far_out_d));
    catch_error(cudaFree(in_d));
    catch_error(cudaFree(out_d));
    return 0;
}
