#!/usr/bin/env python
import numpy as np
import argparse


def gaussian_kernel(r, sigma):
    dim = 2 * r - 1
    kernel = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            x = i - r + 1
            y = j - r + 1
            kernel[i, j] = np.exp(-(x**2 + y**2) /
                                  (2 * sigma**2)) / (2 * np.pi * sigma**2)

    return kernel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('radius',
                        type=int,
                        help='radius of kernel, center included')
    parser.add_argument('sigma', type=float, help='sigma of gaussian kernel')
    parser.add_argument('--normalize',
                        '-n',
                        action='store_true',
                        help='normalize kernel')
    args = parser.parse_args()
    kernel = gaussian_kernel(args.radius, args.sigma)
    if args.normalize:
        kernel = kernel / np.sum(kernel)
    print(kernel)