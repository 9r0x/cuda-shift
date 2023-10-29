import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy
from tqdm.auto import tqdm

def make_kernel(kernel_size, sigma, channels):
    kernel_1d = scipy.signal.windows.gaussian(kernel_size, sigma)
    kernel = kernel_1d.reshape(-1, 1) * kernel_1d
    kernel /= kernel.sum()
    return np.repeat(kernel[..., np.newaxis], channels, axis=-1)
    
def convert_to_toy_view(image, cutoff_ratio=0.5, kernel_size=39):
    assert cutoff_ratio <= 1
    height, width, channels = image.shape
    height_cutoff_0, height_cutoff_1 = int(height / 2 - height*cutoff_ratio/2), int(height / 2 + height*cutoff_ratio/2)
    # split the image into three parts
    image_1 = image[:height_cutoff_0]
    image_2 = image[height_cutoff_0:height_cutoff_1]
    image_3 = image[height_cutoff_1:]
    
    focus_pos = height // 2
    new_height_1 = height_cutoff_0-kernel_size+1
    new_height_2 = height_cutoff_1-height_cutoff_0
    new_height_3 = height-height_cutoff_1-kernel_size+1
    new_width = width-kernel_size+1
    
    new_image = np.empty([new_height_1+new_height_2+new_height_3, new_width, channels])
    # calculating the first part
    for i in tqdm(range(new_height_1)):
        ratio = 20.0*np.abs(i-focus_pos) / focus_pos + 0.2
        kernel = make_kernel(kernel_size, ratio, channels)
        for j in range(new_width):
            patch = image_1[i:i+kernel_size, j:j+kernel_size]    
            new_image[i, j] = np.sum(patch*kernel, axis=(0, 1))
    # calculating the second part
    new_image[new_height_1:new_height_1+new_height_2] = image_2[:, kernel_size//2-1:-(kernel_size-kernel_size//2)]
    # calculating the third part
    for i in tqdm(range(new_height_3)):
        ratio = 20.0*np.abs(i-focus_pos) / focus_pos + 0.2
        kernel = make_kernel(kernel_size, ratio, channels)
        for j in range(new_width):
            patch = image_3[i:i+kernel_size, j:j+kernel_size]
            new_image[-new_height_3+i, j] = np.sum(patch*kernel, axis=(0, 1))

    return new_image

if __name__ == "__main__":
    image_path = ""
    image = Image.open(image_path)
    new_image = convert_to_toy_view(image, kernel_size=100)
    plt.imshow(new_image)
    plt.show()