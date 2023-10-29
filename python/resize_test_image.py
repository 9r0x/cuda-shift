#!/usr/bin/env python
from PIL import Image
import argparse


def resize_image_by_width(input_path, output_path, base_width):
    # Open an image file
    with Image.open(input_path) as img:
        # Calculate the height keeping the aspect ratio constant
        w_percent = (base_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        # Resize image
        img_resized = img.resize((base_width, h_size))
        # Save image
        img_resized.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='input image path')
    parser.add_argument('output_path', type=str, help='output image path')
    parser.add_argument('output_width', type=int, help='output image width')
    args = parser.parse_args()

    resize_image_by_width(args.input_path, args.output_path, args.output_width)