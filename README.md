# OpenCL Image Processing

## Overview

This program reads an image, converts it from RGB to YCbCr, and then performs a dilation operation on the image. The program uses OpenCL for parallelization of these steps and produces two output images.

## Requirements

- OpenCL must be installed on your system (macOS uses `OpenCL.framework`).
- The `stb_image` and `stb_image_write` libraries are used for image loading and saving. These files should be located in the same directory as the `image_processing.c` file.

## Compilation

To compile the program, use the following command:

```bash
gcc -o image_processing image_processing.c -framework OpenCL -I.
```

- `-framework OpenCL`: Uses the OpenCL framework (specific for macOS).
- `-I.`: Instructs the compiler to search for header files in the current directory (needed for `stb_image.h` and `stb_image_write.h`).

## Execution

To run the program, make sure there is an image named `image.png` in the same directory as the executable. Then run:

```bash
./image_processing
```

## Output

The program produces two output images:

1. **output_ycbcr.png**: The image after the RGB-to-YCbCr conversion.
2. **output_dilated.png**: The image after the dilation operation.

## Debugging

- If you encounter the error "Error loading image.", make sure:
  - The image is in the same directory as the executable.
  - The filename and extension are correct (`image.png`).
  - The image format is supported (e.g., PNG).

## Resources and Cleanup

The program creates OpenCL kernels and buffers for processing the image. At the end, all resources are released to avoid memory leaks.
