# OpenCL Image Processing

## Overview

This program processes images using three modes of parallelization:
1. **Non-parallel**: Sequential processing.
2. **OpenCV-based parallelization**: Utilizes OpenCV library functions.
3. **OpenCL-based parallelization**: Processes images with OpenCL kernels using three local work sizes (256, 128, 64).

The program converts the input image from RGB to YCbCr and applies a dilation operation. Results are saved as separate images for each mode and configuration.

## Requirements

- **OpenCL**: Version 1.2 or later is required.
- **OpenCV**: The library must be installed.
- **C++ Compiler**: Supports C++17 or later.
- **CMake**: For building the project.

## Main File

The main entry point for the program is `main.cpp`.

## Build Instructions

This project uses CMake for build configuration and supports multiple build systems like Ninja and Make.

### Steps to Build

1. **Install Dependencies**
   Ensure OpenCV and OpenCL are installed and available in your environment. Verify installation paths or use a package manager to install them.

2. **Navigate to the Project Directory**
   ```bash
   cd /path/to/project
   ```

3. **Create a Build Directory**
   ```bash
   mkdir build && cd build
   ```

4. **Run CMake**
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   ```

  - Replace `Debug` with `Release` for an optimized build.
  - Ensure `OpenCL` and `OpenCV` are detected during this step. If not, set environment variables like `OpenCV_DIR` or `OpenCL_INCLUDE_DIR` to the appropriate paths.

5. **Compile the Project**
   ```bash
   make
   ```

   Or, if you are using Ninja:
   ```bash
   ninja
   ```

## Execution

The program processes all `.png`, `.jpg`, and `.jpeg` images in a specified directory. It supports three modes:

1. `non-parallel`
2. `opencv`
3. `opencl` (with local work sizes: 256, 128, and 64)

### Example Usage

```bash
./open_cl_test <basepath> <outputpath>
```

- `<basepath>`: Path to the directory containing input images.
- `<outputpath>`: Path to the directory where output images and results will be saved.

## Output

For each image processed, the program generates:
1. YCbCr output images.
2. Dilation output images.

Outputs are saved for each mode and local size combination in the specified `<outputpath>` directory.

## CSV Results

The program generates a `results.csv` file in `<outputpath>` with the following columns:

- **InputFile**: Path to the input image.
- **Mode**: The processing mode (`non-parallel`, `opencv`, or `opencl` with local sizes 256, 128, 64).
- **YCbCrOutputFile**: Path to the YCbCr output image.
- **DilatedOutputFile**: Path to the dilation output image.
- **ExecutionTimeSeconds**: Time taken to process the image.


---

## Notes

- This project includes three OpenCL local sizes (256, 128, 64) to test performance across configurations. Optimal size may vary depending on the hardware.
- Performance results are logged in `results.csv` for comparison.
