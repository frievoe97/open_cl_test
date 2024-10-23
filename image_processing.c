#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb_image.h"
#include "./stb_image_write.h"

// OpenCL Kernel zur RGB -> YCbCr Konvertierung
const char *kernelSourceYCbCr = 
"__kernel void rgb_to_ycbcr(__global unsigned char *inputImage, __global unsigned char *outputImage, int width, int height) { \
    int id = get_global_id(0); \
    int x = id % width; \
    int y = id / width; \
    int idx = (y * width + x) * 3; \
    float R = inputImage[idx] / 255.0f; \
    float G = inputImage[idx + 1] / 255.0f; \
    float B = inputImage[idx + 2] / 255.0f; \
    unsigned char Y = (unsigned char)(16 + (65.481 * R + 128.553 * G + 24.966 * B)); \
    unsigned char Cb = (unsigned char)(128 + (-37.797 * R - 74.203 * G + 112.0 * B)); \
    unsigned char Cr = (unsigned char)(128 + (112.0 * R - 93.786 * G - 18.214 * B)); \
    outputImage[idx] = Y; \
    outputImage[idx + 1] = Cb; \
    outputImage[idx + 2] = Cr; \
}";

// OpenCL Kernel zur Dilatation
const char *kernelSourceDilation = 
"__kernel void dilate(__global unsigned char *inputImage, __global unsigned char *outputImage, int width, int height) { \
    int id = get_global_id(0); \
    int x = id % width; \
    int y = id / width; \
    int idx = (y * width + x) * 3; \
    unsigned char maxVal = 0; \
    for (int dy = -1; dy <= 1; dy++) { \
        for (int dx = -1; dx <= 1; dx++) { \
            int nx = x + dx; \
            int ny = y + dy; \
            if (nx >= 0 && ny >= 0 && nx < width && ny < height) { \
                int neighborIdx = (ny * width + nx) * 3; \
                maxVal = max(maxVal, inputImage[neighborIdx]); \
            } \
        } \
    } \
    outputImage[idx] = maxVal; \
    outputImage[idx + 1] = maxVal; \
    outputImage[idx + 2] = maxVal; \
}";

void checkError(cl_int ret, const char* operation) {
    if (ret != CL_SUCCESS) {
        printf("Error during operation '%s': %d\n", operation, ret);
        exit(1);
    }
}

int main() {
    int width, height, channels;
    unsigned char *inputImage = stbi_load("./image.png", &width, &height, &channels, 3);
    if (inputImage == NULL) {
        printf("Error loading image.\n");
        return 1;
    }

    unsigned char *outputYCbCr = (unsigned char *)malloc(width * height * 3);
    unsigned char *outputDilated = (unsigned char *)malloc(width * height * 3);

    // Plattform- und Geräteauswahl
    cl_platform_id platform;
    cl_device_id device;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform, &ret_num_platforms);
    checkError(ret, "clGetPlatformIDs");
    ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &ret_num_devices);
    checkError(ret, "clGetDeviceIDs");

    // Kontext und Warteschlange erstellen
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &ret);
    checkError(ret, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &ret);
    checkError(ret, "clCreateCommandQueue");

    // Speicherpuffer erstellen
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 3, inputImage, &ret);
    cl_mem outputBufferYCbCr = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * 3, NULL, &ret);
    cl_mem outputBufferDilated = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * 3, NULL, &ret);
    checkError(ret, "clCreateBuffer");

    // Programm und Kernel erstellen für YCbCr
    cl_program programYCbCr = clCreateProgramWithSource(context, 1, (const char **)&kernelSourceYCbCr, NULL, &ret);
    checkError(ret, "clCreateProgramWithSource YCbCr");
    ret = clBuildProgram(programYCbCr, 1, &device, NULL, NULL, NULL);
    checkError(ret, "clBuildProgram YCbCr");
    cl_kernel kernelYCbCr = clCreateKernel(programYCbCr, "rgb_to_ycbcr", &ret);
    checkError(ret, "clCreateKernel YCbCr");

    // Kernel-Argumente setzen und ausführen
    ret = clSetKernelArg(kernelYCbCr, 0, sizeof(cl_mem), &inputBuffer);
    ret |= clSetKernelArg(kernelYCbCr, 1, sizeof(cl_mem), &outputBufferYCbCr);
    ret |= clSetKernelArg(kernelYCbCr, 2, sizeof(int), &width);
    ret |= clSetKernelArg(kernelYCbCr, 3, sizeof(int), &height);
    checkError(ret, "clSetKernelArg YCbCr");

    size_t globalSize = width * height;
    ret = clEnqueueNDRangeKernel(queue, kernelYCbCr, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    checkError(ret, "clEnqueueNDRangeKernel YCbCr");
    clFinish(queue);

    // Ergebnisse zurücklesen
    ret = clEnqueueReadBuffer(queue, outputBufferYCbCr, CL_TRUE, 0, width * height * 3, outputYCbCr, 0, NULL, NULL);
    checkError(ret, "clEnqueueReadBuffer YCbCr");

    // Programm und Kernel erstellen für Dilatation
    cl_program programDilation = clCreateProgramWithSource(context, 1, (const char **)&kernelSourceDilation, NULL, &ret);
    checkError(ret, "clCreateProgramWithSource Dilation");
    ret = clBuildProgram(programDilation, 1, &device, NULL, NULL, NULL);
    checkError(ret, "clBuildProgram Dilation");
    cl_kernel kernelDilation = clCreateKernel(programDilation, "dilate", &ret);
    checkError(ret, "clCreateKernel Dilation");

    // Kernel-Argumente setzen und ausführen
    ret = clSetKernelArg(kernelDilation, 0, sizeof(cl_mem), &outputBufferYCbCr);
    ret |= clSetKernelArg(kernelDilation, 1, sizeof(cl_mem), &outputBufferDilated);
    ret |= clSetKernelArg(kernelDilation, 2, sizeof(int), &width);
    ret |= clSetKernelArg(kernelDilation, 3, sizeof(int), &height);
    checkError(ret, "clSetKernelArg Dilation");

    ret = clEnqueueNDRangeKernel(queue, kernelDilation, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    checkError(ret, "clEnqueueNDRangeKernel Dilation");
    clFinish(queue);

    // Ergebnisse zurücklesen
    ret = clEnqueueReadBuffer(queue, outputBufferDilated, CL_TRUE, 0, width * height * 3, outputDilated, 0, NULL, NULL);
    checkError(ret, "clEnqueueReadBuffer Dilation");

    // Speichern der Ausgabebilder
    stbi_write_png("output_ycbcr.png", width, height, 3, outputYCbCr, width * 3);
    stbi_write_png("output_dilated.png", width, height, 3, outputDilated, width * 3);

    // Ressourcen freigeben
    clReleaseKernel(kernelYCbCr);
    clReleaseKernel(kernelDilation);
    clReleaseProgram(programYCbCr);
    clReleaseProgram(programDilation);
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBufferYCbCr);
    clReleaseMemObject(outputBufferDilated);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    stbi_image_free(inputImage);
    free(outputYCbCr);
    free(outputDilated);

    return 0;
}
