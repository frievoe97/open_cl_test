/**
 * @def CL_HPP_ENABLE_EXCEPTIONS
 * @brief Enables exceptions for OpenCL C++ bindings.
 */
#define CL_HPP_ENABLE_EXCEPTIONS
/**
 * @def CL_HPP_TARGET_OPENCL_VERSION
 * @brief Sets the target OpenCL version.
 */
#define CL_HPP_TARGET_OPENCL_VERSION 120
/**
 * @def CL_HPP_MINIMUM_OPENCL_VERSION
 * @brief Sets the minimum OpenCL version.
 */
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

#if __has_include(<CL/opencl.hpp>)
    #include <CL/opencl.hpp>
#elif __has_include(<OpenCL/opencl.h>)
    #include <OpenCL/opencl.h>
#else
    #error "No suitable OpenCL header found. Please install OpenCL or adjust include paths."
#endif

/**
 * @brief Checks for OpenCL errors and prints an error message if necessary.
 * @param ret The return code from an OpenCL operation.
 * @param operation The name of the operation being checked.
 */
void checkError(cl_int ret, const char* operation) {
    if (ret != CL_SUCCESS) {
        std::cerr << "Error during operation '" << operation << "': " << ret << std::endl;
        exit(1);
    }
}

/**
 * @brief OpenCL kernel source code for RGB to YCbCr conversion.
 */
const char* kernelSourceYCbCr = R"(
__kernel void rgb_to_ycbcr(__global unsigned char* inputImage, __global unsigned char* outputImage, int width, int height, int channels) {
    int id = get_global_id(0);
    int x = id % width;
    int y = id / width;
    int idx = (y * width + x) * channels;

    float R = inputImage[idx] / 255.0f;
    float G = inputImage[idx + 1] / 255.0f;
    float B = inputImage[idx + 2] / 255.0f;

    unsigned char Y = (unsigned char)(16 + (65.481 * R + 128.553 * G + 24.966 * B));
    unsigned char Cb = (unsigned char)(128 + (-37.797 * R - 74.203 * G + 112.0 * B));
    unsigned char Cr = (unsigned char)(128 + (112.0 * R - 93.786 * G - 18.214 * B));

    outputImage[idx] = Y;
    outputImage[idx + 1] = Cb;
    outputImage[idx + 2] = Cr;
}
)";

/**
 * @brief OpenCL kernel source code for dilation operation.
 */
const char* kernelSourceDilation = R"(
__kernel void dilate(__global unsigned char* inputImage, __global unsigned char* outputImage, int width, int height) {
    int id = get_global_id(0);

    int x = id % width;
    int y = id / width;

    if (x >= width || y >= height) return;

    unsigned char maxVal = 0;

    for (int dy = -3; dy <= 3; dy++) {
        for (int dx = -3; dx <= 3; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                int neighborIdx = ny * width + nx;
                maxVal = max(maxVal, inputImage[neighborIdx]);
            }
        }
    }

    int idx = y * width + x;
    outputImage[idx] = maxVal;
}
)";

/**
 * @brief OpenCL kernel source code for 2D RGB to YCbCr conversion.
 */
const char* kernelSourceYCbCr2D = R"(
__kernel void rgb_to_ycbcr(__global unsigned char* inputImage, __global
                    unsigned char* outputImage, int width, int height, int channels) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    int idx = (y * width + x) * channels;

    float R = inputImage[idx] / 255.0f;
    float G = inputImage[idx + 1] / 255.0f;
    float B = inputImage[idx + 2] / 255.0f;

    unsigned char Y = (unsigned char)(16 + (65.481 * R + 128.553 * G + 24.966 * B));
    unsigned char Cb = (unsigned char)(128 + (-37.797 * R - 74.203 * G + 112.0 * B));
    unsigned char Cr = (unsigned char)(128 + (112.0 * R - 93.786 * G - 18.214 * B));

    outputImage[idx] = Y;
    outputImage[idx + 1] = Cb;
    outputImage[idx + 2] = Cr;
}
)";

/**
 * @brief OpenCL kernel source code for 2D dilation operation.
 */
const char* kernelSourceDilation2D = R"(
__kernel void dilate(__global unsigned char* inputImage, __global unsigned char* outputImage, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int idx = y * width + x; // Single-channel index
    unsigned char maxVal = 0;

    for (int dy = -3; dy <= 3; dy++) {
        for (int dx = -3; dx <= 3; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                int neighborIdx = ny * width + nx;
                maxVal = max(maxVal, inputImage[neighborIdx]);
            }
        }
    }

    outputImage[idx] = maxVal;
}
)";


cl_context context = nullptr;
cl_command_queue queue = nullptr;
cl_program programYCbCr = nullptr;
cl_program programYCbCr2D = nullptr;
cl_program programDilation = nullptr;
cl_program programDilation2D = nullptr;
cl_kernel kernelYCbCr = nullptr;
cl_kernel kernelYCbCr2D = nullptr;
cl_kernel kernelDilation = nullptr;
cl_kernel kernelDilation2D = nullptr;

/**
 * @brief Initializes OpenCL context, command queue, and compiles kernels.
 */
void initializeOpenCL() {
    cl_platform_id platform;
    cl_device_id device;
    cl_int ret;

    // Initialize OpenCL platform and device
    ret = clGetPlatformIDs(1, &platform, NULL);
    checkError(ret, "clGetPlatformIDs");
    ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    checkError(ret, "clGetDeviceIDs");

    // Create OpenCL context and command queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &ret);
    checkError(ret, "clCreateContext");
    queue = clCreateCommandQueue(context, device, 0, &ret);
    checkError(ret, "clCreateCommandQueue");

    // Compile YCbCr program
    programYCbCr = clCreateProgramWithSource(context, 1, &kernelSourceYCbCr, NULL, &ret);
    checkError(ret, "clCreateProgramWithSource YCbCr");
    ret = clBuildProgram(programYCbCr, 1, &device, NULL, NULL, NULL);
    checkError(ret, "clBuildProgram YCbCr");
    kernelYCbCr = clCreateKernel(programYCbCr, "rgb_to_ycbcr", &ret);
    checkError(ret, "clCreateKernel YCbCr");

    // Compile 2D YCbCr program
    programYCbCr2D = clCreateProgramWithSource(context, 1, &kernelSourceYCbCr2D, NULL, &ret);
    checkError(ret, "clCreateProgramWithSource 2D YCbCr");
    ret = clBuildProgram(programYCbCr2D, 1, &device, NULL, NULL, NULL);
    checkError(ret, "clBuildProgram 2D CbCr");
    kernelYCbCr2D = clCreateKernel(programYCbCr2D, "rgb_to_ycbcr", &ret);
    checkError(ret, "clCreateKernel 2D YCbCr");

    // Compile Dilation program
    programDilation = clCreateProgramWithSource(context, 1, &kernelSourceDilation, NULL, &ret);
    checkError(ret, "clCreateProgramWithSource Dilation");
    ret = clBuildProgram(programDilation, 1, &device, NULL, NULL, NULL);
    checkError(ret, "clBuildProgram Dilation");
    kernelDilation = clCreateKernel(programDilation, "dilate", &ret);
    checkError(ret, "clCreateKernel Dilation");

    // Compile Dilation program
    programDilation2D = clCreateProgramWithSource(context, 1, &kernelSourceDilation2D, NULL, &ret);
    checkError(ret, "clCreateProgramWithSource2D Dilation");
    ret = clBuildProgram(programDilation2D, 1, &device, NULL, NULL, NULL);
    checkError(ret, "clBuildProgram 2D Dilation");
    kernelDilation2D = clCreateKernel(programDilation2D, "dilate", &ret);
    checkError(ret, "clCreateKernel2D Dilation");

    size_t maxLocalSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxLocalSize, NULL);
    std::cout << "Maximum Local Size: " << maxLocalSize << std::endl;

    size_t kernelMaxWorkGroupSize;
    clGetKernelWorkGroupInfo(kernelYCbCr, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernelMaxWorkGroupSize, NULL);
    std::cout << "Kernel Max Work Group Size: " << kernelMaxWorkGroupSize << std::endl;

}

/**
 * @brief Releases OpenCL resources.
 */
void releaseOpenCL() {
    clReleaseKernel(kernelYCbCr);
    clReleaseKernel(kernelYCbCr2D);
    clReleaseKernel(kernelDilation);
    clReleaseKernel(kernelDilation2D);
    clReleaseProgram(programYCbCr);
    clReleaseProgram(programYCbCr2D);
    clReleaseProgram(programDilation);
    clReleaseProgram(programDilation2D);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

/**
 * @brief Processes an image using OpenCL for RGB to YCbCr conversion and dilation.
 *        Supports both 1D and 2D workgroup execution.
 *
 * @param inputImage The input image.
 * @param grayscaleImage The output grayscale image.
 * @param outputYCbCr The output image in YCbCr format.
 * @param outputDilated The output image after dilation.
 * @param use2DMode Boolean flag to select between 1D and 2D processing modes.
 * @param localSizeX The local work size for the X dimension (used in 2D mode).
 * @param localSizeY The local work size for the Y dimension (used in 2D mode).
 * @param localSize The local work size for 1D processing.
 */
void processWithOpenCL(const cv::Mat& inputImage, cv::Mat& grayscaleImage, cv::Mat& outputYCbCr, cv::Mat& outputDilated,
                       bool use2DMode = true, size_t localSizeX = 16, size_t localSizeY = 16, size_t localSize = 0) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();

    std::vector<unsigned char> inputData(inputImage.data, inputImage.data + inputImage.total() * channels);
    std::vector<unsigned char> outputDataYCbCr(width * height * channels);
    std::vector<unsigned char> outputDataDilated(width * height);

    cl_int ret;
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputData.size(), inputData.data(), &ret);
    checkError(ret, "clCreateBuffer Input");
    cl_mem outputBufferYCbCr = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outputDataYCbCr.size(), NULL, &ret);
    checkError(ret, "clCreateBuffer Output YCbCr");

    // Kernel argument setting
    cl_kernel lkernelYCbCr = use2DMode ? kernelYCbCr2D : kernelYCbCr;
    ret = clSetKernelArg(lkernelYCbCr, 0, sizeof(cl_mem), &inputBuffer);
    ret |= clSetKernelArg(lkernelYCbCr, 1, sizeof(cl_mem), &outputBufferYCbCr);
    ret |= clSetKernelArg(lkernelYCbCr, 2, sizeof(int), &width);
    ret |= clSetKernelArg(lkernelYCbCr, 3, sizeof(int), &height);
    ret |= clSetKernelArg(lkernelYCbCr, 4, sizeof(int), &channels);
    checkError(ret, "clSetKernelArg YCbCr");

    size_t globalSize1D = width * height;
    size_t globalSize2D[2] = { (width + localSizeX - 1) / localSizeX * localSizeX, (height + localSizeY - 1) / localSizeY * localSizeY };
    size_t localSize2D[2] = { localSizeX, localSizeY };

    if (use2DMode) {
        ret = clEnqueueNDRangeKernel(queue, lkernelYCbCr, 2, NULL, globalSize2D, localSize2D, 0, NULL, NULL);
    } else {
        if (localSize > 0 && globalSize1D % localSize != 0) {
            globalSize1D = ((globalSize1D / localSize) + 1) * localSize;
        }
        ret = clEnqueueNDRangeKernel(queue, lkernelYCbCr, 1, NULL, &globalSize1D, localSize ? &localSize : NULL, 0, NULL, NULL);
    }
    checkError(ret, "clEnqueueNDRangeKernel YCbCr");
    clFinish(queue);

    ret = clEnqueueReadBuffer(queue, outputBufferYCbCr, CL_TRUE, 0, outputDataYCbCr.size(), outputDataYCbCr.data(), 0, NULL, NULL);
    checkError(ret, "clEnqueueReadBuffer YCbCr");

    // Prepare grayscale data for dilation
    std::vector<unsigned char> grayscaleData(grayscaleImage.data, grayscaleImage.data + grayscaleImage.total());
    cl_mem grayscaleBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, grayscaleData.size(), grayscaleData.data(), &ret);
    checkError(ret, "clCreateBuffer Grayscale");

    cl_mem outputBufferDilated = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outputDataDilated.size(), NULL, &ret);
    checkError(ret, "clCreateBuffer Output Dilated");

    cl_kernel lkernelDilation = use2DMode ? kernelDilation2D : kernelDilation;
    ret = clSetKernelArg(lkernelDilation, 0, sizeof(cl_mem), &grayscaleBuffer);
    ret |= clSetKernelArg(lkernelDilation, 1, sizeof(cl_mem), &outputBufferDilated);
    ret |= clSetKernelArg(lkernelDilation, 2, sizeof(int), &width);
    ret |= clSetKernelArg(lkernelDilation, 3, sizeof(int), &height);
    checkError(ret, "clSetKernelArg Dilation");

    if (use2DMode) {
        ret = clEnqueueNDRangeKernel(queue, lkernelDilation, 2, NULL, globalSize2D, localSize2D, 0, NULL, NULL);
    } else {
        ret = clEnqueueNDRangeKernel(queue, lkernelDilation, 1, NULL, &globalSize1D, localSize ? &localSize : NULL, 0, NULL, NULL);
    }
    checkError(ret, "clEnqueueNDRangeKernel Dilation");
    clFinish(queue);

    ret = clEnqueueReadBuffer(queue, outputBufferDilated, CL_TRUE, 0, outputDataDilated.size(), outputDataDilated.data(), 0, NULL, NULL);
    checkError(ret, "clEnqueueReadBuffer Dilation");

    // Output results
    outputYCbCr = cv::Mat(height, width, inputImage.type(), outputDataYCbCr.data()).clone();
    outputDilated = cv::Mat(height, width, CV_8UC1, outputDataDilated.data()).clone();

    // Release OpenCL resources
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBufferYCbCr);
    clReleaseMemObject(grayscaleBuffer);
    clReleaseMemObject(outputBufferDilated);
}


/**
 * @brief Converts an image from RGB to YCbCr using a non-parallel implementation.
 * @param inputImage The input image.
 * @param outputImage The output image in YCbCr format.
 */
void rgbToYCbCrNonParallel(const cv::Mat& inputImage, cv::Mat& outputImage) {
    CV_Assert(inputImage.channels() == 3);
    outputImage = cv::Mat(inputImage.size(), inputImage.type());
    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            const cv::Vec3b& pixel = inputImage.at<cv::Vec3b>(y, x);
            float R = pixel[0] / 255.0f;
            float G = pixel[1] / 255.0f;
            float B = pixel[2] / 255.0f;

            unsigned char Y = static_cast<unsigned char>(16 + (65.481 * R + 128.553 * G + 24.966 * B));
            unsigned char Cb = static_cast<unsigned char>(128 + (-37.797 * R - 74.203 * G + 112.0 * B));
            unsigned char Cr = static_cast<unsigned char>(128 + (112.0 * R - 93.786 * G - 18.214 * B));

            outputImage.at<cv::Vec3b>(y, x) = cv::Vec3b(Y, Cb, Cr);
        }
    }
}

/**
 * @brief Applies dilation to an image using a non-parallel implementation.
 * @param inputImage The input image.
 * @param outputImage The dilated output image.
 */
void dilateNonParallel(const cv::Mat& inputImage, cv::Mat& outputImage) {

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Prepare the output image
    outputImage = cv::Mat(inputImage.size(), inputImage.type());

    // Access the input and output image data
    const unsigned char* inputData = inputImage.data;
    unsigned char* outputData = outputImage.data;

    // Process each pixel
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char maxVal = 0;

            // Iterate over the 7x7 neighborhood
            for (int dy = -3; dy <= 3; ++dy) {
                for (int dx = -3; dx <= 3; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;

                    // Check boundaries
                    if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                        int neighborIdx = ny * width + nx; // Index for the neighbor pixel
                        maxVal = std::max(maxVal, inputData[neighborIdx]);
                    }
                }
            }

            // Assign the max value to the current pixel in the output image
            outputData[y * width + x] = maxVal;
        }
    }
}



/**
 * @brief Converts an image from RGB to YCbCr using OpenCV.
 * @param inputImage The input image.
 * @param outputImage The output image in YCbCr format.
 */
void rgbToYCbCrOpenCV(const cv::Mat& inputImage, cv::Mat& outputImage) {
    CV_Assert(inputImage.channels() == 3);
    cv::cvtColor(inputImage, outputImage, cv::COLOR_BGR2YCrCb);
}

/**
 * @brief Applies dilation to an image using OpenCV.
 * @param inputImage The input image.
 * @param outputImage The dilated output image.
 */
void dilateOpenCV(const cv::Mat& inputImage, cv::Mat& outputImage) {

    // Create a structuring element
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));

    // Apply dilation on the grayscale image
    cv::Mat dilatedGray;
    cv::dilate(inputImage, dilatedGray, kernel);

    outputImage = dilatedGray;
}

/**
 * @brief Processes an image in a specific mode (non-parallel, OpenCV, or OpenCL).
 * @param mode The processing mode.
 * @param inputImage The input image.
 * @param outputYCbCr The output image in YCbCr format.
 * @param outputDilated The dilated output image.
 * @param elapsedTime The time taken to process the image.
 */
void processImageInMode(const std::string& mode, const cv::Mat& inputImage,
                        cv::Mat& outputYCbCr, cv::Mat& outputDilated, double& elapsedTime) {

    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    auto start = std::chrono::high_resolution_clock::now();

    if (mode == "non-parallel") {
        rgbToYCbCrNonParallel(inputImage, outputYCbCr);
        dilateNonParallel(grayImage, outputDilated);
    } else if (mode == "opencl16x16") {
        processWithOpenCL(inputImage, grayImage, outputYCbCr,
                            outputDilated, true, 16, 16);
    } else if (mode == "opencl8x8") {
        processWithOpenCL(inputImage, grayImage, outputYCbCr,
                            outputDilated, true, 8, 8);
    } else if (mode == "opencl2x128") {
        processWithOpenCL(inputImage, grayImage, outputYCbCr,
                            outputDilated, true, 128, 2);
    } else if (mode == "opencl4x64") {
        processWithOpenCL(inputImage, grayImage, outputYCbCr,
                            outputDilated,true, 64, 4);
    } else if (mode == "opencl8x32") {
        processWithOpenCL(inputImage, grayImage, outputYCbCr,
                            outputDilated, true, 32, 8);
    } else if (mode == "opencl256") {
        processWithOpenCL(inputImage, grayImage, outputYCbCr,
                          outputDilated, false, -1, -1, 256);
    } else if (mode == "opencl128") {
        processWithOpenCL(inputImage,grayImage, outputYCbCr,
                          outputDilated, false, -1, -1, 128);
    } else if (mode == "opencl64") {
        processWithOpenCL(inputImage, grayImage, outputYCbCr,
                          outputDilated, false, -1, -1, 64);
    } else if (mode == "opencv") {
        rgbToYCbCrOpenCV(inputImage, outputYCbCr);
        dilateOpenCV(grayImage, outputDilated);
    }
    else {
        std::cerr << "Invalid mode: " << mode << std::endl;
        return;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    elapsedTime = elapsed.count();
}

/**
 * @brief Processes an image in all modes and saves results to a CSV file.
 * @param inputImage The input image.
 * @param inputPath The path to the input image.
 * @param outputBasePath The base path for saving output images.
 * @param csvFile The CSV file for saving results.
 */
void processImageInAllModes(const cv::Mat& inputImage, const std::string& inputPath,
                            const std::string& outputBasePath, std::ofstream& csvFile) {
    std::vector<std::string> modes = {"opencl16x16", "opencl8x8", "opencv", "non-parallel", "opencl256", "opencl128", "opencl64", "opencl4x64", "opencl8x32", "opencl2x128"};
    for (const auto& mode : modes) {
        cv::Mat outputYCbCr, outputDilated;
        double elapsedTime;

        processImageInMode(mode, inputImage, outputYCbCr, outputDilated, elapsedTime);

        // Save outputs
        std::string ycbcrPath = outputBasePath + "_" + mode + "_ycbcr.png";
        std::string dilatedPath = outputBasePath + "_" + mode + "_dilated.png";

        cv::imwrite(ycbcrPath, outputYCbCr);
        cv::imwrite(dilatedPath, outputDilated);

        // Write results to CSV
        csvFile << inputPath << "," << mode << "," << ycbcrPath << "," << dilatedPath << "," << elapsedTime << "\n";

        std::cout << "Mode: " << mode << " processed for " << inputPath << " in " << elapsedTime << " seconds." << std::endl;
    }
}

/**
 * @brief Processes all images in a directory.
 * @param basePath The base directory containing input images.
 * @param outputPath The directory to save output images and results.
 */
void processImagesInDirectory(const std::string& basePath, const std::string& outputPath) {
    namespace fs = std::filesystem;

    // Ensure the output directory exists
    if (!fs::exists(outputPath)) {
        fs::create_directories(outputPath);
    }

    // Open CSV file for writing results in the output directory
    std::string csvPath = outputPath + "/results.csv";
    std::ofstream csvFile(csvPath);
    if (!csvFile.is_open()) {
        std::cerr << "Error opening CSV file for writing: " << csvPath << std::endl;
        return;
    }

    // Write CSV header
    csvFile << "InputFile,Mode,YCbCrOutputFile,DilatedOutputFile,ExecutionTimeSeconds\n";

    // Recursive traversal of all directories and files in the base path
    for (const auto& entry : fs::recursive_directory_iterator(basePath)) {
        // Process only PNG files
        if (entry.is_regular_file() && entry.path().extension() == ".png" || entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg") {
            std::string inputPath = entry.path().string();

            // Calculate relative path from basePath
            std::string relativePath = fs::relative(entry.path(), basePath).string();

            // Construct the output base path
            std::string outputBasePath = outputPath + "/" + relativePath.substr(0, relativePath.find_last_of('.'));

            // Ensure the output subdirectory exists
            fs::create_directories(fs::path(outputBasePath).parent_path());

            std::cout << "Processing file: " << inputPath << std::endl;

            // Load and process the image
            cv::Mat inputImage = cv::imread(inputPath, cv::IMREAD_COLOR);
            if (inputImage.empty()) {
                std::cerr << "Error loading image: " << inputPath << std::endl;
                continue;
            }

            processImageInAllModes(inputImage, inputPath, outputBasePath, csvFile);
        }
    }

    csvFile.close();
    std::cout << "Results saved to: " << csvPath << std::endl;
}

/**
 * @brief The main function to initialize OpenCL, process images, and release resources.
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments.
 * @return Exit code.
 */
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <basepath> <outputpath>" << std::endl;
        return 1;
    }

    std::string basePath = argv[1];
    std::string outputPath = argv[2];

    std::cout << "Initializing OpenCL..." << std::endl;
    initializeOpenCL();

    std::cout << "Starting image processing in base path: " << basePath << std::endl;
    processImagesInDirectory(basePath, outputPath);

    std::cout << "Releasing OpenCL resources..." << std::endl;
    releaseOpenCL();

    std::cout << "Image processing completed." << std::endl;
    return 0;
}


