cmake_minimum_required(VERSION 3.10)
project(open_cl_test)

# Specify OpenCV directory
set(OpenCV_DIR "/Users/aleksey/build_opencv")

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Set linker flags for OpenMP and OpenCL framework
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework OpenCL")

# Add the opencl-clhpp-headers path manually
set(OPENCL_CLHPP_INCLUDE_DIR "/usr/local/opt/opencl-clhpp-headers/include")

# Add executable
add_executable(open_cl_test main.cpp)

# Link libraries including OpenCV, MPI, OpenMP, and OpenCL framework
target_link_libraries(open_cl_test PUBLIC ${OpenCV_LIBS} "-framework OpenCL")


# Include directories for OpenCV, MPI, OpenMP, OpenCL, and clhpp headers
target_include_directories(open_cl_test PUBLIC ${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS} ${OPENCL_CLHPP_INCLUDE_DIR} .)
