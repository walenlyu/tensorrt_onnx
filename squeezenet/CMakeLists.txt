cmake_minimum_required(VERSION 2.6)

project(squeezenet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/local/tensorrt/include)
link_directories(/usr/local/tensorrt/lib)

add_executable(squeezenet ${PROJECT_SOURCE_DIR}/squeezenet.cpp)
target_link_libraries(squeezenet nvinfer)
target_link_libraries(squeezenet cudart)
target_link_libraries(squeezenet nvonnxparser) 

add_definitions(-O2 -pthread)

