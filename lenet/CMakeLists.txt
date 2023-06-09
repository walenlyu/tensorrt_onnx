cmake_minimum_required(VERSION 2.6)

project(lenet5)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/local/tensorrt/include/)
link_directories(/usr/local/tensorrt/lib/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lenet5  ${PROJECT_SOURCE_DIR}/lenet5.cpp )
target_link_libraries(lenet5 nvinfer)
target_link_libraries(lenet5 nvonnxparser) 
target_link_libraries(lenet5 cudart)
target_link_libraries(lenet5 nvinfer_plugin)
target_link_libraries(lenet5 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

