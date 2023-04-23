cmake_minimum_required(VERSION 2.6)

project(yolov3)

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
include_directories(~/Softwares/tensorrt/include)
link_directories(~/Softwares/tensorrt/lib)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov3 ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/yolov3.cpp)
target_link_libraries(yolov3 nvinfer)
target_link_libraries(yolov3 cudart)
target_link_libraries(yolov3 nvonnxparser) 
target_link_libraries(yolov3 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

