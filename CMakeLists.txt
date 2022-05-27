cmake_minimum_required(VERSION 3.10)

project(mattoarray LANGUAGES CXX CUDA)


add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)



set(CUDA_DIR /usr/local/cuda-10.2/)
find_package(CUDA REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBS}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(/usr/local/cuda/lib64)

# set(OpenCV_DIR  /opt/opencv-4.5.5/lib/cmake/opencv4)
find_package(OpenCV)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# target_include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${OpenCV_INCLUDE_DIRS})
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


add_executable(mattoarray mattoarray.cu)
target_link_libraries(mattoarray ${OpenCV_LIBS} )
target_link_libraries(mattoarray ${CUDA_LIBS} )
target_link_libraries(mattoarray cudart )
# add_definitions(-O2 -pthread)







