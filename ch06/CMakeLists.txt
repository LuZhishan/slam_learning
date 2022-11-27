cmake_minimum_required(VERSION 3.5)
project(use_opencv)

set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV 4 REQUIRED)
include_directories(
    ${OPENCV_INCLUDE_DIR}
)

add_executable(undistort src/undistort.cpp)
target_link_libraries(undistort ${OpenCV_LIBRARIES})