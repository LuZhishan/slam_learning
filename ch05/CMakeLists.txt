cmake_minimum_required(VERSION 3.5)
project(use_opencv)

set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV 4 REQUIRED)
find_package(PCL 1.12 REQUIRED)
include_directories(
    ${OPENCV_INCLUDE_DIR}
    ${PCL_INCLUDE_DIR}
)

add_executable(undistort src/undistort.cpp)
target_link_libraries(undistort ${OpenCV_LIBRARIES})
add_executable(stereo src/stereo.cpp)
target_link_libraries(stereo ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})