cmake_minimum_required(VERSION 3.5)
project(use_g2o)

set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR}
)

add_executable(orb src/orb.cpp)
target_link_libraries(orb ${OpenCV_LIBRARIES})
# add_executable(g2o src/g2o.cpp)
# target_link_libraries(g2o ${OpenCV_LIBRARIES})