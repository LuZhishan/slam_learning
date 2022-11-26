cmake_minimum_required(VERSION 3.5)
project(use_eigen)

set(CMAKE_BUILD_TYPE "Release")

find_package(Eigen3 REQUIRED)
include_directories(
    ${EIGEN3_INCLUDE_DIR}
)
add_executable(eigen_base src/eigen_base.cpp)
add_executable(eigen_geometry src/eigen_geometry.cpp)