cmake_minimum_required(VERSION 3.5)
project(use_eigen)

set(CMAKE_BUILD_TYPE "Release")

find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(sophus src/sophus.cpp)
target_link_libraries(sophus Sophus::Sophus)