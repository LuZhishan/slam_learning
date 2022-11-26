cmake_minimum_required(VERSION 3.5)
project(hello)

set(CMAKE_BUILD_TYPE "Debug")

# 添加依赖的头文件
include_directories(
    include
)

# 添加自定义的库文件
add_library(libhello include/libhello.cpp)

# 添加可执行文件
add_executable(hello src/hello.cpp)
target_link_libraries(hello libhello)