cmake_minimum_required(VERSION 3.5)
project(use_g2o)

set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
LIST(APPEND CMAKE_MODULE_PATH ~/App/g2o-20201223_git/cmake_modules)
find_package(G2O REQUIRED)
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
)

add_executable(orb src/orb.cpp)
target_link_libraries(orb ${OpenCV_LIBRARIES})
add_executable(pose_estimate_2d2d src/pose_estimate_2d2d.cpp)
target_link_libraries(pose_estimate_2d2d ${OpenCV_LIBRARIES})
add_executable(pose_estimate_3d2d src/pose_estimate_3d2d.cpp)
target_link_libraries(pose_estimate_3d2d ${OpenCV_LIBRARIES} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} Sophus::Sophus)