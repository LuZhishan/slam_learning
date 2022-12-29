cmake_minimum_required(VERSION 3.5)
project(pose_estimation)

set(CMAKE_BUILD_TYPE "Release")
# 如果g2o文件夹不在当前路径下，就得加下面一句话，添加cmake_modulesd路径
LIST(APPEND CMAKE_MODULE_PATH ~/App/g2o-20201223_git/cmake_modules)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(G2O REQUIRED)
find_package(Ceres REQUIRED)
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR}
    ${G2O_INCLUDE_DIR}
    ${CERES_INCLUDE_DIR}
)

# add_executable(gaussNewton src/gaussNewton.cpp)
# target_link_libraries(gaussNewton ${OpenCV_LIBRARIES})
# add_executable(use_g2o src/use_g2o.cpp)
# target_link_libraries(use_g2o ${OpenCV_LIBRARIES} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})
# add_executable(use_ceres src/use_ceres.cpp)
# target_link_libraries(use_ceres ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})
# add_executable(use_ceres2 src/use_ceres2.cpp)
# target_link_libraries(use_ceres2 ${CERES_LIBRARIES})
add_executable(use_ceres3 src/use_ceres3.cpp)
target_link_libraries(use_ceres3 ${CERES_LIBRARIES})