cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(g2o REQUIRED)

add_executable(main main.cpp)
add_executable(g2o g2o.cpp)
add_executable(ceres_o ceres.cpp)
add_executable(orb_cv orb_cv.cpp)

target_link_libraries(orb_cv
        ${OpenCV_LIBS})

target_link_libraries(main
        Sophus::Sophus
        Eigen3::Eigen
        ${OpenCV_LIBS}
        Ceres::ceres)

target_link_libraries(ceres_o
        Sophus::Sophus
        Eigen3::Eigen
        ${OpenCV_LIBS}
        Ceres::ceres)

target_link_libraries(g2o
        Eigen3::Eigen
        g2o::core
        g2o::solver_dense
        ${OpenCV_LIBS})
