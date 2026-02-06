# cmake_minimum_required(VERSION 3.8)
# project(basic_predictor)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# add_compile_options(-O3)
# set(CMAKE_CXX_STANDARD 20)

# SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# find_package(OpenCV 4 REQUIRED)
# find_package(Eigen3 3.3 REQUIRED)
# find_package(Ceres REQUIRED)
# find_package(tf2_eigen REQUIRED)
# find_package(vision_msgs REQUIRED)
# find_package(visualization_msgs REQUIRED)
# find_package(rm_msgs REQUIRED)
# find_package(rclcpp REQUIRED)

# add_executable(${PROJECT_NAME}
#   src/node_interface.cpp
#   src/basic_predictor.cpp
#   src/ballistic_utils.cpp
# )

# include_directories(
#   ${CERES_INCLUDE_DIRS}
# )

# target_include_directories(${PROJECT_NAME} PUBLIC include)
# target_include_directories(${PROJECT_NAME} PUBLIC
#   ${CERES_INCLUDE_DIRS}
#   ${OpenCV_INCLUDE_DIRS}
# )
# target_link_libraries(${PROJECT_NAME} PUBLIC
#   ${OpenCV_LIBS}
#   ${CERES_LIBRARIES}
#   Eigen3::Eigen
#   rclcpp::rclcpp
#   tf2_eigen::tf2_eigen
#   tf2_geometry_msgs::tf2_geometry_msgs
# )

# install(
#   DIRECTORY config launch
#   DESTINATION ${PROJECT_NAME}/share
# )