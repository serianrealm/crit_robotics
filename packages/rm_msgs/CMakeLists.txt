cmake_minimum_required(VERSION 3.28)
project(rm_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
<<<<<<< HEAD
find_package(std_msgs REQUIRED)
=======
find_package(sensor_msgs REQUIRED)
>>>>>>> main
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Control.msg"
  "msg/Imu.msg"
  "msg/ImuStamped.msg"
<<<<<<< HEAD
  "msg/RmRobot.msg"
  "msg/State.msg"
  "msg/PcCommon.msg"
  "msg/TestRecv.msg"
  "srv/ToggleVisionFollowId.srv"
  "srv/SetVisionMode.srv"
  DEPENDENCIES
  std_msgs
=======
  "msg/State.msg"
  "srv/ToggleVisionFollowId.srv"
  "srv/SetVisionMode.srv"
  DEPENDENCIES
>>>>>>> main
  geometry_msgs
)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

<<<<<<< HEAD
ament_export_dependencies(rosidl_default_runtime)

=======
>>>>>>> main
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
