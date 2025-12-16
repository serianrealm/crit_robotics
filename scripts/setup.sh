#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$SCRIPT_DIR/env"

print_step() {
  echo -e "\n[setup] $1" >&2
}

if [ -z "${ROS_DISTRO}" ]; then
  print_step "[setup] ROS2 not found, installing ROS2 kilted"
  "$ENV_DIR/install_ros.sh"

  if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
  fi
else
  print_step "[setup] ROS2 detected: ${ROS_DISTRO}"
fi

print_step "ROS environment"
"$ENV_DIR/rosdep.sh"

print_step "System dependencies"
"$ENV_DIR/apt.sh"

print_step "Shell configuration"
"$ENV_DIR/bashrc.sh"

print_step "Python dependencies"
"$ENV_DIR/pyenv.sh"

print_step "Done."
