FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------
# Base system + ROS packages
# ----------------------------
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-colcon-common-extensions \
    python3-rosdep \
    build-essential \
    curl \
    gnupg2 \
    lsb-release \
    udev \
    # ROS core
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-joint-state-broadcaster \
    ros-humble-joint-trajectory-controller \
    ros-humble-rviz2 \
    ros-humble-image-transport \
    ros-humble-cv-bridge \
    ros-humble-robot-state-publisher \
    ros-humble-point-cloud-transport \
    ros-humble-diagnostic-updater \
    ros-humble-xacro \
    ros-humble-ros-gz-bridge \
    ros-humble-ros-gz-sim \
    # Serial support
    ros-humble-serial-driver \
    ros-humble-io-context \
    ros-humble-asio-cmake-module \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Python packages (YOLO / Open3D)
# ----------------------------
RUN pip3 install --no-cache-dir \
    ultralytics \
    numpy \
    open3d \
    opencv-python

# ----------------------------
# rosdep init
# ----------------------------
RUN rosdep init || true && rosdep update

# ----------------------------
# Workspace
# ----------------------------
WORKDIR /ros_ws
RUN mkdir -p src

# Auto-source ROS
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

CMD ["bash"]

