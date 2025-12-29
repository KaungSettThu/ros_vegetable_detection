# Dockerfile for ROS 2 + Gazebo + YOLO/Open3D/RealSense

# Base image with ROS 2 Humble and desktop tools
FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-rviz2 \
    ros-humble-image-transport \
    ros-humble-cv-bridge \
    ros-humble-robot-state-publisher \
    ros-humble-point-cloud-transport \
    ros-humble-xacro \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    ultralytics \
    numpy \
    open3d \
    pyrealsense2 \
    opencv-python

# Set up ROS 2 workspace
WORKDIR /ros_ws
RUN mkdir -p src

# Build ROS workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Automatically source ROS workspace on container start
RUN echo "source /ros_ws/install/setup.bash" >> ~/.bashrc

# Set default command
CMD ["bash"]

