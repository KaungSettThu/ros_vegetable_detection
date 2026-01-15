#!/bin/bash

CONTAINER_NAME=ros2_gazebo_container
IMAGE_NAME=ros2_gazebo_humble

# Project root (one level above docker/)
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Allow Docker to access X server
xhost +local:docker

# Remove old container if it exists
if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
    docker rm -f ${CONTAINER_NAME}
fi

# Detect RealSense devices automatically
DEVICES=""
for d in /dev/video*; do
    DEVICES+=" --device=$d:$d"
done

# Run container
docker run -it \
    --name ${CONTAINER_NAME} \
    --net=host \
    --ipc=host \
    --privileged \
    $DEVICES \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${PROJECT_ROOT}/ros_ws:/ros_ws \
    ${IMAGE_NAME}

