# Vegetable Detection and Depth Perception

This project demonstrates vegetable detection using YOLOv8 and depth perception with Intel RealSense camera.

## Testing of YOLO model (WITHOUT ROS)

### 1. Create a conda environment
```bash
conda create -n yolov8 python=3.10 -y
```

### 2. Activate the conda environment
```bash
conda activate yolov8
```

### 3. Install required packages
```bash
pip install ultralytics pyrealsense2 opencv-python open3d numpy
```

## Checking Available Cameras

List all video devices:
```bash
ls -l /dev/video*
```

Test a specific camera using ffplay (replace `#` with camera number):
```bash
ffplay /dev/video#
```

## Testing Detection Model

To test the YOLOv8 model with your computer camera, run:
```bash
yolo predict model=./runs/detect/vegetable_train/exp_final/weights/best.pt source=# show=True
```
Replace `#` after `source` with your camera number.

## Running Depth Perception

To run depth perception using the RealSense camera, execute:
```bash
python ./scripts/depth_display.py
```

This script will detect vegetables, compute their 3D coordinates, and optionally display the point cloud.

## Notes
- Ensure your RealSense camera is connected and drivers are installed.
- Use `.gitignore` to ignore sensitive files like `.ssh` and prediction folders in `runs/predict*`.
- The 3D coordinates from the point cloud can be used for robot picking.
- Width and height of detected objects may be used for planning robot grip.

## Testing of Camera ROS Node

### 1. Install a docker container

Go to the docker folder. In the terminal, run the command
```bash
sudo ./build_docker.sh
```

### 2. Run / Enter the docker container
If the docker container has not been run before, run the docker by the use of the command
```bash
sudo ./run_docker.sh
```

If the docker container has been run, enter the docker by the use of the command
```bash
sudo ./into_docker.sh
```

### 3. Build the project

Go to the ros_ws
```bash
cd ros_ws
```

Build the ros workspace
```bash
colcon build
```

### 4. Run the ros node
```bash
ros2 run yolo_realsense yolo_node
```

### 5. Debugging
The topics that are published can be checked by running the command
```bash
ros_2 topic list
```

The published information by the topic can be checked by
```bash
ros_2 topic echo <<insert topic name here>>
```


