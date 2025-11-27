# Vegetable Detection and Depth Perception

This project demonstrates vegetable detection using YOLOv8 and depth perception with Intel RealSense camera.

## Setup

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
