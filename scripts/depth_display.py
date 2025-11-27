from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import open3d as o3d  # for  point cloud visualization

# Path to folder where THIS script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# load YOLO model
model_path = os.path.join(SCRIPT_DIR, "../runs/detect/vegetable_train/exp2/weights/best.pt")
model = YOLO(model_path)

# Configure RealSense
"""
Creates a pipeline object.
A pipeline is a high-level RealSense API that lets you:
- configure the camera streams,
- start/stop streaming,
- retrieve frames (color, depth, infrared),
- sync and manage them internally.

Creates a configuration object that describes which camera streams you want.
config.enable_stream(stream_type, width, height, format, framerate)

stream_type
- rs.stream.color for color camera
- rs.stream.depth for depth sensor
- rs.stream.infrared for infrared camera

format
- rs.format.bgr8 for rgb images
- rs.format.z16 for 16 bit formats
- rs.format.y8 for infrared
"""
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Depth scale
depth_sensor = profile.get_device().first_depth_sensor()    # recieve the first depth sensor
depth_scale = depth_sensor.get_depth_scale()
print("Depth scale:", depth_scale, "meters")

# Camera intrinsics
depth_intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx, fy = depth_intr.fx, depth_intr.fy
px, py = depth_intr.ppx, depth_intr.ppy

# Colors for each class
class_colors = {
    'potato': (0, 255, 255),   # yellow
    'tomato': (0, 0, 255),   # red
    'carrot': (255, 128, 0)  # orange
}

# --- Open3D Visualizer setup ---
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Point Cloud", width=640, height=480)
pcd_combined = o3d.geometry.PointCloud()
vis.add_geometry(pcd_combined)

# Detection start
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # YOLO detection
        results = model.predict(color_image, verbose=False)
        points_list = []
        colors_list = []

        for box in results[0].boxes:
            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            # Color for this class
            color = class_colors.get(cls_name, (255, 255, 0))  # default cyan

            # Depth at center of bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            depth = depth_image[cy, cx] * depth_scale  # meters

            # Draw box and text
            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(color_image, f"{cls_name} {depth:.3f}m", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Convert bounding box region of interest to point cloud
            h, w = depth_image.shape
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            # print(f"Clipped bbox: {x1}, {y1}, {x2}, {y2}")

            depth_roi = depth_image[y1:y2, x1:x2]
            color_roi = color_image[y1:y2, x1:x2]
            
            valid_points = np.count_nonzero(depth_roi)

            # print("Depth ROI shape:", depth_roi.shape)
            print(f"Valid depth points in ROI: {valid_points}")


            """
            To create a point cloud for a detected object, the 3D images can be computed by
                - X = (u - cx) * depth * depth_scale / fx
                - X = (v - cy) * depth * depth_scale / fy
                - Z = depth * depth_scale
            """

            u, v = np.meshgrid(np.arange(depth_roi.shape[1]), np.arange(depth_roi.shape[0]))
            z = depth_roi * depth_scale
            valid = z > 0
            u = u[valid]
            v = v[valid]
            z = z[valid]
            x = (u + x1 - px) * z / fx
            y = (v + y1 - py) * z / fy
            # points: (N,3)
            points = np.stack((x, y, z), axis=-1)

            # Correctly extract colors for valid points
            colors = color_roi[valid]  # shape (N,3)
            colors = colors[:, ::-1] / 255.0  # BGR -> RGB

            if points.shape[0] > 0:
                points_list.append(points)
                colors_list.append(colors)

        # Merge all points for visualization
        if points_list:
            all_points = np.vstack(points_list)
            all_colors = np.vstack(colors_list)
            pcd_combined.points = o3d.utility.Vector3dVector(all_points)
            pcd_combined.colors = o3d.utility.Vector3dVector(all_colors)

            # Update Open3D visualizer non-blocking
            vis.update_geometry(pcd_combined)
            vis.poll_events()
            vis.update_renderer()

        cv2.waitKey(1)


        # Display
        cv2.imshow("Image detection", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
