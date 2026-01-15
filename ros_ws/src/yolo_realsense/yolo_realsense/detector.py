import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import os

class Detector:
    def __init__(self, model_path):
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

        self.model = YOLO(model_path)
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)

        # Depth Scales and Camera intrinsics
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        intr = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy = intr.fx, intr.fy
        self.px, self.py = intr.ppx, intr.ppy

    """
    Get the color and the depth frames form the camera
    """
    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image
    

    """
    Detect the objects according to the classes
    and calculate the depth of the objects
    """
    def detect_objects(self, color_image, depth_image):
        results = self.model.predict(color_image, verbose=False)
        detections = []

        for box in results[0].boxes or []:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            z = depth_image[cy, cx] * self.depth_scale

            # ROI point cloud
            depth_roi = depth_image[y1:y2, x1:x2]

            mask = depth_roi > 0
            z_vals = depth_roi[mask] * self.depth_scale

            if z_vals.size == 0:
                continue

            u_coords, v_coords = np.where(mask)
            X = (u_coords + x1 - self.px) * z_vals / self.fx
            Y = (v_coords + y1 - self.py) * z_vals / self.fy
            points_3d = np.stack((X, Y, z_vals), axis=-1)

            if points_3d.shape[0] > 0:
                min_b = points_3d.min(axis=0)
                max_b = points_3d.max(axis=0)
                dims = max_b - min_b
                detections.append({
                    "class": cls_name,
                    "bbox": (x1, y1, x2, y2),
                    "center_depth": z,
                    "dimensions": dims,
                    "points_3d": points_3d
                })
        return detections

    def stop(self):
        self.pipeline.stop()