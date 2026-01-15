from detector import Detector
import time

det = Detector('/ros_ws/src/yolo_realsense/weights/best.pt')

try:
    while True:
        color, depth = det.get_frames()
        if color is None or depth is None:
            continue
        
        detections = det.detect_objects(color, depth)
        if detections:
            for obj in detections:
                print(f"Detected {obj['class']} at center depth {obj['center_depth']:.3f}m")
                print(f"Object dimensions: {obj['dimensions']}")
                print(f"Sample 3D points:\n{obj['points_3d'][:5]}")
        else:
            print("No objects detected in this frame.")

        time.sleep(0.1)
except KeyboardInterrupt:
    det.stop()
    print("Detection stopped.")
