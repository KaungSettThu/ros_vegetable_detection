import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3Stamped

from cv_bridge import CvBridge
import numpy as np
import cv2

from yolo_realsense.detector import Detector


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # YOLO + RealSense detector
        self.detector = Detector('/ros_ws/src/yolo_realsense/weights/best.pt')

        # ROS publishers
        self.class_pub = self.create_publisher(String, 'detected_object/class', 10)
        self.pos_pub = self.create_publisher(PointStamped, 'detected_object/position', 10)
        self.image_pub = self.create_publisher(Image, 'yolo_image', 10)
        self.dim_pub = self.create_publisher(Vector3Stamped, 'detected_object/dimensions', 10)


        self.bridge = CvBridge()

        # 30 Hz
        self.timer = self.create_timer(0.033, self.timer_callback)

        self.get_logger().info("YOLO RealSense node started")

    def timer_callback(self):
        color, depth = self.detector.get_frames()
        if color is None or depth is None:
            return

        detections = self.detector.detect_objects(color, depth)

        for det in detections:
            pts = det['points_3d']
            if pts.shape[0] == 0:
                continue

            # --- 3D centroid ---
            centroid = np.mean(pts, axis=0)

            # Publish class
            class_msg = String()
            class_msg.data = det['class']
            self.class_pub.publish(class_msg)

            # Publish position
            pos_msg = PointStamped()
            pos_msg.header.stamp = self.get_clock().now().to_msg()
            pos_msg.header.frame_id = "camera_link"
            pos_msg.point.x = float(centroid[0])
            pos_msg.point.y = float(centroid[1])
            pos_msg.point.z = float(centroid[2])
            self.pos_pub.publish(pos_msg)

            # Draw bbox for visualization
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                color,
                f"{det['class']} {centroid[2]:.2f}m",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Publish dimensions
            dims = det['dimensions']
            dim_msg = Vector3Stamped()
            dim_msg.header.stamp = self.get_clock().now().to_msg()
            dim_msg.header.frame_id = "camera_link"
            dim_msg.vector.x = float(dims[0])
            dim_msg.vector.y = float(dims[1])
            dim_msg.vector.z = float(dims[2])
            self.dim_pub.publish(dim_msg)

        # Publish annotated image
        img_msg = self.bridge.cv2_to_imgmsg(color, encoding='bgr8')
        self.image_pub.publish(img_msg)

    def destroy_node(self):
        self.detector.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()