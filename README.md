# SOR-Session-4-code-Aditya-Gokhale
This is the source code for my assignment. Following are the changes I made to yolo_detection_node.py to obtain the desired results
```
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import numpy as np
import threading
import time

from ultralytics import YOLO


STOP_DISTANCE = 0.6  # metres


class YoloDetectorNode(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        self.model = YOLO("yolov8s.pt")
        self.get_logger().info("YOLO model loaded")

        self.bridge = CvBridge()

        # Camera RGB
        self.subscription = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            1
        )

        # Depth camera
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth_image',
            self.depth_callback,
            1
        )

        # cmd_vel publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.depth_image = None
        self.depth_lock = threading.Lock()

        self.mission_complete = False
        self.running = True
        self.prev_time = time.time()
        self.last_search_print = 0.0
        self.last_found_print = 0.0

        self.target_object = None
        self.target_lock = threading.Lock()

        # Start ROS spin thread
        self.spin_thread = threading.Thread(
            target=self.spin_thread_func,
            daemon=True
        )
        self.spin_thread.start()

        # Start interactive input thread
        self.input_thread = threading.Thread(
            target=self.input_thread_func,
            daemon=True
        )
        self.input_thread.start()

    def input_thread_func(self):
        print("\n=== YOLO Object Hunt ===")
        print("(type 'stop' to cancel current mission)\n")
        while self.running:
            try:
                target = input("Enter target object (e.g. bottle, chair, person): ").strip().lower()
                if not target or target == 'stop':
                    with self.target_lock:
                        self.target_object = None
                        self.mission_complete = False
                    # publish zero velocity to stop the robot
                    twist = Twist()
                    self.cmd_vel_pub.publish(twist)
                    print("Mission cancelled. Robot stopped.")
                    continue
                with self.target_lock:
                    self.target_object = target
                    self.mission_complete = False
                print(f"Searching for: {self.target_object}")
            except EOFError:
                break
            except Exception:
                break

    def spin_thread_func(self):
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        with self.frame_lock:
            self.latest_frame = frame

    def depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        with self.depth_lock:
            self.depth_image = depth.copy()

    def stop(self):
        self.running = False
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1)

    def display_image(self):
        cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("YOLO Detection", 1600, 900)

        while rclpy.ok() and self.running:
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is not None:
                result = self.run_yolo(frame)
                cv2.imshow("YOLO Detection", result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.running = False
                break

        cv2.destroyAllWindows()

    def get_depth_at(self, cx, cy):
        with self.depth_lock:
            if self.depth_image is None:
                return None
            h, w = self.depth_image.shape[:2]
            r = 5
            x1 = max(cx - r, 0)
            x2 = min(cx + r, w - 1)
            y1 = max(cy - r, 0)
            y2 = min(cy + r, h - 1)
            region = self.depth_image[y1:y2, x1:x2]
            valid = region[np.isfinite(region) & (region > 0)]
            if len(valid) == 0:
                return None
            return float(np.median(valid))

    def run_yolo(self, frame):
        CONF_THRESHOLD = 0.35
        results = self.model(frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)

        detections = []
        target_found = False
        target_cx = None
        target_cy = None
        target_distance = None

        frame_center_x = frame.shape[1] // 2

        with self.target_lock:
            current_target = self.target_object
            current_mission_complete = self.mission_complete

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                color = self.class_color(class_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {confidence:.2f}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_y = max(y1 - 10, th + 10)
                cv2.rectangle(frame, (x1, text_y - th - baseline), (x1 + tw + 10, text_y + baseline), color, -1)
                cv2.putText(frame, label, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 5, color, -1)

                detections.append(f"{class_name} ({confidence:.2f})")

                if (current_target is not None
                        and class_name.lower() == current_target
                        and not current_mission_complete):
                    target_found = True
                    target_cx = cx
                    target_cy = cy
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

        # --- Motion logic ---
        if current_target is not None and not current_mission_complete:
            if target_found:
                target_distance = self.get_depth_at(target_cx, target_cy)

                if target_distance is not None:
                    now = time.time()
                    if now - self.last_found_print >= 1.0:
                        print(f"Target Found | Distance to {current_target}: {target_distance:.2f} m")
                        self.last_found_print = now

                    if target_distance < STOP_DISTANCE:
                        # Stage 6 — stop
                        twist = Twist()
                        self.cmd_vel_pub.publish(twist)
                        with self.target_lock:
                            self.mission_complete = True
                        print("Mission Completed\nTarget Reached Successfully")
                    else:
                        # Stage 5 — approach with angular correction
                        error = target_cx - frame_center_x
                        twist = Twist()
                        twist.linear.x = 0.2
                        twist.angular.z = -error / 500.0
                        self.cmd_vel_pub.publish(twist)
                else:
                    # Target visible but no valid depth
                    twist = Twist()
                    twist.linear.x = 0.1
                    self.cmd_vel_pub.publish(twist)
            else:
                # Stage 3 — search by rotating
                twist = Twist()
                twist.angular.z = 0.3
                self.cmd_vel_pub.publish(twist)
                now = time.time()
                if now - self.last_search_print >= 1.0:
                    print("Searching...")
                    self.last_search_print = now

        # --- Dashboard ---
        current_time = time.time()
        fps = 1.0 / max(current_time - self.prev_time, 1e-6)
        self.prev_time = current_time

        dashboard_width = 350
        dashboard = np.zeros((frame.shape[0], dashboard_width, 3), dtype=np.uint8)

        target_str = current_target if current_target else "None"
        if current_mission_complete:
            status_str = "COMPLETE"
            status_color = (0, 255, 0)
        elif current_target is None:
            status_str = "WAITING"
            status_color = (128, 128, 128)
        elif target_found:
            status_str = "TRACKING"
            status_color = (0, 255, 255)
        else:
            status_str = "SEARCHING"
            status_color = (0, 165, 255)

        cv2.putText(dashboard, "Mission Control", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(dashboard, f"FPS     : {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(dashboard, f"TARGET  : {target_str}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(dashboard, f"STATUS  : {status_str}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        dist_str = f"{target_distance:.2f} m" if target_distance else "N/A"
        cv2.putText(dashboard, f"DISTANCE: {dist_str}", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(dashboard, "Detections", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(dashboard, f"Objects : {len(detections)}", (20, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y = 300
        for det in detections[:20]:
            cv2.putText(dashboard, det, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 28

        combined = np.hstack((frame, dashboard))
        return combined

    def class_color(self, class_id):
        np.random.seed(class_id)
        return tuple(int(c) for c in np.random.randint(100, 255, 3))


def main(args=None):
    print("OpenCV Version:", cv2.__version__)
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        node.display_image()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
