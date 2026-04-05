#!/usr/bin/env python3
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage


class HSVTunerNode(Node):
    def __init__(self):
        super().__init__('hsv_tuner_node')

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('color_topic', '/zedm/zed_node/left/image_rect_color/compressed')
        self.declare_parameter('window_name_original', 'HSV Tuner - Original')
        self.declare_parameter('window_name_mask', 'HSV Tuner - Mask')
        self.declare_parameter('print_hsv_every_n_frames', 30)

        self.use_compressed = bool(self.get_parameter('use_compressed').value)
        self.color_topic = self.get_parameter('color_topic').value
        self.window_name_original = self.get_parameter('window_name_original').value
        self.window_name_mask = self.get_parameter('window_name_mask').value
        self.print_hsv_every_n_frames = int(self.get_parameter('print_hsv_every_n_frames').value)

        self.bridge = CvBridge()
        self.frame_count = 0
        self.latest_bgr = None
        self.latest_hsv = None
        self.last_click = None  # (x, y)

        # Initial HSV values
        self.h_min = 70
        self.h_max = 128
        self.s_min = 60
        self.s_max = 223
        self.v_min = 38
        self.v_max = 255


        # Morphology
        self.open_kernel = 1
        self.close_kernel = 1

        # --------------------------------------------------
        # OpenCV windows / trackbars
        # --------------------------------------------------
        cv2.namedWindow(self.window_name_original, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.window_name_mask, cv2.WINDOW_NORMAL)

        cv2.resizeWindow(self.window_name_original, 960, 540)
        cv2.resizeWindow(self.window_name_mask, 960, 540)

        cv2.setMouseCallback(self.window_name_original, self.on_mouse)

        cv2.createTrackbar('H_MIN', self.window_name_mask, self.h_min, 179, self.on_trackbar)
        cv2.createTrackbar('H_MAX', self.window_name_mask, self.h_max, 179, self.on_trackbar)
        cv2.createTrackbar('S_MIN', self.window_name_mask, self.s_min, 255, self.on_trackbar)
        cv2.createTrackbar('S_MAX', self.window_name_mask, self.s_max, 255, self.on_trackbar)
        cv2.createTrackbar('V_MIN', self.window_name_mask, self.v_min, 255, self.on_trackbar)
        cv2.createTrackbar('V_MAX', self.window_name_mask, self.v_max, 255, self.on_trackbar)

        cv2.createTrackbar('OPEN_K', self.window_name_mask, self.open_kernel, 20, self.on_trackbar)
        cv2.createTrackbar('CLOSE_K', self.window_name_mask, self.close_kernel, 20, self.on_trackbar)

        # --------------------------------------------------
        # Subscribers
        # --------------------------------------------------
        if self.use_compressed:
            self.sub = self.create_subscription(
                CompressedImage,
                self.color_topic,
                self.color_cb_compressed,
                qos_profile_sensor_data
            )
        else:
            self.sub = self.create_subscription(
                Image,
                self.color_topic,
                self.color_cb_raw,
                qos_profile_sensor_data
            )

        # timer for cv2 refresh
        self.timer = self.create_timer(0.03, self.update_view)  # ~33 Hz

        self.get_logger().info('========================================')
        self.get_logger().info('HSV Tuner Node Started')
        self.get_logger().info(f'use_compressed         : {self.use_compressed}')
        self.get_logger().info(f'color_topic            : {self.color_topic}')
        self.get_logger().info(f'window_name_original   : {self.window_name_original}')
        self.get_logger().info(f'window_name_mask       : {self.window_name_mask}')
        self.get_logger().info('Controls:')
        self.get_logger().info('  - Adjust HSV with trackbars')
        self.get_logger().info('  - Click image to inspect pixel HSV')
        self.get_logger().info('  - Press q in image window to quit')
        self.get_logger().info('========================================')

    # --------------------------------------------------
    # Image callbacks
    # --------------------------------------------------
    def color_cb_raw(self, msg: Image):
        try:
            self.latest_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Raw image conversion failed: {repr(e)}')

    def color_cb_compressed(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.latest_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if self.latest_bgr is None:
                self.get_logger().error('Compressed image decode returned None')
        except Exception as e:
            self.get_logger().error(f'Compressed image decode failed: {repr(e)}')

    # --------------------------------------------------
    # UI callbacks
    # --------------------------------------------------
    def on_trackbar(self, _):
        self.h_min = cv2.getTrackbarPos('H_MIN', self.window_name_mask)
        self.h_max = cv2.getTrackbarPos('H_MAX', self.window_name_mask)
        self.s_min = cv2.getTrackbarPos('S_MIN', self.window_name_mask)
        self.s_max = cv2.getTrackbarPos('S_MAX', self.window_name_mask)
        self.v_min = cv2.getTrackbarPos('V_MIN', self.window_name_mask)
        self.v_max = cv2.getTrackbarPos('V_MAX', self.window_name_mask)

        self.open_kernel = cv2.getTrackbarPos('OPEN_K', self.window_name_mask)
        self.close_kernel = cv2.getTrackbarPos('CLOSE_K', self.window_name_mask)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_click = (x, y)
            if self.latest_bgr is not None and self.latest_hsv is not None:
                h, w = self.latest_bgr.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    bgr = self.latest_bgr[y, x]
                    hsv = self.latest_hsv[y, x]
                    self.get_logger().info(
                        f'Clicked pixel ({x}, {y}) | '
                        f'BGR=({int(bgr[0])}, {int(bgr[1])}, {int(bgr[2])}) '
                        f'HSV=({int(hsv[0])}, {int(hsv[1])}, {int(hsv[2])})'
                    )

    # --------------------------------------------------
    # Main display update
    # --------------------------------------------------
    def update_view(self):
        if self.latest_bgr is None:
            cv2.waitKey(1)
            return

        self.frame_count += 1

        bgr = self.latest_bgr.copy()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        self.latest_hsv = hsv

        lower = np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)
        upper = np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        if self.open_kernel > 1:
            k_open = np.ones((self.open_kernel, self.open_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

        if self.close_kernel > 1:
            k_close = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

        masked_bgr = cv2.bitwise_and(bgr, bgr, mask=mask)

        # Draw clicked pixel
        if self.last_click is not None:
            x, y = self.last_click
            h, w = bgr.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(bgr, (x, y), 5, (0, 0, 255), -1)
                px_hsv = hsv[y, x]
                cv2.putText(
                    bgr,
                    f'HSV=({int(px_hsv[0])},{int(px_hsv[1])},{int(px_hsv[2])})',
                    (x + 10, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        # Overlay current config
        info1 = f'H[{self.h_min},{self.h_max}] S[{self.s_min},{self.s_max}] V[{self.v_min},{self.v_max}]'
        info2 = f'OPEN={self.open_kernel} CLOSE={self.close_kernel}'

        cv2.putText(bgr, info1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(bgr, info2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(self.window_name_original, bgr)
        cv2.imshow(self.window_name_mask, mask)

        if self.frame_count % self.print_hsv_every_n_frames == 0:
            self.get_logger().info(
                f'Current HSV: '
                f'h_min={self.h_min}, h_max={self.h_max}, '
                f's_min={self.s_min}, s_max={self.s_max}, '
                f'v_min={self.v_min}, v_max={self.v_max}, '
                f'open_kernel={self.open_kernel}, close_kernel={self.close_kernel}'
            )

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('q pressed, shutting down...')
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = HSVTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received. Shutting down.')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()