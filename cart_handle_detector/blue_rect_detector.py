#!/usr/bin/env python3
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, CameraInfo, CompressedImage


class BlueBlobDetectorZed(Node):
    def __init__(self):
        super().__init__('blue_blob_detector_zed')

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter('color_topic', '/zedm/zed_node/left/image_rect_color/compressed')
        self.declare_parameter('depth_topic', '/zedm/zed_node/depth/depth_registered')
        self.declare_parameter('camera_info_topic', '/zedm/zed_node/left/camera_info')

        self.declare_parameter('output_points_topic', '/cart_handle/blue_points_zed')
        self.declare_parameter('debug_image_topic', '/cart_handle/debug_image_zed')

        self.declare_parameter('camera_frame_override', '')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_log', True)
        self.declare_parameter('log_all_contours', True)
        self.declare_parameter('log_depth_debug', True)

        # HSV
        self.declare_parameter('h_min', 72)
        self.declare_parameter('h_max', 132)
        self.declare_parameter('s_min', 138)
        self.declare_parameter('s_max', 255)
        self.declare_parameter('v_min', 199)
        self.declare_parameter('v_max', 255)

        # Blob filtering
        self.declare_parameter('min_area', 5.0)
        self.declare_parameter('max_area', 100000.0)
        self.declare_parameter('max_blobs', 100)   # 0 or negative => unlimited

        # Morphology
        self.declare_parameter('open_kernel', 1)
        self.declare_parameter('close_kernel', 3)

        # Depth filtering
        self.declare_parameter('depth_roi_half', 4)
        self.declare_parameter('min_depth_m', 0.05)
        self.declare_parameter('max_depth_m', 3.0)

        # Timestamp debug / filter
        self.declare_parameter('max_color_depth_dt_ms', -1.0)  # <=0: no skip, only log

        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.output_points_topic = self.get_parameter('output_points_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value

        self.camera_frame_override = self.get_parameter('camera_frame_override').value
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.debug_log = bool(self.get_parameter('debug_log').value)
        self.log_all_contours = bool(self.get_parameter('log_all_contours').value)
        self.log_depth_debug = bool(self.get_parameter('log_depth_debug').value)

        self.h_min = int(self.get_parameter('h_min').value)
        self.h_max = int(self.get_parameter('h_max').value)
        self.s_min = int(self.get_parameter('s_min').value)
        self.s_max = int(self.get_parameter('s_max').value)
        self.v_min = int(self.get_parameter('v_min').value)
        self.v_max = int(self.get_parameter('v_max').value)

        self.min_area = float(self.get_parameter('min_area').value)
        self.max_area = float(self.get_parameter('max_area').value)
        self.max_blobs = int(self.get_parameter('max_blobs').value)

        self.open_kernel = int(self.get_parameter('open_kernel').value)
        self.close_kernel = int(self.get_parameter('close_kernel').value)

        self.depth_roi_half = int(self.get_parameter('depth_roi_half').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)

        self.max_color_depth_dt_ms = float(self.get_parameter('max_color_depth_dt_ms').value)

        # --------------------------------------------------
        # State
        # --------------------------------------------------
        self.bridge = CvBridge()
        self.latest_depth_msg: Optional[Image] = None
        self.camera_info: Optional[CameraInfo] = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.frame_index = 0

        # --------------------------------------------------
        # Pub/Sub
        # --------------------------------------------------
        self.sub_color = self.create_subscription(
            CompressedImage, self.color_topic, self.color_callback, qos_profile_sensor_data
        )
        self.sub_depth = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data
        )
        self.sub_info = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data
        )

        self.pub_points = self.create_publisher(PointStamped, self.output_points_topic, 100)
        self.pub_debug = self.create_publisher(Image, self.debug_image_topic, 10)

        self.get_logger().info('========================================')
        self.get_logger().info('Blue Blob Detector Initialized (ZED, debug version)')
        self.get_logger().info(f'color_topic             : {self.color_topic}')
        self.get_logger().info(f'depth_topic             : {self.depth_topic}')
        self.get_logger().info(f'camera_info_topic       : {self.camera_info_topic}')
        self.get_logger().info(f'output_points_topic     : {self.output_points_topic}')
        self.get_logger().info(f'debug_image_topic       : {self.debug_image_topic}')
        self.get_logger().info(f'camera_frame_override   : {self.camera_frame_override}')
        self.get_logger().info(
            f'HSV = H[{self.h_min},{self.h_max}] '
            f'S[{self.s_min},{self.s_max}] '
            f'V[{self.v_min},{self.v_max}]'
        )
        self.get_logger().info(
            f'min_area={self.min_area}, max_area={self.max_area}, max_blobs={self.max_blobs}'
        )
        self.get_logger().info(
            f'open_kernel={self.open_kernel}, close_kernel={self.close_kernel}'
        )
        self.get_logger().info(
            f'depth_roi_half={self.depth_roi_half}, '
            f'depth_range=[{self.min_depth_m:.3f}, {self.max_depth_m:.3f}] m'
        )
        self.get_logger().info(
            f'max_color_depth_dt_ms={self.max_color_depth_dt_ms} (<=0 means no skip)'
        )
        self.get_logger().info('========================================')

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------
    def depth_callback(self, msg: Image):
        self.latest_depth_msg = msg

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def color_callback(self, msg: CompressedImage):
        if self.latest_depth_msg is None or self.camera_info is None:
            return

        self.frame_index += 1

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if color is None:
                self.get_logger().error('Compressed color decode failed')
                return
        except Exception as e:
            self.get_logger().error(f'Color image conversion failed: {repr(e)}')
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth image conversion failed: {repr(e)}')
            return

        debug_img = color.copy()

        color_t = self.stamp_to_sec(msg.header.stamp)
        depth_t = self.stamp_to_sec(self.latest_depth_msg.header.stamp)
        dt_ms = abs(color_t - depth_t) * 1000.0

        if self.debug_log:
            self.get_logger().info(
                f'[frame {self.frame_index}] color_stamp={color_t:.6f} '
                f'depth_stamp={depth_t:.6f} dt={dt_ms:.2f} ms'
            )

        if self.max_color_depth_dt_ms > 0.0 and dt_ms > self.max_color_depth_dt_ms:
            if self.debug_log:
                self.get_logger().warn(
                    f'[frame {self.frame_index}] skip frame: '
                    f'color-depth dt too large ({dt_ms:.2f} ms > {self.max_color_depth_dt_ms:.2f} ms)'
                )
            if self.publish_debug_image:
                cv2.putText(debug_img, f'SKIP dt={dt_ms:.1f} ms', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                self.publish_debug(debug_img, msg.header)
            return

        detections, mask, contour_infos = self.find_all_blue_blobs(color)

        mask_nonzero = int(np.count_nonzero(mask))
        raw_contours = len(contour_infos)
        accepted_contours = len(detections)

        if self.debug_log:
            self.get_logger().info(
                f'[frame {self.frame_index}] mask_nonzero={mask_nonzero} '
                f'raw_contours={raw_contours} accepted={accepted_contours}'
            )

        if self.log_all_contours and self.debug_log:
            for info in contour_infos:
                self.get_logger().info(
                    f'[frame {self.frame_index}] contour#{info["raw_idx"]}: '
                    f'area={info["area"]:.1f} bbox=({info["x"]},{info["y"]},{info["w"]},{info["h"]}) '
                    f'status={info["status"]}'
                    + (
                        f' centroid=({info["u"]},{info["v"]})'
                        if info["u"] is not None and info["v"] is not None
                        else ''
                    )
                )

        if len(detections) == 0:
            if self.publish_debug_image:
                cv2.putText(debug_img, 'No accepted blue blobs', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(debug_img,
                            f'mask_nz={mask_nonzero} raw={raw_contours} accepted=0',
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                self.publish_debug(debug_img, msg.header)
            return

        published_count = 0

        for idx, det in enumerate(detections):
            contour = det['contour']
            u = det['u']
            v = det['v']
            area = det['area']
            raw_idx = det['raw_idx']

            if self.debug_log:
                self.get_logger().info(
                    f'[frame {self.frame_index}] accepted_blob#{idx} '
                    f'(raw_contour#{raw_idx}) start: pixel=({u},{v}) area={area:.1f}'
                )

            depth_m, depth_dbg = self.get_depth_median_debug(
                depth, u, v, self.depth_roi_half, self.latest_depth_msg.encoding
            )

            if self.log_depth_debug and self.debug_log:
                self.get_logger().info(
                    f'[frame {self.frame_index}] accepted_blob#{idx} depth_debug: '
                    f'roi=({depth_dbg["u0"]}:{depth_dbg["u1"]}, {depth_dbg["v0"]}:{depth_dbg["v1"]}) '
                    f'roi_shape={depth_dbg["roi_shape"]} total={depth_dbg["total_px"]} '
                    f'valid={depth_dbg["valid_px"]} '
                    f'median_all={depth_dbg["median_all"]} '
                    f'reason={depth_dbg["reason"]}'
                )

            if depth_m is None:
                if self.debug_log:
                    self.get_logger().warn(
                        f'[frame {self.frame_index}] accepted_blob#{idx} DROP: invalid depth | '
                        f'pixel=({u},{v}) area={area:.1f} reason={depth_dbg["reason"]}'
                    )

                cv2.drawContours(debug_img, [contour], -1, (0, 0, 255), 2)
                cv2.circle(debug_img, (u, v), 5, (0, 0, 255), -1)
                cv2.putText(
                    debug_img,
                    f'#{idx} invalid depth a={area:.1f}',
                    (u + 10, max(20, v - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
                continue

            point_xyz = self.project_pixel_to_3d(u, v, depth_m)
            if point_xyz is None:
                if self.debug_log:
                    self.get_logger().warn(
                        f'[frame {self.frame_index}] accepted_blob#{idx} DROP: projection failed | '
                        f'pixel=({u},{v}) area={area:.1f}'
                    )

                cv2.drawContours(debug_img, [contour], -1, (0, 0, 255), 2)
                cv2.circle(debug_img, (u, v), 5, (0, 0, 255), -1)
                cv2.putText(
                    debug_img,
                    f'#{idx} proj fail a={area:.1f}',
                    (u + 10, max(20, v - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
                continue

            x, y, z = point_xyz

            point_msg = PointStamped()
            point_msg.header.stamp = msg.header.stamp
            point_msg.header.frame_id = (
                self.camera_frame_override
                if self.camera_frame_override
                else self.latest_depth_msg.header.frame_id
            )
            point_msg.point.x = float(x)
            point_msg.point.y = float(y)
            point_msg.point.z = float(z)
            self.pub_points.publish(point_msg)
            published_count += 1

            cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)
            cv2.circle(debug_img, (u, v), 6, (0, 0, 255), -1)
            cv2.putText(
                debug_img,
                f'#{idx} ({x:.3f}, {y:.3f}, {z:.3f})',
                (u + 10, max(20, v - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            if self.debug_log:
                self.get_logger().info(
                    f'blob#{idx}: pixel=({u},{v}) area={area:.1f} '
                    f'camera_xyz=({x:.4f}, {y:.4f}, {z:.4f}) '
                    f'frame={point_msg.header.frame_id}'
                )

        if self.publish_debug_image:
            cv2.putText(
                debug_img,
                f'raw={raw_contours} accepted={accepted_contours} published={published_count}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            cv2.putText(
                debug_img,
                f'mask_nz={mask_nonzero} dt={dt_ms:.1f}ms',
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            self.publish_debug(debug_img, msg.header)

    # --------------------------------------------------
    # Blue blob detection
    # --------------------------------------------------
    def find_all_blue_blobs(self, bgr: np.ndarray):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        lower = np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)
        upper = np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        if self.open_kernel > 1:
            k_open = np.ones((self.open_kernel, self.open_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

        if self.close_kernel > 1:
            k_close = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_infos: List[Dict[str, Any]] = []
        detections: List[Dict[str, Any]] = []

        for raw_idx, cnt in enumerate(contours):
            area = float(cv2.contourArea(cnt))
            x, y, w, h = cv2.boundingRect(cnt)

            info = {
                'raw_idx': raw_idx,
                'area': area,
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h),
                'u': None,
                'v': None,
                'status': 'unknown',
            }

            if area < self.min_area:
                info['status'] = 'reject_area_small'
                contour_infos.append(info)
                continue

            if area > self.max_area:
                info['status'] = 'reject_area_large'
                contour_infos.append(info)
                continue

            M = cv2.moments(cnt)
            if abs(M['m00']) < 1e-6:
                info['status'] = 'reject_zero_m00'
                contour_infos.append(info)
                continue

            u = int(M['m10'] / M['m00'])
            v = int(M['m01'] / M['m00'])

            info['u'] = u
            info['v'] = v
            info['status'] = 'accepted_pre_sort'
            contour_infos.append(info)

            detections.append({
                'raw_idx': raw_idx,
                'contour': cnt,
                'u': u,
                'v': v,
                'area': area,
            })

        detections.sort(key=lambda d: d['u'])

        if self.max_blobs > 0 and len(detections) > self.max_blobs:
            keep_raw_idx = {d['raw_idx'] for d in detections[:self.max_blobs]}
            for info in contour_infos:
                if info['status'] == 'accepted_pre_sort' and info['raw_idx'] not in keep_raw_idx:
                    info['status'] = 'reject_max_blobs'
            detections = detections[:self.max_blobs]

        final_keep_raw_idx = {d['raw_idx'] for d in detections}
        for info in contour_infos:
            if info['status'] == 'accepted_pre_sort' and info['raw_idx'] in final_keep_raw_idx:
                info['status'] = 'accepted'

        return detections, mask, contour_infos

    # --------------------------------------------------
    # Depth / 3D
    # --------------------------------------------------
    def get_depth_median_debug(
        self,
        depth_img: np.ndarray,
        u: int,
        v: int,
        half: int,
        encoding: str
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        h, w = depth_img.shape[:2]

        u0 = max(0, u - half)
        u1 = min(w, u + half + 1)
        v0 = max(0, v - half)
        v1 = min(h, v + half + 1)

        roi = depth_img[v0:v1, u0:u1]

        dbg = {
            'u0': u0,
            'u1': u1,
            'v0': v0,
            'v1': v1,
            'roi_shape': tuple(roi.shape),
            'total_px': int(roi.size),
            'valid_px': 0,
            'median_all': 'None',
            'reason': 'unknown',
        }

        if roi.size == 0:
            dbg['reason'] = 'empty_roi'
            return None, dbg

        if '16UC1' in encoding:
            vals = roi[roi > 0].astype(np.float32) * 0.001
        else:
            vals = roi[np.isfinite(roi)]
            vals = vals[vals > 0].astype(np.float32)

        dbg['valid_px'] = int(vals.size)

        if vals.size == 0:
            dbg['reason'] = 'no_positive_depth'
            return None, dbg

        median_all = float(np.median(vals))
        dbg['median_all'] = f'{median_all:.4f}'

        if median_all < self.min_depth_m:
            dbg['reason'] = f'median_below_min({median_all:.4f}<{self.min_depth_m:.4f})'
            return None, dbg

        if median_all > self.max_depth_m:
            dbg['reason'] = f'median_above_max({median_all:.4f}>{self.max_depth_m:.4f})'
            return None, dbg

        dbg['reason'] = 'ok'
        return median_all, dbg

    def project_pixel_to_3d(self, u: int, v: int, depth_m: float) -> Optional[Tuple[float, float, float]]:
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            return None

        x = (float(u) - self.cx) * depth_m / self.fx
        y = (float(v) - self.cy) * depth_m / self.fy
        z = depth_m
        return x, y, z

    # --------------------------------------------------
    # Utils
    # --------------------------------------------------
    @staticmethod
    def stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def publish_debug(self, bgr: np.ndarray, header):
        try:
            msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
            msg.header = header
            self.pub_debug.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Failed to publish debug image: {repr(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = BlueBlobDetectorZed()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()