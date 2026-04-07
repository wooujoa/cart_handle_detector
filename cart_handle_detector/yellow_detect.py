#!/usr/bin/env python3
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image


class YellowCartDetect(Node):
    def __init__(self):
        super().__init__('yellow_cart_detect')

        self.declare_parameter('color_topic', '/zedm/zed_node/left/image_rect_color')

        self.declare_parameter('center_green_px_topic', '/cart_handle/center_green_px_zed')
        self.declare_parameter('end_green_px_topic', '/cart_handle/end_green_px_zed')
        self.declare_parameter('yellow_axis_p0_px_topic', '/cart_handle/yellow_axis_p0_px_zed')
        self.declare_parameter('yellow_axis_p1_px_topic', '/cart_handle/yellow_axis_p1_px_zed')
        self.declare_parameter('debug_image_topic', '/cart_handle/debug_image_zed')

        self.declare_parameter('camera_frame_override', '')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_log', True)
        self.declare_parameter('log_green_contours', True)
        self.declare_parameter('log_yellow_contours', True)

        self.declare_parameter('yellow_h_min', 6)
        self.declare_parameter('yellow_h_max', 44)
        self.declare_parameter('yellow_s_min', 113)
        self.declare_parameter('yellow_s_max', 255)
        self.declare_parameter('yellow_v_min', 229)
        self.declare_parameter('yellow_v_max', 255)
        self.declare_parameter('yellow_open_kernel', 1)
        self.declare_parameter('yellow_close_kernel', 9)

        self.declare_parameter('green_h_min', 52)
        self.declare_parameter('green_h_max', 108)
        self.declare_parameter('green_s_min', 192)
        self.declare_parameter('green_s_max', 255)
        self.declare_parameter('green_v_min', 99)
        self.declare_parameter('green_v_max', 211)
        self.declare_parameter('green_open_kernel', 1)
        self.declare_parameter('green_close_kernel', 1)

        self.declare_parameter('green_min_area', 5.0)
        self.declare_parameter('green_max_area', 10000.0)
        self.declare_parameter('green_min_circularity', 0.15)

        self.declare_parameter('yellow_min_area', 80.0)
        self.declare_parameter('yellow_min_aspect_ratio', 2.0)
        self.declare_parameter('yellow_merge_top_k', 2)

        self.declare_parameter('green_line_weight', 24)
        self.declare_parameter('line_margin_px', 6.0)

        self.color_topic = self.get_parameter('color_topic').value
        self.center_green_px_topic = self.get_parameter('center_green_px_topic').value
        self.end_green_px_topic = self.get_parameter('end_green_px_topic').value
        self.yellow_axis_p0_px_topic = self.get_parameter('yellow_axis_p0_px_topic').value
        self.yellow_axis_p1_px_topic = self.get_parameter('yellow_axis_p1_px_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value

        self.camera_frame_override = self.get_parameter('camera_frame_override').value
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.debug_log = bool(self.get_parameter('debug_log').value)
        self.log_green_contours = bool(self.get_parameter('log_green_contours').value)
        self.log_yellow_contours = bool(self.get_parameter('log_yellow_contours').value)

        self.yellow_h_min = int(self.get_parameter('yellow_h_min').value)
        self.yellow_h_max = int(self.get_parameter('yellow_h_max').value)
        self.yellow_s_min = int(self.get_parameter('yellow_s_min').value)
        self.yellow_s_max = int(self.get_parameter('yellow_s_max').value)
        self.yellow_v_min = int(self.get_parameter('yellow_v_min').value)
        self.yellow_v_max = int(self.get_parameter('yellow_v_max').value)
        self.yellow_open_kernel = int(self.get_parameter('yellow_open_kernel').value)
        self.yellow_close_kernel = int(self.get_parameter('yellow_close_kernel').value)

        self.green_h_min = int(self.get_parameter('green_h_min').value)
        self.green_h_max = int(self.get_parameter('green_h_max').value)
        self.green_s_min = int(self.get_parameter('green_s_min').value)
        self.green_s_max = int(self.get_parameter('green_s_max').value)
        self.green_v_min = int(self.get_parameter('green_v_min').value)
        self.green_v_max = int(self.get_parameter('green_v_max').value)
        self.green_open_kernel = int(self.get_parameter('green_open_kernel').value)
        self.green_close_kernel = int(self.get_parameter('green_close_kernel').value)

        self.green_min_area = float(self.get_parameter('green_min_area').value)
        self.green_max_area = float(self.get_parameter('green_max_area').value)
        self.green_min_circularity = float(self.get_parameter('green_min_circularity').value)

        self.yellow_min_area = float(self.get_parameter('yellow_min_area').value)
        self.yellow_min_aspect_ratio = float(self.get_parameter('yellow_min_aspect_ratio').value)
        self.yellow_merge_top_k = int(self.get_parameter('yellow_merge_top_k').value)

        self.green_line_weight = max(1, int(self.get_parameter('green_line_weight').value))
        self.line_margin_px = float(self.get_parameter('line_margin_px').value)

        self.bridge = CvBridge()
        self.frame_index = 0

        self.sub_color = self.create_subscription(Image, self.color_topic, self.color_callback, qos_profile_sensor_data)

        self.pub_center_green = self.create_publisher(PointStamped, self.center_green_px_topic, 100)
        self.pub_end_green = self.create_publisher(PointStamped, self.end_green_px_topic, 100)
        self.pub_yellow_p0 = self.create_publisher(PointStamped, self.yellow_axis_p0_px_topic, 100)
        self.pub_yellow_p1 = self.create_publisher(PointStamped, self.yellow_axis_p1_px_topic, 100)
        self.pub_debug = self.create_publisher(Image, self.debug_image_topic, 10)

        self.get_logger().info('Yellow Cart Detect Initialized (plane fused)')

    def color_callback(self, msg: Image):
        self.frame_index += 1
        try:
            color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Color image conversion failed: {repr(e)}')
            return

        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        yellow_mask = self.make_mask(
            hsv,
            self.yellow_h_min, self.yellow_h_max,
            self.yellow_s_min, self.yellow_s_max,
            self.yellow_v_min, self.yellow_v_max,
            self.yellow_open_kernel, self.yellow_close_kernel,
        )
        green_mask = self.make_mask(
            hsv,
            self.green_h_min, self.green_h_max,
            self.green_s_min, self.green_s_max,
            self.green_v_min, self.green_v_max,
            self.green_open_kernel, self.green_close_kernel,
        )

        yellow_points, yellow_infos = self.extract_yellow_support_points(yellow_mask)
        greens, green_infos = self.detect_green_stickers(green_mask)
        debug_img = color.copy()

        if self.debug_log:
            self.get_logger().info(
                f'[frame {self.frame_index}] '
                f'yellow_mask_nz={int(np.count_nonzero(yellow_mask))} '
                f'green_mask_nz={int(np.count_nonzero(green_mask))} '
                f'green_candidates={len(greens)}'
            )

        if self.log_yellow_contours and self.debug_log:
            for info in yellow_infos:
                txt = (
                    f'[frame {self.frame_index}] [YELLOW #{info["idx"]}] '
                    f'area={info["area"]:.1f} aspect={info["aspect_ratio"]:.2f} '
                    f'bbox=({info["x"]},{info["y"]},{info["w"]},{info["h"]}) '
                    f'status={info["status"]}'
                )
                if info.get('u') is not None and info.get('v') is not None:
                    txt += f' centroid=({info["u"]:.1f},{info["v"]:.1f})'
                self.get_logger().info(txt)

        if self.log_green_contours and self.debug_log:
            for info in green_infos:
                txt = (
                    f'[frame {self.frame_index}] [GREEN #{info["idx"]}] '
                    f'area={info["area"]:.1f} circ={info["circularity"]:.3f} '
                    f'bbox=({info["x"]},{info["y"]},{info["w"]},{info["h"]}) '
                    f'status={info["status"]}'
                )
                if info['u'] is not None and info['v'] is not None:
                    txt += f' centroid=({info["u"]:.1f},{info["v"]:.1f})'
                self.get_logger().info(txt)

        if yellow_points is None or yellow_points.shape[0] < 20:
            if self.debug_log:
                self.get_logger().warn(f'[frame {self.frame_index}] yellow support not found')
            self.publish_debug(debug_img, msg.header)
            return

        if len(greens) < 2:
            if self.debug_log:
                self.get_logger().warn(f'[frame {self.frame_index}] green stickers not enough: {len(greens)}')
            self.publish_debug(debug_img, msg.header)
            return

        green_centers = np.array(greens, dtype=np.float32)

        yellow_axis, yellow_mid = self.fit_line_from_points(yellow_points)
        if yellow_axis is None or yellow_mid is None:
            if self.debug_log:
                self.get_logger().warn(f'[frame {self.frame_index}] yellow fitLine failed')
            self.publish_debug(debug_img, msg.header)
            return

        center_green_uv, end_green_uv = self.classify_green_stickers(green_centers, yellow_mid, yellow_axis)
        green_vec = np.array(end_green_uv - center_green_uv, dtype=np.float32)
        n_green = np.linalg.norm(green_vec)
        if n_green < 1e-6:
            if self.debug_log:
                self.get_logger().warn(f'[frame {self.frame_index}] green vector too small')
            self.publish_debug(debug_img, msg.header)
            return
        green_unit = green_vec / n_green

        if float(np.dot(yellow_axis, green_unit)) < 0.0:
            yellow_axis = -yellow_axis

        support_points = [yellow_points]
        weighted_green = np.repeat(np.vstack([center_green_uv, end_green_uv]).astype(np.float32), self.green_line_weight, axis=0)
        support_points.append(weighted_green)
        fused_support = np.vstack(support_points).astype(np.float32)

        fused_axis, fused_mid = self.fit_line_from_points(fused_support)
        if fused_axis is None or fused_mid is None:
            if self.debug_log:
                self.get_logger().warn(f'[frame {self.frame_index}] fused fitLine failed')
            self.publish_debug(debug_img, msg.header)
            return

        if float(np.dot(fused_axis, green_unit)) < 0.0:
            fused_axis = -fused_axis

        endpoint_support = np.vstack([yellow_points, center_green_uv[None, :], end_green_uv[None, :]]).astype(np.float32)
        p0_uv, p1_uv = self.endpoints_from_support(endpoint_support, fused_mid, fused_axis, margin_px=self.line_margin_px)
        if p0_uv is None or p1_uv is None:
            if self.debug_log:
                self.get_logger().warn(f'[frame {self.frame_index}] fused endpoints failed')
            self.publish_debug(debug_img, msg.header)
            return

        frame_id = self.camera_frame_override if self.camera_frame_override else msg.header.frame_id
        self.pub_center_green.publish(self.make_uv_point(center_green_uv, msg.header, frame_id))
        self.pub_end_green.publish(self.make_uv_point(end_green_uv, msg.header, frame_id))
        self.pub_yellow_p0.publish(self.make_uv_point(p0_uv, msg.header, frame_id))
        self.pub_yellow_p1.publish(self.make_uv_point(p1_uv, msg.header, frame_id))

        self.draw_fused(debug_img, p0_uv, p1_uv, center_green_uv, end_green_uv)
        self.publish_debug(debug_img, msg.header)

        if self.debug_log:
            self.get_logger().info(
                f'[frame {self.frame_index}] fused_line '
                f'p0=({p0_uv[0]:.1f},{p0_uv[1]:.1f}) '
                f'p1=({p1_uv[0]:.1f},{p1_uv[1]:.1f}) '
                f'axis=({fused_axis[0]:.4f},{fused_axis[1]:.4f}) '
                f'center_green=({center_green_uv[0]:.1f},{center_green_uv[1]:.1f}) '
                f'end_green=({end_green_uv[0]:.1f},{end_green_uv[1]:.1f})'
            )

    @staticmethod
    def make_mask(hsv, h_min, h_max, s_min, s_max, v_min, v_max, open_k, close_k):
        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        if open_k > 1:
            kernel = np.ones((open_k, open_k), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if close_k > 1:
            kernel = np.ones((close_k, close_k), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def detect_green_stickers(self, green_mask: np.ndarray):
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers: List[Tuple[float, float]] = []
        infos: List[Dict[str, Any]] = []

        for idx, cnt in enumerate(contours):
            area = float(cv2.contourArea(cnt))
            x, y, w, h = cv2.boundingRect(cnt)
            peri = float(cv2.arcLength(cnt, True))
            circularity = 0.0 if peri < 1e-6 else 4.0 * math.pi * area / (peri * peri)

            info = {
                'idx': idx, 'area': area, 'circularity': circularity,
                'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                'u': None, 'v': None, 'status': 'unknown',
            }

            if area < self.green_min_area:
                info['status'] = 'reject_area_small'
                infos.append(info)
                continue
            if area > self.green_max_area:
                info['status'] = 'reject_area_large'
                infos.append(info)
                continue
            if circularity < self.green_min_circularity:
                info['status'] = 'reject_circularity'
                infos.append(info)
                continue

            M = cv2.moments(cnt)
            if abs(M['m00']) < 1e-6:
                info['status'] = 'reject_zero_m00'
                infos.append(info)
                continue

            u = float(M['m10'] / M['m00'])
            v = float(M['m01'] / M['m00'])
            info['u'] = u
            info['v'] = v
            info['status'] = 'accepted'
            infos.append(info)
            centers.append((u, v))

        return centers, infos

    def extract_yellow_support_points(self, yellow_mask: np.ndarray):
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        infos: List[Dict[str, Any]] = []
        accepted_masks: List[np.ndarray] = []
        accepted_meta: List[Tuple[float, np.ndarray]] = []

        for idx, cnt in enumerate(contours):
            area = float(cv2.contourArea(cnt))
            x, y, w, h = cv2.boundingRect(cnt)
            rect = cv2.minAreaRect(cnt)
            (_, _), (rw, rh), _ = rect
            long_side = max(rw, rh)
            short_side = max(min(rw, rh), 1e-6)
            aspect_ratio = long_side / short_side

            info = {
                'idx': idx, 'area': area, 'aspect_ratio': aspect_ratio,
                'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                'u': None, 'v': None, 'status': 'unknown',
            }

            if area < self.yellow_min_area:
                info['status'] = 'reject_area_small'
                infos.append(info)
                continue
            if aspect_ratio < self.yellow_min_aspect_ratio:
                info['status'] = 'reject_aspect'
                infos.append(info)
                continue

            M = cv2.moments(cnt)
            if abs(M['m00']) < 1e-6:
                info['status'] = 'reject_zero_m00'
                infos.append(info)
                continue

            u = float(M['m10'] / M['m00'])
            v = float(M['m01'] / M['m00'])
            info['u'] = u
            info['v'] = v
            info['status'] = 'accepted_candidate'
            infos.append(info)

            accepted_meta.append((area, cnt))

        if len(accepted_meta) < 2:
            return None, infos

        accepted_meta.sort(key=lambda x: x[0], reverse=True)
        selected = accepted_meta[:max(2, self.yellow_merge_top_k)]
        selected = selected[:2]

        union_mask = np.zeros_like(yellow_mask)
        for _, cnt in selected:
            cv2.drawContours(union_mask, [cnt], -1, 255, thickness=-1)

        ys, xs = np.nonzero(union_mask)
        if xs.size < 20:
            return None, infos

        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        return pts, infos

    @staticmethod
    def fit_line_from_points(points: np.ndarray):
        if points is None or len(points) < 2:
            return None, None
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        try:
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        except Exception:
            return None, None
        axis = np.array([float(vx), float(vy)], dtype=np.float32).reshape(2)
        n = np.linalg.norm(axis)
        if n < 1e-6:
            return None, None
        axis /= n
        mid = np.array([float(x0), float(y0)], dtype=np.float32).reshape(2)
        return axis, mid

    @staticmethod
    def classify_green_stickers(green_centers: np.ndarray, line_mid_uv: np.ndarray, axis_unit: np.ndarray):
        rel = green_centers - line_mid_uv[None, :]
        proj = rel @ axis_unit
        if len(green_centers) > 2:
            idx = np.argsort(np.abs(proj))[::-1][:2]
            candidates = green_centers[idx]
        else:
            candidates = green_centers

        rel2 = candidates - line_mid_uv[None, :]
        proj2 = rel2 @ axis_unit
        idx_center = int(np.argmin(np.abs(proj2)))
        idx_end = 1 - idx_center
        return candidates[idx_center], candidates[idx_end]

    @staticmethod
    def endpoints_from_support(points: np.ndarray, line_mid: np.ndarray, axis_unit: np.ndarray, margin_px: float = 0.0):
        if points is None or points.shape[0] < 2:
            return None, None
        rel = points - line_mid[None, :]
        proj = rel @ axis_unit
        i0 = int(np.argmin(proj))
        i1 = int(np.argmax(proj))
        p0 = points[i0].astype(np.float32)
        p1 = points[i1].astype(np.float32)
        if margin_px > 1e-6:
            p0 = p0 - axis_unit * margin_px
            p1 = p1 + axis_unit * margin_px
        return p0, p1

    @staticmethod
    def make_uv_point(uv, header, frame_id):
        msg = PointStamped()
        msg.header = header
        msg.header.frame_id = frame_id
        msg.point.x = float(uv[0])
        msg.point.y = float(uv[1])
        msg.point.z = 0.0
        return msg

    @staticmethod
    def draw_fused(img, p0_uv, p1_uv, center_green_uv, end_green_uv):
        p0 = (int(round(p0_uv[0])), int(round(p0_uv[1])))
        p1 = (int(round(p1_uv[0])), int(round(p1_uv[1])))
        cg = (int(round(center_green_uv[0])), int(round(center_green_uv[1])))
        eg = (int(round(end_green_uv[0])), int(round(end_green_uv[1])))
        cv2.line(img, p0, p1, (0, 255, 255), 3)
        cv2.circle(img, p0, 6, (255, 255, 0), -1)
        cv2.circle(img, p1, 6, (255, 255, 0), -1)
        cv2.circle(img, cg, 7, (255, 0, 0), -1)
        cv2.circle(img, eg, 7, (0, 255, 0), -1)
        cv2.line(img, cg, eg, (0, 255, 0), 2)

    def publish_debug(self, bgr: np.ndarray, header):
        if not self.publish_debug_image:
            return
        try:
            msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
            msg.header = header
            self.pub_debug.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Failed to publish debug image: {repr(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = YellowCartDetect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()