#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
feature_detect_center_ygy.py

RGB-only feature detector.

Core idea:
  1) Do NOT connect arbitrary green pairs.
  2) First find a CENTER green marker whose local neighborhood is Y-G-Y
     (yellow on both sides of the center green).
  3) Use that local Y-G-Y to determine the handle axis.
  4) Search the END green marker only inside a narrow corridor around that axis.
  5) If no end green exists near the center Y-G-Y axis, publish nothing.

Physical pattern along handle:
  0.0 ~ 3.0 cm      : green end marker
  3.0 ~ 25.7 cm     : yellow
  25.7 ~ 28.7 cm    : green center marker
  28.7 ~ 54.5 cm    : yellow

Publishes the same topics as the previous node:
  /cart_handle/end_green_px_zed
  /cart_handle/center_green_px_zed
  /cart_handle/yellow_axis_p0_px_zed
  /cart_handle/yellow_axis_p1_px_zed
"""

import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image


class FeatureDetectCenterYGY(Node):
    def __init__(self):
        super().__init__('feature_detect_center_ygy')

        # Topics
        self.declare_parameter('color_topic', '/zedm/zed_node/rgb/image_rect_color')
        self.declare_parameter('end_green_px_topic', '/cart_handle/end_green_px_zed')
        self.declare_parameter('center_green_px_topic', '/cart_handle/center_green_px_zed')
        self.declare_parameter('yellow_axis_p0_px_topic', '/cart_handle/yellow_axis_p0_px_zed')
        self.declare_parameter('yellow_axis_p1_px_topic', '/cart_handle/yellow_axis_p1_px_zed')
        self.declare_parameter('debug_image_topic', '/cart_handle/debug_image_zed')
        self.declare_parameter('camera_frame_override', '')

        # Debug/profile
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_log', True)
        self.declare_parameter('profile', True)
        self.declare_parameter('profile_period_sec', 2.0)

        # HSV thresholds
        self.declare_parameter('yellow_h_min', 18)
        self.declare_parameter('yellow_h_max', 42)
        self.declare_parameter('yellow_s_min', 60)
        self.declare_parameter('yellow_s_max', 255)
        self.declare_parameter('yellow_v_min', 120)
        self.declare_parameter('yellow_v_max', 255)


        self.declare_parameter('green_h_min', 37)
        self.declare_parameter('green_h_max', 128)
        self.declare_parameter('green_s_min', 53)
        self.declare_parameter('green_s_max', 255)
        self.declare_parameter('green_v_min', 100)
        self.declare_parameter('green_v_max', 255)

        self.declare_parameter('open_kernel', 1)
        self.declare_parameter('close_kernel', 3)

        # Green component filtering
        self.declare_parameter('green_min_area', 18.0)
        self.declare_parameter('green_max_area', 30000.0)
        self.declare_parameter('max_green_candidates', 10)

        # Physical layout [cm]
        self.declare_parameter('handle_length_cm', 54.5)
        self.declare_parameter('end_green_start_cm', 0.0)
        self.declare_parameter('end_green_end_cm', 3.0)
        self.declare_parameter('center_green_start_cm', 25.7)
        self.declare_parameter('center_green_end_cm', 28.7)

        # Center Y-G-Y axis search
        self.declare_parameter('angle_search_step_deg', 3.0)
        self.declare_parameter('center_axis_local_radius_px', 230.0)
        self.declare_parameter('center_axis_corridor_half_width_px', 16.0)
        self.declare_parameter('center_axis_min_side_yellow_px', 25)
        self.declare_parameter('center_axis_min_balance', 0.35)
        self.declare_parameter('center_axis_min_s_px', 8.0)
        self.declare_parameter('center_axis_max_s_px', 260.0)

        # End green must lie on the center Y-G-Y axis corridor
        self.declare_parameter('end_green_max_perp_px', 18.0)
        self.declare_parameter('end_green_min_dist_px', 18.0)
        self.declare_parameter('end_green_max_dist_px', 260.0)

        # Endpoint sanity. This kills diagonal candidates whose p0/p1 fly out of image.
        self.declare_parameter('endpoint_margin_px', 25.0)

        # Pattern validation after end green is selected
        self.declare_parameter('pattern_corridor_half_width_px', 26.0)
        self.declare_parameter('min_yellow_total_pixels', 60)
        self.declare_parameter('min_yellow_left_pixels', 20)
        self.declare_parameter('min_yellow_right_pixels', 20)
        self.declare_parameter('min_green_end_pixels', 6)
        self.declare_parameter('min_green_center_pixels', 6)
        self.declare_parameter('min_yellow_balance', 0.55)

        # Green band length check in physical coordinates. Relaxed because mask may fragment.
        self.declare_parameter('check_green_band_length', True)
        self.declare_parameter('min_green_band_len_cm', 0.4)
        self.declare_parameter('max_green_band_len_cm', 8.0)

        # Scoring
        self.declare_parameter('wrong_color_penalty', 1.5)
        self.declare_parameter('end_perp_penalty', 2.0)

        # Load params
        self.color_topic = self.get_parameter('color_topic').value
        self.end_green_px_topic = self.get_parameter('end_green_px_topic').value
        self.center_green_px_topic = self.get_parameter('center_green_px_topic').value
        self.yellow_axis_p0_px_topic = self.get_parameter('yellow_axis_p0_px_topic').value
        self.yellow_axis_p1_px_topic = self.get_parameter('yellow_axis_p1_px_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.camera_frame_override = self.get_parameter('camera_frame_override').value

        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.debug_log = bool(self.get_parameter('debug_log').value)
        self.profile = bool(self.get_parameter('profile').value)
        self.profile_period_sec = float(self.get_parameter('profile_period_sec').value)

        self.yellow_h_min = int(self.get_parameter('yellow_h_min').value)
        self.yellow_h_max = int(self.get_parameter('yellow_h_max').value)
        self.yellow_s_min = int(self.get_parameter('yellow_s_min').value)
        self.yellow_s_max = int(self.get_parameter('yellow_s_max').value)
        self.yellow_v_min = int(self.get_parameter('yellow_v_min').value)
        self.yellow_v_max = int(self.get_parameter('yellow_v_max').value)

        self.green_h_min = int(self.get_parameter('green_h_min').value)
        self.green_h_max = int(self.get_parameter('green_h_max').value)
        self.green_s_min = int(self.get_parameter('green_s_min').value)
        self.green_s_max = int(self.get_parameter('green_s_max').value)
        self.green_v_min = int(self.get_parameter('green_v_min').value)
        self.green_v_max = int(self.get_parameter('green_v_max').value)

        self.open_kernel = int(self.get_parameter('open_kernel').value)
        self.close_kernel = int(self.get_parameter('close_kernel').value)

        self.green_min_area = float(self.get_parameter('green_min_area').value)
        self.green_max_area = float(self.get_parameter('green_max_area').value)
        self.max_green_candidates = int(self.get_parameter('max_green_candidates').value)

        self.handle_length_cm = float(self.get_parameter('handle_length_cm').value)
        self.end_green_start_cm = float(self.get_parameter('end_green_start_cm').value)
        self.end_green_end_cm = float(self.get_parameter('end_green_end_cm').value)
        self.center_green_start_cm = float(self.get_parameter('center_green_start_cm').value)
        self.center_green_end_cm = float(self.get_parameter('center_green_end_cm').value)

        self.end_green_center_cm = 0.5 * (self.end_green_start_cm + self.end_green_end_cm)
        self.center_green_center_cm = 0.5 * (self.center_green_start_cm + self.center_green_end_cm)
        self.green_gap_cm = self.center_green_center_cm - self.end_green_center_cm

        self.angle_search_step_deg = float(self.get_parameter('angle_search_step_deg').value)
        self.center_axis_local_radius_px = float(self.get_parameter('center_axis_local_radius_px').value)
        self.center_axis_corridor_half_width_px = float(self.get_parameter('center_axis_corridor_half_width_px').value)
        self.center_axis_min_side_yellow_px = int(self.get_parameter('center_axis_min_side_yellow_px').value)
        self.center_axis_min_balance = float(self.get_parameter('center_axis_min_balance').value)
        self.center_axis_min_s_px = float(self.get_parameter('center_axis_min_s_px').value)
        self.center_axis_max_s_px = float(self.get_parameter('center_axis_max_s_px').value)

        self.end_green_max_perp_px = float(self.get_parameter('end_green_max_perp_px').value)
        self.end_green_min_dist_px = float(self.get_parameter('end_green_min_dist_px').value)
        self.end_green_max_dist_px = float(self.get_parameter('end_green_max_dist_px').value)
        self.endpoint_margin_px = float(self.get_parameter('endpoint_margin_px').value)

        self.pattern_corridor_half_width_px = float(self.get_parameter('pattern_corridor_half_width_px').value)
        self.min_yellow_total_pixels = int(self.get_parameter('min_yellow_total_pixels').value)
        self.min_yellow_left_pixels = int(self.get_parameter('min_yellow_left_pixels').value)
        self.min_yellow_right_pixels = int(self.get_parameter('min_yellow_right_pixels').value)
        self.min_green_end_pixels = int(self.get_parameter('min_green_end_pixels').value)
        self.min_green_center_pixels = int(self.get_parameter('min_green_center_pixels').value)
        self.min_yellow_balance = float(self.get_parameter('min_yellow_balance').value)

        self.check_green_band_length = bool(self.get_parameter('check_green_band_length').value)
        self.min_green_band_len_cm = float(self.get_parameter('min_green_band_len_cm').value)
        self.max_green_band_len_cm = float(self.get_parameter('max_green_band_len_cm').value)

        self.wrong_color_penalty = float(self.get_parameter('wrong_color_penalty').value)
        self.end_perp_penalty = float(self.get_parameter('end_perp_penalty').value)

        if self.green_gap_cm <= 1e-9:
            raise RuntimeError('Invalid feature pattern: center green must be after end green.')

        self.bridge = CvBridge()
        self.cb_count = 0
        self.pub_count = 0
        self.frame_index = 0
        self.last_cb_ms = 0.0
        self.fail_counts = defaultdict(int)
        self.last_profile_print = time.perf_counter()

        self.sub_color = self.create_subscription(
            Image,
            self.color_topic,
            self.color_callback,
            qos_profile_sensor_data,
        )

        self.pub_end_green = self.create_publisher(PointStamped, self.end_green_px_topic, 100)
        self.pub_center_green = self.create_publisher(PointStamped, self.center_green_px_topic, 100)
        self.pub_yellow_p0 = self.create_publisher(PointStamped, self.yellow_axis_p0_px_topic, 100)
        self.pub_yellow_p1 = self.create_publisher(PointStamped, self.yellow_axis_p1_px_topic, 100)
        self.pub_debug = self.create_publisher(Image, self.debug_image_topic, 10)

        self.get_logger().info('=== feature_detect_center_ygy ready ===')
        self.get_logger().info(
            'CENTER Y-G-Y axis first -> end green only inside that axis corridor. '
            f'gap={self.green_gap_cm:.1f}cm, L={self.handle_length_cm:.1f}cm'
        )

    def color_callback(self, msg: Image):
        t0 = time.perf_counter()
        self.cb_count += 1
        self.frame_index += 1

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.fail('decode')
            self.get_logger().warn(f'image decode failed: {repr(e)}')
            self.finish_callback(t0)
            return

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        yellow_mask = self.make_mask(
            hsv,
            self.yellow_h_min,
            self.yellow_h_max,
            self.yellow_s_min,
            self.yellow_s_max,
            self.yellow_v_min,
            self.yellow_v_max,
        )
        green_mask = self.make_mask(
            hsv,
            self.green_h_min,
            self.green_h_max,
            self.green_s_min,
            self.green_s_max,
            self.green_v_min,
            self.green_v_max,
        )

        green_components = self.detect_green_components(green_mask)
        debug_img = bgr.copy()

        if len(green_components) < 1:
            self.fail('no_green')
            self.draw_debug(debug_img, None, green_components)
            self.publish_debug(debug_img, msg.header)
            self.finish_callback(t0)
            return

        best = self.find_best_feature(green_components, yellow_mask, green_mask, bgr.shape)

        if best is None:
            self.fail('no_center_ygy_end_green')
            self.draw_debug(debug_img, None, green_components)
            self.publish_debug(debug_img, msg.header)
            self.finish_callback(t0)
            return

        frame_id = self.camera_frame_override if self.camera_frame_override else msg.header.frame_id
        self.pub_end_green.publish(self.make_uv_point(best['end_green_uv'], msg.header, frame_id))
        self.pub_center_green.publish(self.make_uv_point(best['center_green_uv'], msg.header, frame_id))
        self.pub_yellow_p0.publish(self.make_uv_point(best['p0_uv'], msg.header, frame_id))
        self.pub_yellow_p1.publish(self.make_uv_point(best['p1_uv'], msg.header, frame_id))
        self.pub_count += 1

        self.draw_debug(debug_img, best, green_components)
        self.publish_debug(debug_img, msg.header)

        if self.debug_log:
            self.get_logger().info(
                f'[feature] frame={self.frame_index} score={best["score"]:.1f} '
                f'center_axis_score={best["center_axis_score"]:.1f} green_n={len(green_components)} '
                f'yL={best["yellow_left"]} yR={best["yellow_right"]} '
                f'gE={best["green_end"]} gC={best["green_center"]} '
                f'balance={best["yellow_balance"]:.2f} end_perp={best["end_perp_px"]:.1f}px '
                f'end=({best["end_green_uv"][0]:.1f},{best["end_green_uv"][1]:.1f}) '
                f'center=({best["center_green_uv"][0]:.1f},{best["center_green_uv"][1]:.1f}) '
                f'p0=({best["p0_uv"][0]:.1f},{best["p0_uv"][1]:.1f}) '
                f'p1=({best["p1_uv"][0]:.1f},{best["p1_uv"][1]:.1f})'
            )

        self.finish_callback(t0)

    def find_best_feature(
        self,
        green_components: List[dict],
        yellow_mask: np.ndarray,
        green_mask: np.ndarray,
        image_shape,
    ) -> Optional[dict]:
        comps = green_components[:max(1, self.max_green_candidates)]
        yellow_pts = self.mask_points(yellow_mask)

        if yellow_pts.shape[0] < self.min_yellow_total_pixels:
            self.fail('yellow_too_small')
            return None

        best = None
        best_score = -1e18

        for center_comp in comps:
            center_uv = center_comp['center'].astype(np.float64)
            center_axis = self.find_center_ygy_axis(center_uv, yellow_pts)
            if center_axis is None:
                continue

            axis_unoriented = center_axis['axis']
            normal = np.array([-axis_unoriented[1], axis_unoriented[0]], dtype=np.float64)

            # Only now search end green, and only near this center axis.
            for end_comp in comps:
                if end_comp['idx'] == center_comp['idx']:
                    continue

                end_uv = end_comp['center'].astype(np.float64)
                rel = end_uv - center_uv
                s = float(np.dot(rel, axis_unoriented))
                perp = abs(float(np.dot(rel, normal)))
                dist = float(np.linalg.norm(rel))

                if perp > self.end_green_max_perp_px:
                    continue
                if dist < self.end_green_min_dist_px or dist > self.end_green_max_dist_px:
                    continue
                if abs(s) < self.end_green_min_dist_px:
                    continue

                # Orient axis from end green -> center green.
                # If end = center + s*a, then center-end = -s*a.
                sign = -1.0 if s > 0.0 else 1.0
                axis_dir = sign * axis_unoriented
                axis_dir = axis_dir / max(float(np.linalg.norm(axis_dir)), 1e-9)

                measured_gap_px = float(np.linalg.norm(center_uv - end_uv))
                px_per_cm = measured_gap_px / self.green_gap_cm
                if px_per_cm < 1e-6:
                    continue

                p0_uv = end_uv - self.end_green_center_cm * px_per_cm * axis_dir
                p1_uv = end_uv + (self.handle_length_cm - self.end_green_center_cm) * px_per_cm * axis_dir

                if not self.endpoint_sanity_check(p0_uv, p1_uv, image_shape):
                    continue

                score_info = self.score_full_pattern(
                    p0_uv=p0_uv,
                    axis=axis_dir,
                    px_per_cm=px_per_cm,
                    yellow_mask=yellow_mask,
                    green_mask=green_mask,
                    end_comp=end_comp,
                    center_comp=center_comp,
                )
                if score_info is None:
                    continue

                score = center_axis['score'] + score_info['pattern_score'] - self.end_perp_penalty * perp

                candidate = {
                    'score': float(score),
                    'center_axis_score': float(center_axis['score']),
                    'center_axis_pos': int(center_axis['pos']),
                    'center_axis_neg': int(center_axis['neg']),
                    'end_perp_px': float(perp),
                    'end_s_px': float(s),
                    'end_green_uv': end_uv,
                    'center_green_uv': center_uv,
                    'p0_uv': p0_uv,
                    'p1_uv': p1_uv,
                    'axis': axis_dir,
                    'px_per_cm': px_per_cm,
                    **score_info,
                }

                if score > best_score:
                    best_score = score
                    best = candidate

        return best

    def find_center_ygy_axis(self, center_uv: np.ndarray, yellow_pts: np.ndarray) -> Optional[dict]:
        rel = yellow_pts - center_uv[None, :]
        r = np.linalg.norm(rel, axis=1)
        local = r <= self.center_axis_local_radius_px
        if not np.any(local):
            return None

        rel_local = rel[local]
        if rel_local.shape[0] < 2 * self.center_axis_min_side_yellow_px:
            return None

        best = None
        best_score = -1e18
        step = max(0.5, self.angle_search_step_deg)

        for th in np.deg2rad(np.arange(0.0, 180.0, step, dtype=np.float64)):
            axis = np.array([math.cos(th), math.sin(th)], dtype=np.float64)
            normal = np.array([-axis[1], axis[0]], dtype=np.float64)

            s = rel_local @ axis
            perp = np.abs(rel_local @ normal)
            in_corridor = perp <= self.center_axis_corridor_half_width_px
            if not np.any(in_corridor):
                continue

            s_corr = s[in_corridor]
            pos = int(np.count_nonzero(
                (s_corr >= self.center_axis_min_s_px) & (s_corr <= self.center_axis_max_s_px)
            ))
            neg = int(np.count_nonzero(
                (s_corr <= -self.center_axis_min_s_px) & (s_corr >= -self.center_axis_max_s_px)
            ))

            if pos < self.center_axis_min_side_yellow_px:
                continue
            if neg < self.center_axis_min_side_yellow_px:
                continue

            balance = min(pos, neg) / max(max(pos, neg), 1)
            if balance < self.center_axis_min_balance:
                continue

            score = 2.0 * min(pos, neg) + 0.25 * max(pos, neg) + 200.0 * balance
            if score > best_score:
                best_score = score
                best = {'axis': axis, 'score': float(score), 'pos': pos, 'neg': neg, 'balance': float(balance)}

        return best

    def score_full_pattern(
        self,
        p0_uv: np.ndarray,
        axis: np.ndarray,
        px_per_cm: float,
        yellow_mask: np.ndarray,
        green_mask: np.ndarray,
        end_comp: dict,
        center_comp: dict,
    ) -> Optional[Dict[str, float]]:
        normal = np.array([-axis[1], axis[0]], dtype=np.float64)

        yellow_counts = self.color_counts_along_pattern(yellow_mask, p0_uv, axis, normal, px_per_cm)
        green_counts = self.color_counts_along_pattern(green_mask, p0_uv, axis, normal, px_per_cm)

        y_left = yellow_counts['yellow_left']
        y_right = yellow_counts['yellow_right']
        y_total = y_left + y_right
        g_end = green_counts['green_end']
        g_center = green_counts['green_center']

        if y_total < self.min_yellow_total_pixels:
            return None
        if y_left < self.min_yellow_left_pixels:
            return None
        if y_right < self.min_yellow_right_pixels:
            return None
        if g_end < self.min_green_end_pixels:
            return None
        if g_center < self.min_green_center_pixels:
            return None

        y_balance = min(y_left, y_right) / max(max(y_left, y_right), 1)
        if y_balance < self.min_yellow_balance:
            return None

        if self.check_green_band_length:
            end_len_cm = self.component_extent_cm(end_comp, axis, px_per_cm)
            center_len_cm = self.component_extent_cm(center_comp, axis, px_per_cm)
            if end_len_cm is None or center_len_cm is None:
                return None
            if not (self.min_green_band_len_cm <= end_len_cm <= self.max_green_band_len_cm):
                return None
            if not (self.min_green_band_len_cm <= center_len_cm <= self.max_green_band_len_cm):
                return None
        else:
            end_len_cm = 0.0
            center_len_cm = 0.0

        yellow_wrong = yellow_counts['green_end'] + yellow_counts['green_center']
        green_wrong = green_counts['yellow_left'] + green_counts['yellow_right']

        pattern_score = (
            2.0 * y_total
            + 1.2 * (g_end + g_center)
            + 250.0 * y_balance
            - self.wrong_color_penalty * (yellow_wrong + green_wrong)
        )

        return {
            'pattern_score': float(pattern_score),
            'yellow_left': int(y_left),
            'yellow_right': int(y_right),
            'yellow_total': int(y_total),
            'yellow_balance': float(y_balance),
            'green_end': int(g_end),
            'green_center': int(g_center),
            'yellow_wrong': int(yellow_wrong),
            'green_wrong': int(green_wrong),
            'end_green_len_cm': float(end_len_cm),
            'center_green_len_cm': float(center_len_cm),
        }

    def component_extent_cm(self, comp: dict, axis: np.ndarray, px_per_cm: float) -> Optional[float]:
        contour = comp.get('contour', None)
        if contour is None:
            return None
        pts = contour.reshape(-1, 2).astype(np.float64)
        if pts.shape[0] < 2:
            return None
        rel = pts - comp['center'][None, :]
        s = rel @ axis
        extent_px = float(np.max(s) - np.min(s))
        return extent_px / max(px_per_cm, 1e-6)

    def color_counts_along_pattern(
        self,
        mask: np.ndarray,
        p0_uv: np.ndarray,
        axis: np.ndarray,
        normal: np.ndarray,
        px_per_cm: float,
    ) -> Dict[str, int]:
        pts = self.mask_points(mask)
        out = {'green_end': 0, 'yellow_left': 0, 'green_center': 0, 'yellow_right': 0}
        if pts.shape[0] == 0:
            return out

        rel = pts - p0_uv[None, :]
        s_px = rel @ axis
        perp = rel @ normal
        s_cm = s_px / max(px_per_cm, 1e-6)

        in_corridor = np.abs(perp) <= self.pattern_corridor_half_width_px
        if not np.any(in_corridor):
            return out

        s = s_cm[in_corridor]
        out['green_end'] = int(np.count_nonzero(
            (s >= self.end_green_start_cm) & (s <= self.end_green_end_cm)
        ))
        out['yellow_left'] = int(np.count_nonzero(
            (s > self.end_green_end_cm) & (s < self.center_green_start_cm)
        ))
        out['green_center'] = int(np.count_nonzero(
            (s >= self.center_green_start_cm) & (s <= self.center_green_end_cm)
        ))
        out['yellow_right'] = int(np.count_nonzero(
            (s > self.center_green_end_cm) & (s <= self.handle_length_cm)
        ))
        return out

    def make_mask(self, hsv, h_min, h_max, s_min, s_max, v_min, v_max):
        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        if self.open_kernel > 1:
            k = np.ones((self.open_kernel, self.open_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if self.close_kernel > 1:
            k = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    @staticmethod
    def mask_points(mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)

    def detect_green_components(self, green_mask: np.ndarray) -> List[dict]:
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        comps = []
        for idx, cnt in enumerate(contours):
            area = float(cv2.contourArea(cnt))
            if area < self.green_min_area or area > self.green_max_area:
                continue
            M = cv2.moments(cnt)
            if abs(M['m00']) < 1e-6:
                continue
            u = float(M['m10'] / M['m00'])
            v = float(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(cnt)
            comps.append({
                'idx': idx,
                'area': area,
                'center': np.array([u, v], dtype=np.float64),
                'bbox': (int(x), int(y), int(w), int(h)),
                'contour': cnt,
            })
        comps.sort(key=lambda c: c['area'], reverse=True)
        return comps

    def endpoint_sanity_check(self, p0_uv: np.ndarray, p1_uv: np.ndarray, image_shape) -> bool:
        h, w = image_shape[:2]
        m = self.endpoint_margin_px
        for p in (p0_uv, p1_uv):
            if p[0] < -m or p[0] > w + m:
                return False
            if p[1] < -m or p[1] > h + m:
                return False
        return True

    def draw_debug(self, img: np.ndarray, best: Optional[dict], green_components: List[dict]):
        for c in green_components[:self.max_green_candidates]:
            u, v = c['center']
            x, y, w, h = c['bbox']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 120, 0), 1)
            cv2.circle(img, (int(round(u)), int(round(v))), 4, (0, 160, 0), -1)

        if best is None:
            cv2.putText(img, 'NO CENTER Y-G-Y + END GREEN', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            return

        p0 = tuple(np.round(best['p0_uv']).astype(int))
        p1 = tuple(np.round(best['p1_uv']).astype(int))
        end_g = tuple(np.round(best['end_green_uv']).astype(int))
        center_g = tuple(np.round(best['center_green_uv']).astype(int))

        cv2.line(img, p0, p1, (255, 255, 0), 3)
        cv2.circle(img, p0, 6, (255, 255, 0), -1)
        cv2.circle(img, p1, 6, (255, 255, 0), -1)

        cv2.line(img, end_g, center_g, (0, 255, 0), 2)
        cv2.circle(img, end_g, 8, (0, 255, 0), -1)
        cv2.circle(img, center_g, 8, (255, 0, 0), -1)

        axis = best['axis']
        px_per_cm = best['px_per_cm']
        p0_uv = best['p0_uv']
        normal = np.array([-axis[1], axis[0]], dtype=np.float64)

        for cm, color in [
            (0.0, (0, 255, 0)),
            (3.0, (0, 255, 255)),
            (25.7, (0, 255, 0)),
            (28.7, (0, 255, 255)),
            (54.5, (255, 255, 0)),
        ]:
            p = p0_uv + axis * cm * px_per_cm
            a = p - normal * 12.0
            b = p + normal * 12.0
            cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, 2)

        txt = (
            f'score={best["score"]:.0f} '
            f'yL={best["yellow_left"]} yR={best["yellow_right"]} '
            f'gE={best["green_end"]} gC={best["green_center"]} '
            f'bal={best["yellow_balance"]:.2f}'
        )
        cv2.putText(img, txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

    def publish_debug(self, bgr: np.ndarray, header):
        if not self.publish_debug_image:
            return
        try:
            msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
            msg.header = header
            self.pub_debug.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'debug image publish failed: {repr(e)}')

    @staticmethod
    def make_uv_point(uv: np.ndarray, header, frame_id: str) -> PointStamped:
        msg = PointStamped()
        msg.header = header
        msg.header.frame_id = frame_id
        msg.point.x = float(uv[0])
        msg.point.y = float(uv[1])
        msg.point.z = 0.0
        return msg

    def fail(self, name: str):
        self.fail_counts[name] += 1

    def finish_callback(self, t0: float):
        self.last_cb_ms = (time.perf_counter() - t0) * 1000.0
        self.print_profile_if_needed()

    def print_profile_if_needed(self):
        if not self.profile:
            return
        now = time.perf_counter()
        if now - self.last_profile_print < self.profile_period_sec:
            return

        elapsed = now - self.last_profile_print
        cb_hz = self.cb_count / max(elapsed, 1e-6)
        pub_hz = self.pub_count / max(elapsed, 1e-6)
        fails = 'none' if not self.fail_counts else ', '.join([f'{k}:{v}' for k, v in self.fail_counts.items()])

        self.get_logger().info(
            f'[FEATURE PROFILE] cb_hz={cb_hz:.2f}, pub_hz={pub_hz:.2f}, '
            f'last_cb={self.last_cb_ms:.1f}ms, fails={fails}'
        )

        self.cb_count = 0
        self.pub_count = 0
        self.fail_counts.clear()
        self.last_profile_print = now


def main(args=None):
    rclpy.init(args=args)
    node = FeatureDetectCenterYGY()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()