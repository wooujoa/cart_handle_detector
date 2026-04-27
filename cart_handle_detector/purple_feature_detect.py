#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
feature_detect_purple_green.py

RGB-only feature node for the new cart handle marker layout.

New marker layout:
  - Direction marker: PURPLE
  - Center marker   : GREEN
  - Purple marker is about 7.5 cm away from the center green marker.
  - Center green is still near the physical handle center.

This node keeps the same output topics as before, so pose_detect/cart_detect
does NOT need to be changed.

Published topic mapping:
  /cart_handle/end_green_px_zed       <-- PURPLE direction marker pixel
  /cart_handle/center_green_px_zed    <-- GREEN center marker pixel
  /cart_handle/yellow_axis_p0_px_zed  <-- estimated handle physical 0 cm endpoint
  /cart_handle/yellow_axis_p1_px_zed  <-- estimated handle physical 54.5 cm endpoint

The pose node can keep using:
  end_green_center_m    = purple_center_m
  center_green_center_m = green_center_m

Recommended pose params if purple-green center distance is 7.5cm:
  center_green_center_m = 0.272
  end_green_center_m    = 0.197
because 0.272 - 0.075 = 0.197.

No depth, no point cloud.
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


class FeatureDetectPurpleGreen(Node):
    def __init__(self):
        super().__init__('feature_detect_purple_green')

        # ============================================================
        # Topics
        # ============================================================
        self.declare_parameter('color_topic', '/zedm/zed_node/rgb/image_rect_color')
        self.declare_parameter('debug_image_topic', '/cart_handle/debug_image_zed')

        # Keep old topic names for pose_detect compatibility.
        self.declare_parameter('end_green_px_topic', '/cart_handle/end_green_px_zed')
        self.declare_parameter('center_green_px_topic', '/cart_handle/center_green_px_zed')
        self.declare_parameter('yellow_axis_p0_px_topic', '/cart_handle/yellow_axis_p0_px_zed')
        self.declare_parameter('yellow_axis_p1_px_topic', '/cart_handle/yellow_axis_p1_px_zed')

        self.declare_parameter('camera_frame_override', '')

        # ============================================================
        # Debug / profile
        # ============================================================
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_log', True)
        self.declare_parameter('profile', True)
        self.declare_parameter('profile_period_sec', 2.0)

        # ============================================================
        # HSV thresholds
        # ============================================================
        # Yellow handle
        self.declare_parameter('yellow_h_min', 18)
        self.declare_parameter('yellow_h_max', 42)
        self.declare_parameter('yellow_s_min', 60)
        self.declare_parameter('yellow_s_max', 255)
        self.declare_parameter('yellow_v_min', 120)
        self.declare_parameter('yellow_v_max', 255)

        # Center green marker
        self.declare_parameter('green_h_min', 45)
        self.declare_parameter('green_h_max', 100)
        self.declare_parameter('green_s_min', 60)
        self.declare_parameter('green_s_max', 255)
        self.declare_parameter('green_v_min', 50)
        self.declare_parameter('green_v_max', 255)

        # Purple direction marker: user-provided values
        self.declare_parameter('purple_h_min', 122)
        self.declare_parameter('purple_h_max', 179)
        self.declare_parameter('purple_s_min', 20)
        self.declare_parameter('purple_s_max', 168)
        self.declare_parameter('purple_v_min', 87)
        self.declare_parameter('purple_v_max', 207)

        self.declare_parameter('open_kernel', 1)
        self.declare_parameter('close_kernel', 0)

        # ============================================================
        # Component filtering
        # ============================================================
        self.declare_parameter('green_min_area', 18.0)
        self.declare_parameter('green_max_area', 30000.0)
        self.declare_parameter('purple_min_area', 18.0)
        self.declare_parameter('purple_max_area', 30000.0)
        self.declare_parameter('max_green_candidates', 8)
        self.declare_parameter('max_purple_candidates', 8)

        # ============================================================
        # Physical layout [cm]
        # ============================================================
        self.declare_parameter('handle_length_cm', 54.5)

        # Center green is still the physical center marker.
        self.declare_parameter('center_green_center_cm', 27.2)

        # Purple is about 7.5cm before center green.
        # If you re-measure this, change marker_gap_cm only.
        self.declare_parameter('marker_gap_cm', 11.2)
        self.declare_parameter('purple_width_cm', 3.0)
        self.declare_parameter('green_width_cm', 3.0)

        # If purple is on the opposite side of center green, set this false.
        # Current image: purple is to the left of center green along physical handle axis.
        # Physical coordinate is:
        #   p0 side ... purple ... center green ... p1 side
        self.declare_parameter('purple_before_center', True)

        # ============================================================
        # Pair and pattern constraints
        # ============================================================
        self.declare_parameter('min_pair_dist_px', 18.0)
        self.declare_parameter('max_pair_dist_px', 260.0)

        # p0/p1 are estimated using purple-green distance.
        # Reject if endpoint is too far outside image.
        self.declare_parameter('endpoint_margin_px', 80.0)

        # Color validation along the purple->green axis.
        self.declare_parameter('pattern_corridor_half_width_px', 28.0)

        # Yellow support in three yellow intervals:
        #   [0, purple_start], [purple_end, green_start], [green_end, handle_length]
        self.declare_parameter('min_yellow_total_pixels', 80)
        self.declare_parameter('min_yellow_left_pixels', 20)
        self.declare_parameter('min_yellow_middle_pixels', 10)
        self.declare_parameter('min_yellow_right_pixels', 25)

        self.declare_parameter('min_purple_pixels', 8)
        self.declare_parameter('min_green_pixels', 8)

        # middle yellow is physically short now, so do not require high balance.
        self.declare_parameter('min_lr_yellow_balance', 0.15)

        # Component band apparent length along axis. Relaxed for far/rotated views.
        self.declare_parameter('check_band_length', True)
        self.declare_parameter('min_marker_len_cm', 0.3)
        self.declare_parameter('max_marker_len_cm', 9.0)

        self.declare_parameter('wrong_color_penalty', 1.5)
        self.declare_parameter('endpoint_outside_penalty', 1.0)

        # ============================================================
        # Load params
        # ============================================================
        self.color_topic = self.get_parameter('color_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.end_green_px_topic = self.get_parameter('end_green_px_topic').value
        self.center_green_px_topic = self.get_parameter('center_green_px_topic').value
        self.yellow_axis_p0_px_topic = self.get_parameter('yellow_axis_p0_px_topic').value
        self.yellow_axis_p1_px_topic = self.get_parameter('yellow_axis_p1_px_topic').value
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

        self.purple_h_min = int(self.get_parameter('purple_h_min').value)
        self.purple_h_max = int(self.get_parameter('purple_h_max').value)
        self.purple_s_min = int(self.get_parameter('purple_s_min').value)
        self.purple_s_max = int(self.get_parameter('purple_s_max').value)
        self.purple_v_min = int(self.get_parameter('purple_v_min').value)
        self.purple_v_max = int(self.get_parameter('purple_v_max').value)

        self.open_kernel = int(self.get_parameter('open_kernel').value)
        self.close_kernel = int(self.get_parameter('close_kernel').value)

        self.green_min_area = float(self.get_parameter('green_min_area').value)
        self.green_max_area = float(self.get_parameter('green_max_area').value)
        self.purple_min_area = float(self.get_parameter('purple_min_area').value)
        self.purple_max_area = float(self.get_parameter('purple_max_area').value)
        self.max_green_candidates = int(self.get_parameter('max_green_candidates').value)
        self.max_purple_candidates = int(self.get_parameter('max_purple_candidates').value)

        self.handle_length_cm = float(self.get_parameter('handle_length_cm').value)
        self.center_green_center_cm = float(self.get_parameter('center_green_center_cm').value)
        self.marker_gap_cm = float(self.get_parameter('marker_gap_cm').value)
        self.purple_width_cm = float(self.get_parameter('purple_width_cm').value)
        self.green_width_cm = float(self.get_parameter('green_width_cm').value)
        self.purple_before_center = bool(self.get_parameter('purple_before_center').value)

        if self.purple_before_center:
            self.purple_center_cm = self.center_green_center_cm - self.marker_gap_cm
        else:
            self.purple_center_cm = self.center_green_center_cm + self.marker_gap_cm

        self.purple_start_cm = self.purple_center_cm - 0.5 * self.purple_width_cm
        self.purple_end_cm = self.purple_center_cm + 0.5 * self.purple_width_cm
        self.green_start_cm = self.center_green_center_cm - 0.5 * self.green_width_cm
        self.green_end_cm = self.center_green_center_cm + 0.5 * self.green_width_cm

        self.min_pair_dist_px = float(self.get_parameter('min_pair_dist_px').value)
        self.max_pair_dist_px = float(self.get_parameter('max_pair_dist_px').value)
        self.endpoint_margin_px = float(self.get_parameter('endpoint_margin_px').value)
        self.pattern_corridor_half_width_px = float(self.get_parameter('pattern_corridor_half_width_px').value)

        self.min_yellow_total_pixels = int(self.get_parameter('min_yellow_total_pixels').value)
        self.min_yellow_left_pixels = int(self.get_parameter('min_yellow_left_pixels').value)
        self.min_yellow_middle_pixels = int(self.get_parameter('min_yellow_middle_pixels').value)
        self.min_yellow_right_pixels = int(self.get_parameter('min_yellow_right_pixels').value)
        self.min_purple_pixels = int(self.get_parameter('min_purple_pixels').value)
        self.min_green_pixels = int(self.get_parameter('min_green_pixels').value)
        self.min_lr_yellow_balance = float(self.get_parameter('min_lr_yellow_balance').value)

        self.check_band_length = bool(self.get_parameter('check_band_length').value)
        self.min_marker_len_cm = float(self.get_parameter('min_marker_len_cm').value)
        self.max_marker_len_cm = float(self.get_parameter('max_marker_len_cm').value)

        self.wrong_color_penalty = float(self.get_parameter('wrong_color_penalty').value)
        self.endpoint_outside_penalty = float(self.get_parameter('endpoint_outside_penalty').value)

        if not (0.0 < self.purple_center_cm < self.handle_length_cm):
            raise RuntimeError(
                f'Invalid purple_center_cm={self.purple_center_cm:.2f}. '
                f'Check center_green_center_cm and marker_gap_cm.'
            )
        if abs(self.marker_gap_cm) < 1e-6:
            raise RuntimeError('marker_gap_cm must be non-zero.')

        self.bridge = CvBridge()

        self.cb_count = 0
        self.pub_count = 0
        self.frame_index = 0
        self.last_cb_ms = 0.0
        self.fail_counts = defaultdict(int)
        self.last_profile_print = time.perf_counter()

        self.sub_color = self.create_subscription(
            Image, self.color_topic, self.color_callback, qos_profile_sensor_data
        )

        self.pub_end_green = self.create_publisher(PointStamped, self.end_green_px_topic, 100)
        self.pub_center_green = self.create_publisher(PointStamped, self.center_green_px_topic, 100)
        self.pub_yellow_p0 = self.create_publisher(PointStamped, self.yellow_axis_p0_px_topic, 100)
        self.pub_yellow_p1 = self.create_publisher(PointStamped, self.yellow_axis_p1_px_topic, 100)
        self.pub_debug = self.create_publisher(Image, self.debug_image_topic, 10)

        self.get_logger().info('=== feature_detect_purple_green ready ===')
        self.get_logger().info(
            f'purple_center={self.purple_center_cm:.2f}cm, '
            f'green_center={self.center_green_center_cm:.2f}cm, '
            f'gap={self.marker_gap_cm:.2f}cm'
        )
        self.get_logger().info(
            f'purple HSV=({self.purple_h_min},{self.purple_h_max}, '
            f'{self.purple_s_min},{self.purple_s_max}, '
            f'{self.purple_v_min},{self.purple_v_max})'
        )

    # ============================================================
    # Main callback
    # ============================================================
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
            hsv, self.yellow_h_min, self.yellow_h_max,
            self.yellow_s_min, self.yellow_s_max,
            self.yellow_v_min, self.yellow_v_max
        )
        green_mask = self.make_mask(
            hsv, self.green_h_min, self.green_h_max,
            self.green_s_min, self.green_s_max,
            self.green_v_min, self.green_v_max
        )
        purple_mask = self.make_mask(
            hsv, self.purple_h_min, self.purple_h_max,
            self.purple_s_min, self.purple_s_max,
            self.purple_v_min, self.purple_v_max
        )

        green_components = self.detect_components(
            green_mask, self.green_min_area, self.green_max_area
        )
        purple_components = self.detect_components(
            purple_mask, self.purple_min_area, self.purple_max_area
        )

        debug_img = bgr.copy()

        if len(green_components) < 1:
            self.fail('no_green')
            self.draw_debug(debug_img, None, green_components, purple_components)
            self.publish_debug(debug_img, msg.header)
            self.finish_callback(t0)
            return

        if len(purple_components) < 1:
            self.fail('no_purple')
            self.draw_debug(debug_img, None, green_components, purple_components)
            self.publish_debug(debug_img, msg.header)
            self.finish_callback(t0)
            return

        best = self.find_best_pair(
            green_components[:self.max_green_candidates],
            purple_components[:self.max_purple_candidates],
            yellow_mask,
            green_mask,
            purple_mask,
            bgr.shape,
        )

        if best is None:
            self.fail('no_valid_purple_green_pair')
            self.draw_debug(debug_img, None, green_components, purple_components)
            self.publish_debug(debug_img, msg.header)
            self.finish_callback(t0)
            return

        frame_id = self.camera_frame_override if self.camera_frame_override else msg.header.frame_id

        # Keep old topic semantics:
        # end_green topic receives the PURPLE direction marker.
        self.pub_end_green.publish(self.make_uv_point(best['purple_uv'], msg.header, frame_id))
        self.pub_center_green.publish(self.make_uv_point(best['green_uv'], msg.header, frame_id))
        self.pub_yellow_p0.publish(self.make_uv_point(best['p0_uv'], msg.header, frame_id))
        self.pub_yellow_p1.publish(self.make_uv_point(best['p1_uv'], msg.header, frame_id))
        self.pub_count += 1

        self.draw_debug(debug_img, best, green_components, purple_components)
        self.publish_debug(debug_img, msg.header)

        if self.debug_log:
            self.get_logger().info(
                f'[feature_pg] frame={self.frame_index} '
                f'score={best["score"]:.1f} '
                f'green_n={len(green_components)} purple_n={len(purple_components)} '
                f'yL={best["yellow_left"]} yM={best["yellow_middle"]} yR={best["yellow_right"]} '
                f'P={best["purple_count"]} G={best["green_count"]} '
                f'bal={best["lr_yellow_balance"]:.2f} '
                f'purple=({best["purple_uv"][0]:.1f},{best["purple_uv"][1]:.1f}) '
                f'green=({best["green_uv"][0]:.1f},{best["green_uv"][1]:.1f}) '
                f'p0=({best["p0_uv"][0]:.1f},{best["p0_uv"][1]:.1f}) '
                f'p1=({best["p1_uv"][0]:.1f},{best["p1_uv"][1]:.1f})'
            )

        self.finish_callback(t0)

    # ============================================================
    # Candidate selection
    # ============================================================
    def find_best_pair(
        self,
        green_components: List[dict],
        purple_components: List[dict],
        yellow_mask: np.ndarray,
        green_mask: np.ndarray,
        purple_mask: np.ndarray,
        image_shape,
    ) -> Optional[dict]:
        best = None
        best_score = -1e18

        for g in green_components:
            green_uv = g['center'].astype(np.float64)

            for p in purple_components:
                purple_uv = p['center'].astype(np.float64)

                dist_px = float(np.linalg.norm(green_uv - purple_uv))
                if dist_px < self.min_pair_dist_px or dist_px > self.max_pair_dist_px:
                    continue

                # Physical axis orientation: purple -> green if purple_before_center.
                # If purple is after center, axis should be green -> purple for p0->p1.
                if self.purple_before_center:
                    axis = green_uv - purple_uv
                else:
                    axis = purple_uv - green_uv

                n = float(np.linalg.norm(axis))
                if n < 1e-6:
                    continue
                axis /= n

                px_per_cm = dist_px / abs(self.marker_gap_cm)

                # p0/p1 from the purple marker physical coordinate.
                if self.purple_before_center:
                    # purple is at purple_center_cm, green is after it.
                    p0_uv = purple_uv - self.purple_center_cm * px_per_cm * axis
                    p1_uv = purple_uv + (self.handle_length_cm - self.purple_center_cm) * px_per_cm * axis
                else:
                    # purple is after center; axis still p0->p1.
                    p0_uv = purple_uv - self.purple_center_cm * px_per_cm * axis
                    p1_uv = purple_uv + (self.handle_length_cm - self.purple_center_cm) * px_per_cm * axis

                if not self.endpoint_sanity_check(p0_uv, p1_uv, image_shape):
                    continue

                score_info = self.score_candidate(
                    p0_uv=p0_uv,
                    axis=axis,
                    px_per_cm=px_per_cm,
                    yellow_mask=yellow_mask,
                    green_mask=green_mask,
                    purple_mask=purple_mask,
                    green_comp=g,
                    purple_comp=p,
                )
                if score_info is None:
                    continue

                score = score_info['score']

                candidate = {
                    'score': float(score),
                    'purple_uv': purple_uv,
                    'green_uv': green_uv,
                    'p0_uv': p0_uv,
                    'p1_uv': p1_uv,
                    'axis': axis,
                    'px_per_cm': px_per_cm,
                    'green_comp_idx': g['idx'],
                    'purple_comp_idx': p['idx'],
                    **score_info,
                }

                if score > best_score:
                    best_score = score
                    best = candidate

        return best

    def score_candidate(
        self,
        p0_uv: np.ndarray,
        axis: np.ndarray,
        px_per_cm: float,
        yellow_mask: np.ndarray,
        green_mask: np.ndarray,
        purple_mask: np.ndarray,
        green_comp: dict,
        purple_comp: dict,
    ) -> Optional[Dict[str, float]]:
        normal = np.array([-axis[1], axis[0]], dtype=np.float64)

        yellow_counts = self.count_color_intervals(
            yellow_mask, p0_uv, axis, normal, px_per_cm
        )
        green_counts = self.count_color_intervals(
            green_mask, p0_uv, axis, normal, px_per_cm
        )
        purple_counts = self.count_color_intervals(
            purple_mask, p0_uv, axis, normal, px_per_cm
        )

        yellow_left = yellow_counts['yellow_left']
        yellow_middle = yellow_counts['yellow_middle']
        yellow_right = yellow_counts['yellow_right']
        yellow_total = yellow_left + yellow_middle + yellow_right

        purple_count = purple_counts['purple']
        green_count = green_counts['green']

        if yellow_total < self.min_yellow_total_pixels:
            return None
        if yellow_left < self.min_yellow_left_pixels:
            return None
        if yellow_middle < self.min_yellow_middle_pixels:
            return None
        if yellow_right < self.min_yellow_right_pixels:
            return None
        if purple_count < self.min_purple_pixels:
            return None
        if green_count < self.min_green_pixels:
            return None

        lr_yellow_balance = min(yellow_left, yellow_right) / max(max(yellow_left, yellow_right), 1)
        if lr_yellow_balance < self.min_lr_yellow_balance:
            return None

        if self.check_band_length:
            purple_len_cm = self.component_extent_cm(purple_comp, axis, px_per_cm)
            green_len_cm = self.component_extent_cm(green_comp, axis, px_per_cm)

            if purple_len_cm is None or green_len_cm is None:
                return None

            if not (self.min_marker_len_cm <= purple_len_cm <= self.max_marker_len_cm):
                return None
            if not (self.min_marker_len_cm <= green_len_cm <= self.max_marker_len_cm):
                return None
        else:
            purple_len_cm = 0.0
            green_len_cm = 0.0

        # Wrong color penalties:
        # yellow in marker intervals and marker colors in yellow intervals.
        yellow_wrong = yellow_counts['purple'] + yellow_counts['green']
        green_wrong = green_counts['yellow_left'] + green_counts['yellow_middle'] + green_counts['yellow_right']
        purple_wrong = purple_counts['yellow_left'] + purple_counts['yellow_middle'] + purple_counts['yellow_right']

        score = (
            2.0 * yellow_total
            + 1.5 * purple_count
            + 1.5 * green_count
            + 180.0 * lr_yellow_balance
            - self.wrong_color_penalty * (yellow_wrong + green_wrong + purple_wrong)
        )

        return {
            'score': float(score),
            'yellow_left': int(yellow_left),
            'yellow_middle': int(yellow_middle),
            'yellow_right': int(yellow_right),
            'yellow_total': int(yellow_total),
            'purple_count': int(purple_count),
            'green_count': int(green_count),
            'yellow_wrong': int(yellow_wrong),
            'green_wrong': int(green_wrong),
            'purple_wrong': int(purple_wrong),
            'lr_yellow_balance': float(lr_yellow_balance),
            'purple_len_cm': float(purple_len_cm),
            'green_len_cm': float(green_len_cm),
        }

    def count_color_intervals(
        self,
        mask: np.ndarray,
        p0_uv: np.ndarray,
        axis: np.ndarray,
        normal: np.ndarray,
        px_per_cm: float,
    ) -> Dict[str, int]:
        pts = self.mask_points(mask)
        out = {
            'yellow_left': 0,
            'purple': 0,
            'yellow_middle': 0,
            'green': 0,
            'yellow_right': 0,
        }
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

        out['yellow_left'] = int(np.count_nonzero(
            (s >= 0.0) & (s < self.purple_start_cm)
        ))
        out['purple'] = int(np.count_nonzero(
            (s >= self.purple_start_cm) & (s <= self.purple_end_cm)
        ))
        out['yellow_middle'] = int(np.count_nonzero(
            (s > self.purple_end_cm) & (s < self.green_start_cm)
        ))
        out['green'] = int(np.count_nonzero(
            (s >= self.green_start_cm) & (s <= self.green_end_cm)
        ))
        out['yellow_right'] = int(np.count_nonzero(
            (s > self.green_end_cm) & (s <= self.handle_length_cm)
        ))

        return out

    def component_extent_cm(self, comp: dict, axis: np.ndarray, px_per_cm: float) -> Optional[float]:
        cnt = comp.get('contour', None)
        if cnt is None:
            return None
        pts = cnt.reshape(-1, 2).astype(np.float64)
        if pts.shape[0] < 2:
            return None
        rel = pts - comp['center'][None, :]
        s = rel @ axis
        extent_px = float(np.max(s) - np.min(s))
        return extent_px / max(px_per_cm, 1e-6)

    # ============================================================
    # Mask / contours
    # ============================================================
    def make_mask(
        self,
        hsv: np.ndarray,
        h_min: int,
        h_max: int,
        s_min: int,
        s_max: int,
        v_min: int,
        v_max: int,
    ) -> np.ndarray:
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

    def detect_components(self, mask: np.ndarray, min_area: float, max_area: float) -> List[dict]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        comps: List[dict] = []

        for idx, cnt in enumerate(contours):
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
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

    # ============================================================
    # Debug drawing
    # ============================================================
    def draw_debug(
        self,
        img: np.ndarray,
        best: Optional[dict],
        green_components: List[dict],
        purple_components: List[dict],
    ):
        # Green candidates
        for c in green_components[:self.max_green_candidates]:
            u, v = c['center']
            x, y, w, h = c['bbox']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 160, 0), 1)
            cv2.circle(img, (int(round(u)), int(round(v))), 4, (0, 220, 0), -1)

        # Purple candidates
        for c in purple_components[:self.max_purple_candidates]:
            u, v = c['center']
            x, y, w, h = c['bbox']
            cv2.rectangle(img, (x, y), (x + w, y + h), (180, 0, 180), 1)
            cv2.circle(img, (int(round(u)), int(round(v))), 4, (220, 0, 220), -1)

        if best is None:
            cv2.putText(
                img,
                'NO VALID PURPLE-GREEN HANDLE FEATURE',
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 255),
                2,
            )
            return

        p0 = tuple(np.round(best['p0_uv']).astype(int))
        p1 = tuple(np.round(best['p1_uv']).astype(int))
        purple = tuple(np.round(best['purple_uv']).astype(int))
        green = tuple(np.round(best['green_uv']).astype(int))

        # Full handle axis
        cv2.line(img, p0, p1, (255, 255, 0), 3)
        cv2.circle(img, p0, 6, (255, 255, 0), -1)
        cv2.circle(img, p1, 6, (255, 255, 0), -1)

        # Purple -> Green direction marker
        cv2.line(img, purple, green, (0, 255, 0), 2)
        cv2.circle(img, purple, 8, (220, 0, 220), -1)
        cv2.circle(img, green, 8, (0, 255, 0), -1)

        axis = best['axis']
        px_per_cm = best['px_per_cm']
        p0_uv = best['p0_uv']
        normal = np.array([-axis[1], axis[0]], dtype=np.float64)

        # Pattern ticks
        ticks = [
            (0.0, (255, 255, 0)),
            (self.purple_start_cm, (220, 0, 220)),
            (self.purple_end_cm, (220, 0, 220)),
            (self.green_start_cm, (0, 255, 0)),
            (self.green_end_cm, (0, 255, 0)),
            (self.handle_length_cm, (255, 255, 0)),
        ]
        for cm, color in ticks:
            p = p0_uv + axis * cm * px_per_cm
            a = p - normal * 12.0
            b = p + normal * 12.0
            cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, 2)

        txt = (
            f'score={best["score"]:.0f} '
            f'yL={best["yellow_left"]} yM={best["yellow_middle"]} yR={best["yellow_right"]} '
            f'P={best["purple_count"]} G={best["green_count"]}'
        )
        cv2.putText(img, txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    def publish_debug(self, bgr: np.ndarray, header):
        if not self.publish_debug_image:
            return
        try:
            msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
            msg.header = header
            self.pub_debug.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'debug image publish failed: {repr(e)}')

    # ============================================================
    # ROS helpers
    # ============================================================
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

        fails = 'none'
        if self.fail_counts:
            fails = ', '.join([f'{k}:{v}' for k, v in self.fail_counts.items()])

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
    node = FeatureDetectPurpleGreen()
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