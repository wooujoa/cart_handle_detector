#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cart_detect_ypg_temporal.py

Cart handle pose detector for the updated Y-P-Y-G-Y handle marker layout.

Expected feature_detect outputs:
  /cart_handle/end_green_px_zed       <- PURPLE marker pixel center
  /cart_handle/center_green_px_zed    <- GREEN marker pixel center
  /cart_handle/yellow_axis_p0_px_zed  <- estimated handle 0 cm endpoint pixel
  /cart_handle/yellow_axis_p1_px_zed  <- estimated handle 54.5 cm endpoint pixel

Physical default layout:
  purple: 14.5~17.5 cm -> center 16.0 cm
  green : 25.7~28.7 cm -> center 27.2 cm
  gap   : 11.2 cm

No depth and no point cloud are used. Pixels are projected to the known base_link
z-plane using camera_info + TF.

Temporal logic:
  - A single-frame center/yaw jump is rejected and previous accepted pose is held.
  - If the new jumped pose appears consistently for N frames, it is accepted.
  - Accepted poses can be smoothed with EMA to prevent sudden controller motion.
"""

import math
import time
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import Point, PointStamped, PoseStamped
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class CartDetectYPGTemporal(Node):
    def __init__(self):
        super().__init__('cart_detect_ypg_temporal')

        # Input topics
        self.declare_parameter('camera_info_topic', '/zedm/zed_node/rgb/camera_info')
        self.declare_parameter('end_green_px_topic', '/cart_handle/end_green_px_zed')
        self.declare_parameter('center_green_px_topic', '/cart_handle/center_green_px_zed')
        self.declare_parameter('yellow_axis_p0_px_topic', '/cart_handle/yellow_axis_p0_px_zed')
        self.declare_parameter('yellow_axis_p1_px_topic', '/cart_handle/yellow_axis_p1_px_zed')

        # Output topics
        self.declare_parameter('output_handle_topic', '/cart_handle/handle_pose_base')
        self.declare_parameter('output_goal_pose_topic', '/cart_handle/goal_pose_base')
        self.declare_parameter('output_marker_topic', '/cart_handle/goal_pose_markers')

        # Frames / TF
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame_fallback', 'zedm_left_camera_optical_frame')
        self.declare_parameter('use_msg_timestamp', False)
        self.declare_parameter('tf_timeout_sec', 0.05)

        # Physical constants
        self.declare_parameter('handle_length_m', 0.545)
        self.declare_parameter('end_green_center_m', 0.160)     # purple center = 16.0 cm
        self.declare_parameter('center_green_center_m', 0.272)  # green center = 27.2 cm
        self.declare_parameter('handle_z_min_m', 1.010)
        self.declare_parameter('handle_z_max_m', 1.010)
        self.declare_parameter('handle_z_step_m', 0.005)

        # Cylindrical visible surface -> centerline correction
        self.declare_parameter('use_cylinder_centerline_correction', True)
        self.declare_parameter('handle_radius_m', 0.015)

        # Parallel pixel offset search around the purple-green line
        self.declare_parameter('perp_offset_search_px', 45.0)
        self.declare_parameter('perp_offset_step_px', 2.0)

        # Candidate validity thresholds. Purple-green baseline is short, so gap tolerance
        # should be less strict than old green-green layout.
        self.declare_parameter('max_green_gap_error_m', 0.055)
        self.declare_parameter('max_handle_length_error_m', 0.160)
        self.declare_parameter('min_axis_consistency', 0.55)

        # Optional projected marker position check along p0->p1 axis.
        self.declare_parameter('use_marker_position_check', False)
        self.declare_parameter('max_marker_position_error_m', 0.10)
        self.declare_parameter('max_marker_lateral_error_m', 0.06)

        # Candidate score weights
        self.declare_parameter('green_gap_weight', 180.0)
        self.declare_parameter('handle_length_weight', 60.0)
        self.declare_parameter('axis_consistency_weight', 25.0)
        self.declare_parameter('offset_penalty_weight', 0.002)
        self.declare_parameter('z_center_penalty_weight', 4.0)
        self.declare_parameter('marker_position_weight', 30.0)
        self.declare_parameter('marker_lateral_weight', 20.0)

        # Docking goal
        self.declare_parameter('basket_side', 'left')
        self.declare_parameter('standoff_m', 0.45)
        self.declare_parameter('publish_forward_offset_m', 0.02)
        # Publish-time yaw offset. Positive value means counter-clockwise in base_link.
        # This is applied only to the published goal theta/handle theta, not to detection.
        self.declare_parameter('publish_yaw_offset_deg', 0.0)

        # Temporal gate / smoothing
        self.declare_parameter('enable_temporal_gate', True)
        self.declare_parameter('max_center_jump_m', 0.10)
        self.declare_parameter('max_goal_jump_m', 0.15)
        self.declare_parameter('max_yaw_jump_deg', 18.0)
        self.declare_parameter('pending_accept_count', 4)
        self.declare_parameter('pending_similarity_center_m', 0.04)
        self.declare_parameter('pending_similarity_goal_m', 0.06)
        self.declare_parameter('pending_similarity_yaw_deg', 8.0)
        self.declare_parameter('hold_previous_on_reject', True)
        self.declare_parameter('max_hold_sec', 1.0)
        self.declare_parameter('force_accept_after_rejects', 10)
        self.declare_parameter('enable_smoothing', True)
        self.declare_parameter('xy_alpha', 0.55)
        self.declare_parameter('yaw_alpha', 0.55)

        # Debug / profile
        self.declare_parameter('debug', True)
        self.declare_parameter('profile', True)
        self.declare_parameter('profile_period_sec', 2.0)
        self.declare_parameter('publish_markers', True)

        # Load params
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.end_green_px_topic = self.get_parameter('end_green_px_topic').value
        self.center_green_px_topic = self.get_parameter('center_green_px_topic').value
        self.yellow_axis_p0_px_topic = self.get_parameter('yellow_axis_p0_px_topic').value
        self.yellow_axis_p1_px_topic = self.get_parameter('yellow_axis_p1_px_topic').value
        self.output_handle_topic = self.get_parameter('output_handle_topic').value
        self.output_goal_pose_topic = self.get_parameter('output_goal_pose_topic').value
        self.output_marker_topic = self.get_parameter('output_marker_topic').value

        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame_fallback = self.get_parameter('camera_frame_fallback').value
        self.use_msg_timestamp = bool(self.get_parameter('use_msg_timestamp').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)

        self.handle_length_m = float(self.get_parameter('handle_length_m').value)
        self.end_green_center_m = float(self.get_parameter('end_green_center_m').value)
        self.center_green_center_m = float(self.get_parameter('center_green_center_m').value)
        self.green_gap_m = self.center_green_center_m - self.end_green_center_m
        self.handle_z_min_m = float(self.get_parameter('handle_z_min_m').value)
        self.handle_z_max_m = float(self.get_parameter('handle_z_max_m').value)
        self.handle_z_step_m = float(self.get_parameter('handle_z_step_m').value)

        self.use_cylinder_centerline_correction = bool(self.get_parameter('use_cylinder_centerline_correction').value)
        self.handle_radius_m = float(self.get_parameter('handle_radius_m').value)

        self.perp_offset_search_px = float(self.get_parameter('perp_offset_search_px').value)
        self.perp_offset_step_px = float(self.get_parameter('perp_offset_step_px').value)

        self.max_green_gap_error_m = float(self.get_parameter('max_green_gap_error_m').value)
        self.max_handle_length_error_m = float(self.get_parameter('max_handle_length_error_m').value)
        self.min_axis_consistency = float(self.get_parameter('min_axis_consistency').value)
        self.use_marker_position_check = bool(self.get_parameter('use_marker_position_check').value)
        self.max_marker_position_error_m = float(self.get_parameter('max_marker_position_error_m').value)
        self.max_marker_lateral_error_m = float(self.get_parameter('max_marker_lateral_error_m').value)

        self.green_gap_weight = float(self.get_parameter('green_gap_weight').value)
        self.handle_length_weight = float(self.get_parameter('handle_length_weight').value)
        self.axis_consistency_weight = float(self.get_parameter('axis_consistency_weight').value)
        self.offset_penalty_weight = float(self.get_parameter('offset_penalty_weight').value)
        self.z_center_penalty_weight = float(self.get_parameter('z_center_penalty_weight').value)
        self.marker_position_weight = float(self.get_parameter('marker_position_weight').value)
        self.marker_lateral_weight = float(self.get_parameter('marker_lateral_weight').value)

        self.basket_side = str(self.get_parameter('basket_side').value).lower().strip()
        self.standoff_m = float(self.get_parameter('standoff_m').value)
        self.publish_forward_offset_m = float(self.get_parameter('publish_forward_offset_m').value)
        self.publish_yaw_offset_rad = math.radians(float(self.get_parameter('publish_yaw_offset_deg').value))

        self.enable_temporal_gate = bool(self.get_parameter('enable_temporal_gate').value)
        self.max_center_jump_m = float(self.get_parameter('max_center_jump_m').value)
        self.max_goal_jump_m = float(self.get_parameter('max_goal_jump_m').value)
        self.max_yaw_jump_rad = math.radians(float(self.get_parameter('max_yaw_jump_deg').value))
        self.pending_accept_count = int(self.get_parameter('pending_accept_count').value)
        self.pending_similarity_center_m = float(self.get_parameter('pending_similarity_center_m').value)
        self.pending_similarity_goal_m = float(self.get_parameter('pending_similarity_goal_m').value)
        self.pending_similarity_yaw_rad = math.radians(float(self.get_parameter('pending_similarity_yaw_deg').value))
        self.hold_previous_on_reject = bool(self.get_parameter('hold_previous_on_reject').value)
        self.max_hold_sec = float(self.get_parameter('max_hold_sec').value)
        self.force_accept_after_rejects = int(self.get_parameter('force_accept_after_rejects').value)
        self.enable_smoothing = bool(self.get_parameter('enable_smoothing').value)
        self.xy_alpha = max(0.0, min(1.0, float(self.get_parameter('xy_alpha').value)))
        self.yaw_alpha = max(0.0, min(1.0, float(self.get_parameter('yaw_alpha').value)))

        self.debug = bool(self.get_parameter('debug').value)
        self.profile = bool(self.get_parameter('profile').value)
        self.profile_period_sec = float(self.get_parameter('profile_period_sec').value)
        self.publish_markers_enabled = bool(self.get_parameter('publish_markers').value)

        if self.green_gap_m <= 1e-6:
            raise RuntimeError('center_green_center_m must be larger than end_green_center_m')
        if self.basket_side not in ('left', 'right'):
            raise RuntimeError("basket_side must be 'left' or 'right'")
        if self.handle_z_step_m <= 0:
            raise RuntimeError('handle_z_step_m must be positive')
        if self.perp_offset_step_px <= 0:
            raise RuntimeError('perp_offset_step_px must be positive')

        # State
        self.camera_info: Optional[CameraInfo] = None
        self.fx = self.fy = self.cx = self.cy = None
        self.end_green_px_msg: Optional[PointStamped] = None
        self.center_green_px_msg: Optional[PointStamped] = None
        self.yellow_p0_px_msg: Optional[PointStamped] = None
        self.yellow_p1_px_msg: Optional[PointStamped] = None
        self.last_processed_key: Optional[Tuple[int, int]] = None

        # Temporal state
        self.last_state = None
        self.last_accept_wall_time = None
        self.pending_state = None
        self.pending_count = 0
        self.consecutive_rejects = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cb_count = 0
        self.pub_count = 0
        self.hold_pub_count = 0
        self.reject_count = 0
        self.force_accept_count = 0
        self.fail_counts = defaultdict(int)
        self.last_cb_ms = 0.0
        self.last_profile_print = time.perf_counter()

        # ROS IO
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data)
        self.sub_end = self.create_subscription(PointStamped, self.end_green_px_topic, self.end_callback, 100)
        self.sub_center = self.create_subscription(PointStamped, self.center_green_px_topic, self.center_callback, 100)
        self.sub_y0 = self.create_subscription(PointStamped, self.yellow_axis_p0_px_topic, self.y0_callback, 100)
        self.sub_y1 = self.create_subscription(PointStamped, self.yellow_axis_p1_px_topic, self.y1_callback, 100)

        self.pub_handle = self.create_publisher(PointStamped, self.output_handle_topic, 10)
        self.pub_goal = self.create_publisher(PoseStamped, self.output_goal_pose_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.output_marker_topic, 10)

        self.get_logger().info('=== cart_detect_ypg_temporal ready ===')
        self.get_logger().info(
            f'green_gap={self.green_gap_m:.3f}m, handle_length={self.handle_length_m:.3f}m, '
            f'z=[{self.handle_z_min_m:.3f},{self.handle_z_max_m:.3f}], offset=±{self.perp_offset_search_px:.1f}px, '
            f'cyl={self.use_cylinder_centerline_correction}, radius={self.handle_radius_m:.3f}m, '
            f'basket_side={self.basket_side}, publish_forward_offset={self.publish_forward_offset_m:.3f}m, '
            f'publish_yaw_offset={math.degrees(self.publish_yaw_offset_rad):.2f}deg'
        )
        self.get_logger().info(
            f'temporal_gate={self.enable_temporal_gate}, max_center_jump={self.max_center_jump_m:.2f}m, '
            f'max_goal_jump={self.max_goal_jump_m:.2f}m, max_yaw_jump={math.degrees(self.max_yaw_jump_rad):.1f}deg, '
            f'pending_accept_count={self.pending_accept_count}'
        )

    # Callbacks
    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def end_callback(self, msg: PointStamped):
        self.cb_count += 1
        self.end_green_px_msg = msg
        self.try_process()

    def center_callback(self, msg: PointStamped):
        self.cb_count += 1
        self.center_green_px_msg = msg
        self.try_process()

    def y0_callback(self, msg: PointStamped):
        self.cb_count += 1
        self.yellow_p0_px_msg = msg
        self.try_process()

    def y1_callback(self, msg: PointStamped):
        self.cb_count += 1
        self.yellow_p1_px_msg = msg
        self.try_process()

    def try_process(self):
        t0 = time.perf_counter()

        if self.camera_info is None:
            self.fail('no_camera_info')
            self.finish_callback(t0)
            return
        if self.end_green_px_msg is None or self.center_green_px_msg is None or self.yellow_p0_px_msg is None or self.yellow_p1_px_msg is None:
            self.fail('waiting_features')
            self.finish_callback(t0)
            return

        key_end = self.stamp_key(self.end_green_px_msg)
        key_center = self.stamp_key(self.center_green_px_msg)
        key_y0 = self.stamp_key(self.yellow_p0_px_msg)
        key_y1 = self.stamp_key(self.yellow_p1_px_msg)
        if not (key_end == key_center == key_y0 == key_y1):
            self.fail('waiting_stamp_match')
            self.finish_callback(t0)
            return
        if self.last_processed_key == key_end:
            self.finish_callback(t0)
            return

        source_frame = self.end_green_px_msg.header.frame_id or self.camera_frame_fallback
        tf_msg = self.lookup_base_from_camera(self.end_green_px_msg, source_frame)
        if tf_msg is None:
            self.fail('tf_lookup')
            self.finish_callback(t0)
            return
        T_base_from_cam = self.make_transform_matrix(tf_msg)
        camera_origin_base = T_base_from_cam[:3, 3].astype(np.float64)

        candidate = self.search_best_candidate(T_base_from_cam, camera_origin_base)
        if candidate is None:
            self.fail('no_valid_constraint_candidate')
            self.finish_callback(t0)
            return

        raw_state = self.build_state(candidate)
        final_state, status = self.apply_temporal_gate(raw_state)
        if final_state is None:
            self.finish_callback(t0)
            return

        self.publish_state(final_state)
        self.pub_count += 1
        if status == 'hold':
            self.hold_pub_count += 1
        self.last_processed_key = key_end

        if self.debug:
            c = raw_state['candidate']
            self.get_logger().info(
                '[CART DETECT YPG RESULT]\n'
                f'  gate_status    = {status}\n'
                f'  score          = {c["score"]:.4f}\n'
                f'  selected_z     = {c["z"]:.4f} m\n'
                f'  selected_off   = {c["offset_px"]:.2f} px\n'
                f'  green_gap      = {c["green_gap_m"]:.4f} m (err={c["green_gap_error_m"]:.4f})\n'
                f'  handle_len     = {c["handle_len_m"]:.4f} m (err={c["handle_len_error_m"]:.4f})\n'
                f'  axis_cons      = {c["axis_consistency"]:.4f}\n'
                f'  center_raw     = ({raw_state["center_xyz"][0]:.4f}, {raw_state["center_xyz"][1]:.4f}, {raw_state["center_xyz"][2]:.4f})\n'
                f'  center_pub     = ({final_state["center_xyz"][0]:.4f}, {final_state["center_xyz"][1]:.4f}, {final_state["center_xyz"][2]:.4f})\n'
                f'  goal_pub       = ({final_state["goal_x"]:.4f}, {final_state["goal_y"]:.4f})\n'
                f'  goal_theta     = {final_state["goal_theta"]:.6f} rad ({math.degrees(final_state["goal_theta"]):.2f} deg)'
            )

        self.finish_callback(t0)

    def search_best_candidate(self, T_base_from_cam: np.ndarray, camera_origin_base: np.ndarray) -> Optional[dict]:
        end_uv0 = np.array([self.end_green_px_msg.point.x, self.end_green_px_msg.point.y], dtype=np.float64)
        center_uv0 = np.array([self.center_green_px_msg.point.x, self.center_green_px_msg.point.y], dtype=np.float64)
        p0_uv0 = np.array([self.yellow_p0_px_msg.point.x, self.yellow_p0_px_msg.point.y], dtype=np.float64)
        p1_uv0 = np.array([self.yellow_p1_px_msg.point.x, self.yellow_p1_px_msg.point.y], dtype=np.float64)

        uv_axis = center_uv0 - end_uv0
        n = float(np.linalg.norm(uv_axis))
        if n < 1e-6:
            return None
        uv_axis /= n
        uv_normal = np.array([-uv_axis[1], uv_axis[0]], dtype=np.float64)

        offsets = self.make_offsets()
        z_values = self.make_z_values()
        z_mid = 0.5 * (self.handle_z_min_m + self.handle_z_max_m)
        z_span = max(0.5 * (self.handle_z_max_m - self.handle_z_min_m), 1e-6)

        best = None
        best_score = -1e18

        for z in z_values:
            for off in offsets:
                shift = off * uv_normal
                end_uv = end_uv0 + shift
                center_uv = center_uv0 + shift
                p0_uv = p0_uv0 + shift
                p1_uv = p1_uv0 + shift

                end_raw = self.project_uv_to_base_plane(end_uv[0], end_uv[1], T_base_from_cam, z)
                center_raw = self.project_uv_to_base_plane(center_uv[0], center_uv[1], T_base_from_cam, z)
                p0_raw = self.project_uv_to_base_plane(p0_uv[0], p0_uv[1], T_base_from_cam, z)
                p1_raw = self.project_uv_to_base_plane(p1_uv[0], p1_uv[1], T_base_from_cam, z)
                if end_raw is None or center_raw is None or p0_raw is None or p1_raw is None:
                    continue

                raw_vec = center_raw[:2] - end_raw[:2]
                raw_gap = float(np.linalg.norm(raw_vec))
                if raw_gap < 1e-6:
                    continue
                raw_axis = raw_vec / raw_gap

                end_base, shift_end = self.correct_visible_surface_to_centerline(end_raw, camera_origin_base, raw_axis, z)
                center_base, shift_center = self.correct_visible_surface_to_centerline(center_raw, camera_origin_base, raw_axis, z)
                p0_base, _ = self.correct_visible_surface_to_centerline(p0_raw, camera_origin_base, raw_axis, z)
                p1_base, _ = self.correct_visible_surface_to_centerline(p1_raw, camera_origin_base, raw_axis, z)

                green_vec = center_base[:2] - end_base[:2]
                green_gap = float(np.linalg.norm(green_vec))
                if green_gap < 1e-6:
                    continue
                green_axis = green_vec / green_gap
                green_gap_err = abs(green_gap - self.green_gap_m)
                if green_gap_err > self.max_green_gap_error_m:
                    continue

                p_vec = p1_base[:2] - p0_base[:2]
                handle_len = float(np.linalg.norm(p_vec))
                if handle_len < 1e-6:
                    continue
                len_err = abs(handle_len - self.handle_length_m)
                if len_err > self.max_handle_length_error_m:
                    continue

                p_axis = p_vec / handle_len
                axis_consistency = abs(float(np.dot(green_axis, p_axis)))
                if axis_consistency < self.min_axis_consistency:
                    continue

                normal = np.array([-p_axis[1], p_axis[0]], dtype=np.float64)
                s_end = float(np.dot(end_base[:2] - p0_base[:2], p_axis))
                s_center = float(np.dot(center_base[:2] - p0_base[:2], p_axis))
                lat_end = abs(float(np.dot(end_base[:2] - p0_base[:2], normal)))
                lat_center = abs(float(np.dot(center_base[:2] - p0_base[:2], normal)))
                marker_pos_err = abs(s_end - self.end_green_center_m) + abs(s_center - self.center_green_center_m)
                marker_lat_err = lat_end + lat_center
                if self.use_marker_position_check:
                    if abs(s_end - self.end_green_center_m) > self.max_marker_position_error_m:
                        continue
                    if abs(s_center - self.center_green_center_m) > self.max_marker_position_error_m:
                        continue
                    if lat_end > self.max_marker_lateral_error_m or lat_center > self.max_marker_lateral_error_m:
                        continue

                center_s = 0.5 * self.handle_length_m
                ratio_to_center = (center_s - self.end_green_center_m) / self.green_gap_m
                center_xy = end_base[:2] + ratio_to_center * (center_base[:2] - end_base[:2])
                center_xyz = np.array([center_xy[0], center_xy[1], z], dtype=np.float64)

                end0_xyz = center_xyz.copy()
                end1_xyz = center_xyz.copy()
                end0_xyz[:2] = center_xyz[:2] - 0.5 * self.handle_length_m * green_axis
                end1_xyz[:2] = center_xyz[:2] + 0.5 * self.handle_length_m * green_axis

                z_norm = abs(z - z_mid) / z_span
                score = (
                    -self.green_gap_weight * green_gap_err
                    -self.handle_length_weight * len_err
                    + self.axis_consistency_weight * axis_consistency
                    - self.offset_penalty_weight * abs(off)
                    - self.z_center_penalty_weight * z_norm
                    - self.marker_position_weight * marker_pos_err
                    - self.marker_lateral_weight * marker_lat_err
                )

                if score > best_score:
                    best_score = score
                    best = {
                        'score': float(score),
                        'z': float(z),
                        'offset_px': float(off),
                        'end_green_base_raw': end_raw,
                        'center_green_base_raw': center_raw,
                        'p0_base_raw': p0_raw,
                        'p1_base_raw': p1_raw,
                        'end_green_base': end_base,
                        'center_green_base': center_base,
                        'p0_base': p0_base,
                        'p1_base': p1_base,
                        'cyl_shift_end_m': float(shift_end),
                        'cyl_shift_center_m': float(shift_center),
                        'green_gap_m': green_gap,
                        'green_gap_error_m': green_gap_err,
                        'handle_len_m': handle_len,
                        'handle_len_error_m': len_err,
                        'axis_consistency': axis_consistency,
                        'marker_pos_error_m': marker_pos_err,
                        'marker_lat_error_m': marker_lat_err,
                        'axis_xy': green_axis,
                        'center_xyz': center_xyz,
                        'end0_xyz': end0_xyz,
                        'end1_xyz': end1_xyz,
                    }
        return best

    def build_state(self, candidate: dict) -> dict:
        center_xyz = candidate['center_xyz'].copy()
        axis = candidate['axis_xy'].copy()
        end0_xyz = candidate['end0_xyz'].copy()
        end1_xyz = candidate['end1_xyz'].copy()

        # Publish-time forward offset in base_link +x. Does not affect candidate search.
        if abs(self.publish_forward_offset_m) > 1e-12:
            delta = np.array([self.publish_forward_offset_m, 0.0], dtype=np.float64)
            center_xyz[:2] += delta
            end0_xyz[:2] += delta
            end1_xyz[:2] += delta

        left_normal = np.array([-axis[1], axis[0]], dtype=np.float64)
        basket_normal = left_normal if self.basket_side == 'left' else -left_normal
        robot_normal = -basket_normal
        goal_x = float(center_xyz[0] + self.standoff_m * robot_normal[0])
        goal_y = float(center_xyz[1] + self.standoff_m * robot_normal[1])
        goal_theta = wrap_pi(math.atan2(float(center_xyz[1] - goal_y), float(center_xyz[0] - goal_x)))

        # Publish-time yaw offset. Positive is counter-clockwise.
        # This changes only the commanded/published orientation; it does not affect
        # feature detection, physical constraint search, center x/y, or standoff position.
        if abs(self.publish_yaw_offset_rad) > 1e-12:
            goal_theta = wrap_pi(goal_theta + self.publish_yaw_offset_rad)

        c = dict(candidate)
        c['center_xyz'] = center_xyz.copy()
        c['end0_xyz'] = end0_xyz.copy()
        c['end1_xyz'] = end1_xyz.copy()

        return {
            'stamp': self.end_green_px_msg.header.stamp,
            'center_xyz': center_xyz,
            'axis_xy': axis,
            'end0_xyz': end0_xyz,
            'end1_xyz': end1_xyz,
            'goal_x': goal_x,
            'goal_y': goal_y,
            'goal_theta': goal_theta,
            'basket_normal': basket_normal,
            'candidate': c,
        }

    # Temporal gate: reject one-frame spikes, accept persistent new poses.
    def apply_temporal_gate(self, raw: dict):
        now = time.perf_counter()

        if not self.enable_temporal_gate or self.last_state is None:
            state = self.copy_state(raw)
            self.accept_state(state, now)
            return state, 'init' if self.last_state is state else 'accept'

        center_jump = self.xy_dist(raw['center_xyz'], self.last_state['center_xyz'])
        goal_jump = math.hypot(raw['goal_x'] - self.last_state['goal_x'], raw['goal_y'] - self.last_state['goal_y'])
        yaw_jump = abs(wrap_pi(raw['goal_theta'] - self.last_state['goal_theta']))

        normal = center_jump <= self.max_center_jump_m and goal_jump <= self.max_goal_jump_m and yaw_jump <= self.max_yaw_jump_rad
        if normal:
            state = self.smooth_state(raw, self.last_state)
            self.accept_state(state, now)
            self.pending_state = None
            self.pending_count = 0
            return state, 'accept'

        # Not normal: possible detection spike or real reinitialization.
        self.reject_count += 1
        self.consecutive_rejects += 1

        if self.pending_state is not None and self.states_similar(raw, self.pending_state):
            self.pending_count += 1
        else:
            self.pending_state = self.copy_state(raw)
            self.pending_count = 1

        if self.pending_count >= self.pending_accept_count:
            # Same jumped value is persistent; accept as new reality.
            state = self.copy_state(raw)
            self.accept_state(state, now)
            self.pending_state = None
            self.pending_count = 0
            return state, 'persistent_accept'

        if self.force_accept_after_rejects > 0 and self.consecutive_rejects >= self.force_accept_after_rejects:
            state = self.copy_state(raw)
            self.accept_state(state, now)
            self.pending_state = None
            self.pending_count = 0
            self.force_accept_count += 1
            return state, 'force_accept'

        if self.hold_previous_on_reject and self.last_state is not None and self.last_accept_wall_time is not None and (now - self.last_accept_wall_time) <= self.max_hold_sec:
            held = self.copy_state(self.last_state)
            held['stamp'] = raw['stamp']
            return held, 'hold'

        if self.debug:
            self.get_logger().warn(
                f'[TEMPORAL GATE] rejected without publish: center_jump={center_jump:.3f}, '
                f'goal_jump={goal_jump:.3f}, yaw_jump={math.degrees(yaw_jump):.1f}deg, '
                f'pending={self.pending_count}/{self.pending_accept_count}'
            )
        return None, 'reject_no_publish'

    def accept_state(self, state: dict, now: float):
        self.last_state = self.copy_state(state)
        self.last_accept_wall_time = now
        self.consecutive_rejects = 0

    def states_similar(self, a: dict, b: dict) -> bool:
        center_d = self.xy_dist(a['center_xyz'], b['center_xyz'])
        goal_d = math.hypot(a['goal_x'] - b['goal_x'], a['goal_y'] - b['goal_y'])
        yaw_d = abs(wrap_pi(a['goal_theta'] - b['goal_theta']))
        return center_d <= self.pending_similarity_center_m and goal_d <= self.pending_similarity_goal_m and yaw_d <= self.pending_similarity_yaw_rad

    def smooth_state(self, raw: dict, prev: dict) -> dict:
        if not self.enable_smoothing:
            return self.copy_state(raw)
        out = self.copy_state(raw)
        a = self.xy_alpha
        prev_center = prev['center_xyz']
        raw_center = raw['center_xyz']
        new_center = raw_center.copy()
        new_center[0] = a * raw_center[0] + (1.0 - a) * prev_center[0]
        new_center[1] = a * raw_center[1] + (1.0 - a) * prev_center[1]
        out['center_xyz'] = new_center

        # Translate handle endpoints by center delta to keep marker visualization aligned.
        delta = new_center[:2] - raw_center[:2]
        out['end0_xyz'] = raw['end0_xyz'].copy()
        out['end1_xyz'] = raw['end1_xyz'].copy()
        out['end0_xyz'][:2] += delta
        out['end1_xyz'][:2] += delta
        out['candidate'] = dict(raw['candidate'])
        out['candidate']['center_xyz'] = new_center.copy()
        out['candidate']['end0_xyz'] = out['end0_xyz'].copy()
        out['candidate']['end1_xyz'] = out['end1_xyz'].copy()

        out['goal_x'] = a * raw['goal_x'] + (1.0 - a) * prev['goal_x']
        out['goal_y'] = a * raw['goal_y'] + (1.0 - a) * prev['goal_y']
        ay = self.yaw_alpha
        out['goal_theta'] = wrap_pi(prev['goal_theta'] + ay * wrap_pi(raw['goal_theta'] - prev['goal_theta']))
        return out

    @staticmethod
    def xy_dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(math.hypot(float(a[0] - b[0]), float(a[1] - b[1])))

    @staticmethod
    def copy_state(s: dict) -> dict:
        return {
            'stamp': s['stamp'],
            'center_xyz': s['center_xyz'].copy(),
            'axis_xy': s['axis_xy'].copy(),
            'end0_xyz': s['end0_xyz'].copy(),
            'end1_xyz': s['end1_xyz'].copy(),
            'goal_x': float(s['goal_x']),
            'goal_y': float(s['goal_y']),
            'goal_theta': float(s['goal_theta']),
            'basket_normal': s['basket_normal'].copy(),
            'candidate': dict(s['candidate']),
        }

    def publish_state(self, state: dict):
        out = PointStamped()
        out.header.stamp = state['stamp']
        out.header.frame_id = self.base_frame
        out.point.x = float(state['center_xyz'][0])
        out.point.y = float(state['center_xyz'][1])
        out.point.z = float(state['goal_theta'])
        self.pub_handle.publish(out)

        goal = PoseStamped()
        goal.header = out.header
        goal.pose.position.x = float(state['goal_x'])
        goal.pose.position.y = float(state['goal_y'])
        goal.pose.position.z = 0.0
        qx, qy, qz, qw = self.yaw_to_quaternion(state['goal_theta'])
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        self.pub_goal.publish(goal)

        if self.publish_markers_enabled:
            self.publish_debug_markers(
                stamp=out.header.stamp,
                candidate=state['candidate'],
                goal_x=state['goal_x'],
                goal_y=state['goal_y'],
                goal_theta=state['goal_theta'],
                basket_normal=state['basket_normal'],
            )

    def correct_visible_surface_to_centerline(self, p_base: np.ndarray, camera_origin_base: np.ndarray, axis_xy: np.ndarray, target_z: float):
        if (not self.use_cylinder_centerline_correction) or self.handle_radius_m <= 1e-9:
            q = p_base.copy()
            q[2] = target_z
            return q, 0.0
        axis = axis_xy.astype(np.float64)
        na = float(np.linalg.norm(axis))
        if na < 1e-9:
            q = p_base.copy()
            q[2] = target_z
            return q, 0.0
        axis /= na
        view_xy = camera_origin_base[:2] - p_base[:2]
        view_perp = view_xy - float(np.dot(view_xy, axis)) * axis
        nv = float(np.linalg.norm(view_perp))
        if nv < 1e-9:
            q = p_base.copy()
            q[2] = target_z
            return q, 0.0
        u_to_camera = view_perp / nv
        q = p_base.copy()
        q[:2] = p_base[:2] - self.handle_radius_m * u_to_camera
        q[2] = target_z
        return q, self.handle_radius_m

    def make_offsets(self) -> List[float]:
        max_off = max(0.0, self.perp_offset_search_px)
        step = max(0.25, self.perp_offset_step_px)
        vals = list(np.arange(-max_off, max_off + 0.5 * step, step, dtype=np.float64))
        if not any(abs(float(v)) < 1e-9 for v in vals):
            vals.append(0.0)
        return sorted([float(v) for v in vals], key=lambda x: (abs(x) > 1e-9, abs(x)))

    def make_z_values(self) -> List[float]:
        zmin = min(self.handle_z_min_m, self.handle_z_max_m)
        zmax = max(self.handle_z_min_m, self.handle_z_max_m)
        step = max(0.001, self.handle_z_step_m)
        vals = list(np.arange(zmin, zmax + 0.5 * step, step, dtype=np.float64))
        if not vals:
            vals = [0.5 * (zmin + zmax)]
        return [float(v) for v in vals]

    def project_uv_to_base_plane(self, u: float, v: float, T_base_from_cam: np.ndarray, target_z: float) -> Optional[np.ndarray]:
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            return None
        x_n = (u - self.cx) / self.fx
        y_n = (v - self.cy) / self.fy
        ray_cam = np.array([x_n, y_n, 1.0], dtype=np.float64)
        nr = float(np.linalg.norm(ray_cam))
        if nr < 1e-12:
            return None
        ray_cam /= nr
        origin_base = T_base_from_cam[:3, 3]
        ray_base = T_base_from_cam[:3, :3] @ ray_cam
        dz = float(ray_base[2])
        if abs(dz) < 1e-9:
            return None
        scale = (target_z - float(origin_base[2])) / dz
        if scale <= 0.0:
            return None
        p = origin_base + scale * ray_base
        return np.array([float(p[0]), float(p[1]), float(target_z)], dtype=np.float64)

    def lookup_base_from_camera(self, msg: PointStamped, source_frame: str):
        if self.use_msg_timestamp:
            try:
                target_time = Time.from_msg(msg.header.stamp)
                return self.tf_buffer.lookup_transform(self.base_frame, source_frame, target_time, timeout=Duration(seconds=self.tf_timeout_sec))
            except Exception:
                pass
        try:
            return self.tf_buffer.lookup_transform(self.base_frame, source_frame, Time(), timeout=Duration(seconds=self.tf_timeout_sec))
        except Exception as e:
            if self.debug:
                self.get_logger().warn(f'TF lookup failed: {self.base_frame} <- {source_frame} | {e}')
            return None

    @staticmethod
    def make_transform_matrix(transform_stamped):
        q = transform_stamped.transform.rotation
        t = transform_stamped.transform.translation
        rot = CartDetectYPGTemporal.quat_to_rotmat(q.x, q.y, q.z, q.w)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    @staticmethod
    def quat_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
        n = math.sqrt(x * x + y * y + z * z + w * w)
        if n < 1e-12:
            return np.eye(3, dtype=np.float64)
        x /= n; y /= n; z /= n; w /= n
        xx = x * x; yy = y * y; zz = z * z
        xy = x * y; xz = x * z; yz = y * z
        wx = w * x; wy = w * y; wz = w * z
        return np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ], dtype=np.float64)

    def publish_debug_markers(self, stamp, candidate: dict, goal_x: float, goal_y: float, goal_theta: float, basket_normal: np.ndarray):
        markers = MarkerArray()
        delete = Marker()
        delete.header.frame_id = self.base_frame
        delete.header.stamp = stamp
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)
        markers.markers.append(self.make_sphere_marker(1, 'purple_marker_base', stamp, candidate['end_green_base'], (0.8, 0.0, 0.8), 0.035))
        markers.markers.append(self.make_sphere_marker(2, 'green_marker_base', stamp, candidate['center_green_base'], (0.0, 1.0, 0.0), 0.035))
        markers.markers.append(self.make_sphere_marker(3, 'projected_p0', stamp, candidate['p0_base'], (1.0, 1.0, 0.0), 0.020))
        markers.markers.append(self.make_sphere_marker(4, 'projected_p1', stamp, candidate['p1_base'], (0.0, 1.0, 1.0), 0.020))
        markers.markers.append(self.make_sphere_marker(5, 'handle_center', stamp, candidate['center_xyz'], (1.0, 0.5, 0.0), 0.040))
        markers.markers.append(self.make_sphere_marker(6, 'handle_end0', stamp, candidate['end0_xyz'], (1.0, 1.0, 0.0), 0.025))
        markers.markers.append(self.make_sphere_marker(7, 'handle_end1', stamp, candidate['end1_xyz'], (0.0, 1.0, 1.0), 0.025))
        markers.markers.append(self.make_arrow_marker(20, 'handle_axis', stamp, candidate['end0_xyz'], candidate['end1_xyz'], (0.0, 1.0, 1.0)))
        center = candidate['center_xyz']
        basket_end = np.array([center[0] + 0.25 * basket_normal[0], center[1] + 0.25 * basket_normal[1], center[2]], dtype=np.float64)
        markers.markers.append(self.make_arrow_marker(21, 'basket_normal', stamp, center, basket_end, (1.0, 0.0, 1.0)))
        goal_start = np.array([goal_x, goal_y, 0.05], dtype=np.float64)
        goal_end = np.array([goal_x + 0.25 * math.cos(goal_theta), goal_y + 0.25 * math.sin(goal_theta), 0.05], dtype=np.float64)
        markers.markers.append(self.make_arrow_marker(22, 'goal_heading', stamp, goal_start, goal_end, (1.0, 0.5, 0.0)))
        self.pub_markers.publish(markers)

    def make_sphere_marker(self, marker_id: int, ns: str, stamp, xyz: np.ndarray, rgb: Tuple[float, float, float], scale: float):
        mk = Marker()
        mk.header.frame_id = self.base_frame
        mk.header.stamp = stamp
        mk.ns = ns
        mk.id = marker_id
        mk.type = Marker.SPHERE
        mk.action = Marker.ADD
        mk.pose.position.x = float(xyz[0])
        mk.pose.position.y = float(xyz[1])
        mk.pose.position.z = float(xyz[2])
        mk.pose.orientation.w = 1.0
        mk.scale.x = scale
        mk.scale.y = scale
        mk.scale.z = scale
        mk.color.a = 1.0
        mk.color.r = float(rgb[0])
        mk.color.g = float(rgb[1])
        mk.color.b = float(rgb[2])
        return mk

    def make_arrow_marker(self, marker_id: int, ns: str, stamp, start_xyz: np.ndarray, end_xyz: np.ndarray, rgb: Tuple[float, float, float]):
        mk = Marker()
        mk.header.frame_id = self.base_frame
        mk.header.stamp = stamp
        mk.ns = ns
        mk.id = marker_id
        mk.type = Marker.ARROW
        mk.action = Marker.ADD
        mk.pose.orientation.w = 1.0
        mk.scale.x = 0.015
        mk.scale.y = 0.030
        mk.scale.z = 0.050
        mk.color.a = 1.0
        mk.color.r = float(rgb[0])
        mk.color.g = float(rgb[1])
        mk.color.b = float(rgb[2])
        p0 = Point(); p0.x = float(start_xyz[0]); p0.y = float(start_xyz[1]); p0.z = float(start_xyz[2])
        p1 = Point(); p1.x = float(end_xyz[0]); p1.y = float(end_xyz[1]); p1.z = float(end_xyz[2])
        mk.points = [p0, p1]
        return mk

    @staticmethod
    def stamp_key(msg) -> Tuple[int, int]:
        return (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))

    def fail(self, stage: str):
        self.fail_counts[stage] += 1

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
            f'[POSE PROFILE] cb_hz={cb_hz:.2f}, pub_hz={pub_hz:.2f}, '
            f'last_cb={self.last_cb_ms:.2f}ms, holds={self.hold_pub_count}, rejects={self.reject_count}, '
            f'force_accepts={self.force_accept_count}, fails={fails}'
        )
        self.cb_count = 0
        self.pub_count = 0
        self.hold_pub_count = 0
        self.reject_count = 0
        self.force_accept_count = 0
        self.fail_counts.clear()
        self.last_profile_print = now

    @staticmethod
    def yaw_to_quaternion(yaw: float):
        half = 0.5 * yaw
        return (0.0, 0.0, math.sin(half), math.cos(half))


def main(args=None):
    rclpy.init(args=args)
    node = CartDetectYPGTemporal()
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