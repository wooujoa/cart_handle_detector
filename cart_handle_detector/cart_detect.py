#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cart_handle_pose_green_constraint_fast.py

Principled fast pose node for cart docking.

Inputs:
  - end_green_px
  - center_green_px
  - yellow_axis_p0_px / yellow_axis_p1_px from the feature node
  - CameraInfo
  - TF

No depth or point cloud.

Main corrections:
  1) Parallel pixel offset search:
     The visible color blobs can lie on a surface line, not the true centerline.
     Search nearby parallel image lines and pick the candidate that best matches
     physical green gap and handle length.

  2) Cylindrical visible-surface -> centerline correction:
     A wrapped band on a cylindrical handle is usually detected on the camera-side
     visible surface. Projecting that pixel directly to z=handle_z gives a line
     biased toward the camera. To generalize across 0/45/90 deg views, shift the
     projected point away from the camera in the direction perpendicular to the
     handle axis by the physical handle radius.

     centerline_point ~= visible_surface_point - radius * unit(point_to_camera,
                                                             perpendicular_to_axis)

This is NOT a fixed base_link x/y offset.
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


class CartHandlePoseGreenConstraintFast(Node):
    def __init__(self):
        super().__init__('cart_handle_pose_green_constraint_fast')

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
        self.declare_parameter('end_green_center_m', 0.015)
        self.declare_parameter('center_green_center_m', 0.272)

        # Real handle center height.
        self.declare_parameter('handle_z_min_m', 1.020)
        self.declare_parameter('handle_z_max_m', 1.030)
        self.declare_parameter('handle_z_step_m', 0.005)

        # Cylindrical centerline correction
        self.declare_parameter('use_cylinder_centerline_correction', True)
        self.declare_parameter('handle_radius_m', 0.020)

        # Pixel perpendicular offset search around the green-pair line
        self.declare_parameter('perp_offset_search_px', 45.0)
        self.declare_parameter('perp_offset_step_px', 3.0)

        # Candidate validity thresholds
        self.declare_parameter('max_green_gap_error_m', 0.08)
        self.declare_parameter('max_handle_length_error_m', 0.12)

        # Candidate weights
        self.declare_parameter('green_gap_weight', 200.0)
        self.declare_parameter('handle_length_weight', 120.0)
        self.declare_parameter('axis_consistency_weight', 20.0)
        self.declare_parameter('offset_penalty_weight', 0.002)
        self.declare_parameter('z_center_penalty_weight', 4.0)

        # Docking goal
        # basket_side is relative to handle axis end_green -> center_green:
        #   left  : basket is on +left normal = [-axis_y, axis_x]
        #   right : basket is on -left normal
        self.declare_parameter('basket_side', 'right')
        self.declare_parameter('standoff_m', 0.45)

        # Final publish-time offset in base_link frame.
        # +0.02 means publish handle/goal 2cm forward along base_link +x.
        # This does NOT affect detection, constraint search, green-gap check,
        # handle-length check, or yaw estimation.
        self.declare_parameter('publish_forward_offset_m', 0.02)

        # Debug / profile
        self.declare_parameter('debug', True)
        self.declare_parameter('profile', True)
        self.declare_parameter('profile_period_sec', 2.0)
        self.declare_parameter('publish_markers', True)

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

        self.use_cylinder_centerline_correction = bool(
            self.get_parameter('use_cylinder_centerline_correction').value
        )
        self.handle_radius_m = float(self.get_parameter('handle_radius_m').value)

        self.perp_offset_search_px = float(self.get_parameter('perp_offset_search_px').value)
        self.perp_offset_step_px = float(self.get_parameter('perp_offset_step_px').value)

        self.max_green_gap_error_m = float(self.get_parameter('max_green_gap_error_m').value)
        self.max_handle_length_error_m = float(self.get_parameter('max_handle_length_error_m').value)

        self.green_gap_weight = float(self.get_parameter('green_gap_weight').value)
        self.handle_length_weight = float(self.get_parameter('handle_length_weight').value)
        self.axis_consistency_weight = float(self.get_parameter('axis_consistency_weight').value)
        self.offset_penalty_weight = float(self.get_parameter('offset_penalty_weight').value)
        self.z_center_penalty_weight = float(self.get_parameter('z_center_penalty_weight').value)

        self.basket_side = str(self.get_parameter('basket_side').value).lower().strip()
        self.standoff_m = float(self.get_parameter('standoff_m').value)
        self.publish_forward_offset_m = float(self.get_parameter('publish_forward_offset_m').value)

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
        if self.handle_radius_m < 0:
            raise RuntimeError('handle_radius_m must be non-negative')

        self.camera_info: Optional[CameraInfo] = None
        self.fx = self.fy = self.cx = self.cy = None

        self.end_green_px_msg: Optional[PointStamped] = None
        self.center_green_px_msg: Optional[PointStamped] = None
        self.yellow_p0_px_msg: Optional[PointStamped] = None
        self.yellow_p1_px_msg: Optional[PointStamped] = None
        self.last_processed_key: Optional[Tuple[int, int]] = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cb_count = 0
        self.pub_count = 0
        self.fail_counts = defaultdict(int)
        self.last_cb_ms = 0.0
        self.last_profile_print = time.perf_counter()

        self.sub_info = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data
        )
        self.sub_end = self.create_subscription(PointStamped, self.end_green_px_topic, self.end_callback, 100)
        self.sub_center = self.create_subscription(PointStamped, self.center_green_px_topic, self.center_callback, 100)
        self.sub_y0 = self.create_subscription(PointStamped, self.yellow_axis_p0_px_topic, self.y0_callback, 100)
        self.sub_y1 = self.create_subscription(PointStamped, self.yellow_axis_p1_px_topic, self.y1_callback, 100)

        self.pub_handle = self.create_publisher(PointStamped, self.output_handle_topic, 10)
        self.pub_goal = self.create_publisher(PoseStamped, self.output_goal_pose_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.output_marker_topic, 10)

        self.get_logger().info('=== cart_handle_pose_green_constraint_fast ready ===')
        self.get_logger().info(
            f'green_gap={self.green_gap_m:.3f}m, handle_length={self.handle_length_m:.3f}m, '
            f'z=[{self.handle_z_min_m:.3f},{self.handle_z_max_m:.3f}], '
            f'offset=±{self.perp_offset_search_px:.1f}px, '
            f'cylinder_correction={self.use_cylinder_centerline_correction}, '
            f'handle_radius={self.handle_radius_m:.3f}m, basket_side={self.basket_side}, '
            f'publish_forward_offset={self.publish_forward_offset_m:.3f}m'
        )

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Main
    # ------------------------------------------------------------
    def try_process(self):
        t0 = time.perf_counter()

        if self.camera_info is None:
            self.fail('no_camera_info')
            self.finish_callback(t0)
            return

        if (
            self.end_green_px_msg is None or
            self.center_green_px_msg is None or
            self.yellow_p0_px_msg is None or
            self.yellow_p1_px_msg is None
        ):
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

        candidate = self.search_best_centerline_candidate(T_base_from_cam, camera_origin_base)
        if candidate is None:
            self.fail('no_valid_constraint_candidate')
            self.finish_callback(t0)
            return

        center_xyz = candidate['center_xyz']
        axis = candidate['axis_xy']
        end0_xyz = candidate['end0_xyz']
        end1_xyz = candidate['end1_xyz']

        # ------------------------------------------------------------
        # Publish-time base_link +x offset.
        # Applied AFTER detection/projection/constraint search.
        # Therefore yaw, selected offset, green gap, and handle length checks
        # remain unchanged. Only the published handle center, goal pose, and
        # visualization markers are shifted forward.
        # ------------------------------------------------------------
        if abs(self.publish_forward_offset_m) > 1e-12:
            offset_xy = np.array([self.publish_forward_offset_m, 0.0], dtype=np.float64)

            center_xyz = center_xyz.copy()
            end0_xyz = end0_xyz.copy()
            end1_xyz = end1_xyz.copy()

            center_xyz[:2] += offset_xy
            end0_xyz[:2] += offset_xy
            end1_xyz[:2] += offset_xy

            candidate = dict(candidate)
            candidate['center_xyz'] = center_xyz
            candidate['end0_xyz'] = end0_xyz
            candidate['end1_xyz'] = end1_xyz

        left_normal = np.array([-axis[1], axis[0]], dtype=np.float64)
        if self.basket_side == 'left':
            basket_normal = left_normal
        else:
            basket_normal = -left_normal
        robot_normal = -basket_normal

        goal_x = float(center_xyz[0] + self.standoff_m * robot_normal[0])
        goal_y = float(center_xyz[1] + self.standoff_m * robot_normal[1])
        goal_theta = math.atan2(float(center_xyz[1] - goal_y), float(center_xyz[0] - goal_x))

        out = PointStamped()
        out.header = self.end_green_px_msg.header
        out.header.frame_id = self.base_frame
        out.point.x = float(center_xyz[0])
        out.point.y = float(center_xyz[1])
        out.point.z = float(goal_theta)
        self.pub_handle.publish(out)

        goal = PoseStamped()
        goal.header = out.header
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.position.z = 0.0
        qx, qy, qz, qw = self.yaw_to_quaternion(goal_theta)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        self.pub_goal.publish(goal)

        if self.publish_markers_enabled:
            self.publish_debug_markers(
                stamp=out.header.stamp,
                candidate=candidate,
                goal_x=goal_x,
                goal_y=goal_y,
                goal_theta=goal_theta,
                basket_normal=basket_normal,
            )

        self.pub_count += 1
        self.last_processed_key = key_end

        if self.debug:
            handle_yaw = math.atan2(axis[1], axis[0])
            self.get_logger().info(
                '[CONSTRAINT HANDLE RESULT]\n'
                f'  score          = {candidate["score"]:.4f}\n'
                f'  selected_z     = {candidate["z"]:.4f} m\n'
                f'  selected_off   = {candidate["offset_px"]:.2f} px\n'
                f'  pub_x_offset   = {self.publish_forward_offset_m:.4f} m\n'
                f'  cyl_shift_end  = {candidate["cyl_shift_end_m"]:.4f} m\n'
                f'  cyl_shift_ctr  = {candidate["cyl_shift_center_m"]:.4f} m\n'
                f'  green_gap      = {candidate["green_gap_m"]:.4f} m '
                f'(err={candidate["green_gap_error_m"]:.4f})\n'
                f'  handle_len     = {candidate["handle_len_m"]:.4f} m '
                f'(err={candidate["handle_len_error_m"]:.4f})\n'
                f'  axis_cons      = {candidate["axis_consistency"]:.4f}\n'
                f'  center         = ({center_xyz[0]:.4f}, {center_xyz[1]:.4f}, {center_xyz[2]:.4f})\n'
                f'  handle_yaw     = {handle_yaw:.6f} rad ({math.degrees(handle_yaw):.2f} deg)\n'
                f'  goal_pos       = ({goal_x:.4f}, {goal_y:.4f})\n'
                f'  goal_theta     = {goal_theta:.6f} rad ({math.degrees(goal_theta):.2f} deg)'
            )

        self.finish_callback(t0)

    def search_best_centerline_candidate(
        self,
        T_base_from_cam: np.ndarray,
        camera_origin_base: np.ndarray,
    ) -> Optional[dict]:
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

        best = None
        best_score = -1e18

        z_mid = 0.5 * (self.handle_z_min_m + self.handle_z_max_m)
        z_span = max(0.5 * (self.handle_z_max_m - self.handle_z_min_m), 1e-6)

        for z in z_values:
            for off in offsets:
                shift = off * uv_normal

                end_uv = end_uv0 + shift
                center_uv = center_uv0 + shift
                p0_uv = p0_uv0 + shift
                p1_uv = p1_uv0 + shift

                end_base_raw = self.project_uv_to_base_plane(end_uv[0], end_uv[1], T_base_from_cam, z)
                center_green_base_raw = self.project_uv_to_base_plane(center_uv[0], center_uv[1], T_base_from_cam, z)
                p0_base_raw = self.project_uv_to_base_plane(p0_uv[0], p0_uv[1], T_base_from_cam, z)
                p1_base_raw = self.project_uv_to_base_plane(p1_uv[0], p1_uv[1], T_base_from_cam, z)

                if end_base_raw is None or center_green_base_raw is None:
                    continue
                if p0_base_raw is None or p1_base_raw is None:
                    continue

                # Initial axis from raw green points, needed to know the cylinder axis.
                raw_green_vec = center_green_base_raw[:2] - end_base_raw[:2]
                raw_green_gap = float(np.linalg.norm(raw_green_vec))
                if raw_green_gap < 1e-6:
                    continue
                raw_axis = raw_green_vec / raw_green_gap

                # Correct visible-surface points to the cylinder centerline.
                end_base, shift_end = self.correct_visible_surface_to_centerline(
                    end_base_raw, camera_origin_base, raw_axis, z
                )
                center_green_base, shift_center = self.correct_visible_surface_to_centerline(
                    center_green_base_raw, camera_origin_base, raw_axis, z
                )
                p0_base, _ = self.correct_visible_surface_to_centerline(
                    p0_base_raw, camera_origin_base, raw_axis, z
                )
                p1_base, _ = self.correct_visible_surface_to_centerline(
                    p1_base_raw, camera_origin_base, raw_axis, z
                )

                green_vec = center_green_base[:2] - end_base[:2]
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

                center_s = 0.5 * self.handle_length_m
                ratio_to_center = (center_s - self.end_green_center_m) / self.green_gap_m
                handle_center_xy = end_base[:2] + ratio_to_center * (center_green_base[:2] - end_base[:2])

                center_xyz = np.array([handle_center_xy[0], handle_center_xy[1], z], dtype=np.float64)

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
                )

                if score > best_score:
                    best_score = score
                    best = {
                        'score': float(score),
                        'z': float(z),
                        'offset_px': float(off),
                        'end_green_base_raw': end_base_raw,
                        'center_green_base_raw': center_green_base_raw,
                        'p0_base_raw': p0_base_raw,
                        'p1_base_raw': p1_base_raw,
                        'end_green_base': end_base,
                        'center_green_base': center_green_base,
                        'p0_base': p0_base,
                        'p1_base': p1_base,
                        'cyl_shift_end_m': float(shift_end),
                        'cyl_shift_center_m': float(shift_center),
                        'green_gap_m': green_gap,
                        'green_gap_error_m': green_gap_err,
                        'handle_len_m': handle_len,
                        'handle_len_error_m': len_err,
                        'axis_consistency': axis_consistency,
                        'axis_xy': green_axis,
                        'center_xyz': center_xyz,
                        'end0_xyz': end0_xyz,
                        'end1_xyz': end1_xyz,
                    }

        return best

    def correct_visible_surface_to_centerline(
        self,
        p_base: np.ndarray,
        camera_origin_base: np.ndarray,
        axis_xy: np.ndarray,
        target_z: float,
    ) -> Tuple[np.ndarray, float]:
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

        # Direction from visible surface point toward camera in base xy.
        view_xy = camera_origin_base[:2] - p_base[:2]

        # Remove component along the handle axis, leaving cross-section direction.
        view_perp = view_xy - float(np.dot(view_xy, axis)) * axis
        nv = float(np.linalg.norm(view_perp))
        if nv < 1e-9:
            q = p_base.copy()
            q[2] = target_z
            return q, 0.0

        u_to_camera = view_perp / nv

        # Visible surface point is camera-side surface:
        # surface ~= center + radius * u_to_camera
        # therefore center ~= surface - radius * u_to_camera.
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
        if len(vals) == 0:
            vals = [0.5 * (zmin + zmax)]
        return [float(v) for v in vals]

    # ------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------
    def project_uv_to_base_plane(
        self, u: float, v: float, T_base_from_cam: np.ndarray, target_z: float
    ) -> Optional[np.ndarray]:
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

    # ------------------------------------------------------------
    # TF
    # ------------------------------------------------------------
    def lookup_base_from_camera(self, msg: PointStamped, source_frame: str):
        if self.use_msg_timestamp:
            try:
                target_time = Time.from_msg(msg.header.stamp)
                return self.tf_buffer.lookup_transform(
                    self.base_frame,
                    source_frame,
                    target_time,
                    timeout=Duration(seconds=self.tf_timeout_sec),
                )
            except Exception:
                pass

        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as e:
            if self.debug:
                self.get_logger().warn(f'TF lookup failed: {self.base_frame} <- {source_frame} | {e}')
            return None

    @staticmethod
    def make_transform_matrix(transform_stamped):
        q = transform_stamped.transform.rotation
        t = transform_stamped.transform.translation
        rot = CartHandlePoseGreenConstraintFast.quat_to_rotmat(q.x, q.y, q.z, q.w)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    @staticmethod
    def quat_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
        n = math.sqrt(x * x + y * y + z * z + w * w)
        if n < 1e-12:
            return np.eye(3, dtype=np.float64)

        x /= n
        y /= n
        z /= n
        w /= n

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ], dtype=np.float64)

    # ------------------------------------------------------------
    # Markers
    # ------------------------------------------------------------
    def publish_debug_markers(self, stamp, candidate: dict, goal_x: float, goal_y: float, goal_theta: float, basket_normal: np.ndarray):
        markers = MarkerArray()

        delete = Marker()
        delete.header.frame_id = self.base_frame
        delete.header.stamp = stamp
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)

        # Raw projected visible-surface points
        markers.markers.append(self.make_sphere_marker(101, 'raw_green_end', stamp, candidate['end_green_base_raw'], (0.2, 0.6, 0.2), 0.020))
        markers.markers.append(self.make_sphere_marker(102, 'raw_green_center', stamp, candidate['center_green_base_raw'], (0.2, 0.2, 0.8), 0.020))

        # Corrected centerline points
        markers.markers.append(self.make_sphere_marker(1, 'green_end_centerline', stamp, candidate['end_green_base'], (0.0, 1.0, 0.0), 0.035))
        markers.markers.append(self.make_sphere_marker(2, 'green_center_centerline', stamp, candidate['center_green_base'], (0.0, 0.2, 1.0), 0.035))
        markers.markers.append(self.make_sphere_marker(3, 'projected_p0_centerline', stamp, candidate['p0_base'], (1.0, 1.0, 0.0), 0.020))
        markers.markers.append(self.make_sphere_marker(4, 'projected_p1_centerline', stamp, candidate['p1_base'], (0.0, 1.0, 1.0), 0.020))
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

        p0 = Point()
        p0.x = float(start_xyz[0])
        p0.y = float(start_xyz[1])
        p0.z = float(start_xyz[2])
        p1 = Point()
        p1.x = float(end_xyz[0])
        p1.y = float(end_xyz[1])
        p1.z = float(end_xyz[2])
        mk.points = [p0, p1]
        return mk

    # ------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------
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

        fails = 'none'
        if self.fail_counts:
            fails = ', '.join([f'{k}:{v}' for k, v in self.fail_counts.items()])

        self.get_logger().info(
            f'[POSE PROFILE] cb_hz={cb_hz:.2f}, pub_hz={pub_hz:.2f}, '
            f'last_cb={self.last_cb_ms:.2f}ms, fails={fails}'
        )

        self.cb_count = 0
        self.pub_count = 0
        self.fail_counts.clear()
        self.last_profile_print = now

    @staticmethod
    def yaw_to_quaternion(yaw: float):
        half = 0.5 * yaw
        return (0.0, 0.0, math.sin(half), math.cos(half))


def main(args=None):
    rclpy.init(args=args)
    node = CartHandlePoseGreenConstraintFast()
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