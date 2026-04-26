#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Point, PointStamped, PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


class YGYCartHandleDetector(Node):
    def __init__(self):
        super().__init__('ygy_cart_handle_detector')

        # --------------------------------------------------
        # Topics
        # --------------------------------------------------
        self.declare_parameter('color_topic', '/zedm/zed_node/rgb/image_rect_color')
        self.declare_parameter('depth_topic', '/zedm/zed_node/depth/depth_registered')
        self.declare_parameter('camera_info_topic', '/zedm/zed_node/rgb/camera_info')

        self.declare_parameter('output_handle_topic', '/cart_handle/handle_pose_base')
        self.declare_parameter('output_goal_pose_topic', '/cart_handle/goal_pose_base')
        self.declare_parameter('output_marker_topic', '/cart_handle/ygy_handle_markers')

        # --------------------------------------------------
        # Frames
        # --------------------------------------------------
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame_fallback', 'zedm_left_camera_optical_frame')
        self.declare_parameter('use_msg_timestamp', False)
        self.declare_parameter('tf_timeout_sec', 0.2)

        # --------------------------------------------------
        # Yellow HSV
        # --------------------------------------------------
        self.declare_parameter('yellow_h_min', 20)
        self.declare_parameter('yellow_h_max', 40)
        self.declare_parameter('yellow_s_min', 60)
        self.declare_parameter('yellow_s_max', 223)
        self.declare_parameter('yellow_v_min', 221)
        self.declare_parameter('yellow_v_max', 255)

        # --------------------------------------------------
        # Green HSV
        # --------------------------------------------------
        self.declare_parameter('green_h_min', 70)
        self.declare_parameter('green_h_max', 122)
        self.declare_parameter('green_s_min', 73)
        self.declare_parameter('green_s_max', 255)
        self.declare_parameter('green_v_min', 90)
        self.declare_parameter('green_v_max', 233)

        self.declare_parameter('open_kernel', 1)
        self.declare_parameter('close_kernel', 1)

        # --------------------------------------------------
        # Depth
        # --------------------------------------------------
        self.declare_parameter('depth_scale', 1.0)   # 32FC1=1.0, 16UC1=0.001
        self.declare_parameter('min_depth_m', 0.10)
        self.declare_parameter('max_depth_m', 3.00)
        self.declare_parameter('depth_local_search_radius', 2)

        # --------------------------------------------------
        # Geometry priors
        # --------------------------------------------------
        self.declare_parameter('handle_z_nominal_m', 1.02)
        self.declare_parameter('handle_length_m', 0.61)
        self.declare_parameter('axis_sample_count', 41)
        self.declare_parameter('standoff_m', 0.45)

        # --------------------------------------------------
        # Yellow global-axis fit
        # --------------------------------------------------
        self.declare_parameter('min_yellow_pixels', 80)
        self.declare_parameter('yellow_axis_min_linearity', 2.0)
        self.declare_parameter('yellow_axis_max_mean_perp_px', 35.0)
        self.declare_parameter('yellow_axis_half_width_px', 34.0)

        # --------------------------------------------------
        # Constraint-driven candidate search
        # --------------------------------------------------
        # Instead of accepting the first global yellow-axis corridor, search
        # nearby parallel corridors and select the one that best matches the
        # physical handle priors: z ~= handle_z_nominal_m and length ~= handle_length_m.
        self.declare_parameter('use_constraint_candidate_search', True)
        self.declare_parameter('constraint_perp_search_px', 90.0)
        self.declare_parameter('constraint_perp_step_px', 3.0)
        self.declare_parameter('constraint_max_length_error_m', 0.14)
        self.declare_parameter('constraint_length_weight', 120.0)
        self.declare_parameter('constraint_yellow_count_weight', 0.004)
        self.declare_parameter('constraint_green_count_weight', 0.010)
        self.declare_parameter('constraint_offset_weight', 0.012)

        # --------------------------------------------------
        # Green support / Y-G-Y split
        # --------------------------------------------------
        self.declare_parameter('green_corridor_half_width_px', 40.0)
        self.declare_parameter('green_margin_px', 18.0)
        self.declare_parameter('pair_min_green_pixels', 12)
        self.declare_parameter('yellow_split_gap_px', 6.0)
        self.declare_parameter('min_side_yellow_pixels', 80)
        self.declare_parameter('green_mid_ratio_max', 0.55)

        # --------------------------------------------------
        # Handle mask / local ROI point cloud
        # --------------------------------------------------
        self.declare_parameter('handle_mask_line_thickness_px', 28)
        self.declare_parameter('handle_mask_dilate_px', 11)
        self.declare_parameter('pc_max_points', 2500)

        # --------------------------------------------------
        # Basket-side probing
        # --------------------------------------------------
        self.declare_parameter('probe_px_min', 8)
        self.declare_parameter('probe_px_max', 90)
        self.declare_parameter('probe_px_step', 4)
        self.declare_parameter('side_z_window_m', 0.18)

        # --------------------------------------------------
        # Temporal smoothing / hysteresis
        # --------------------------------------------------
        self.declare_parameter('ema_alpha_center', 1.0)
        self.declare_parameter('ema_alpha_yaw', 1.0)
        self.declare_parameter('basket_score_hysteresis', 0.02)

        self.declare_parameter('sync_slop', 0.08)
        self.declare_parameter('debug', True)
        self.declare_parameter('verbose_failure_logs', True)

        # --------------------------------------------------
        # Params load
        # --------------------------------------------------
        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value

        self.output_handle_topic = self.get_parameter('output_handle_topic').value
        self.output_goal_pose_topic = self.get_parameter('output_goal_pose_topic').value
        self.output_marker_topic = self.get_parameter('output_marker_topic').value

        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame_fallback = self.get_parameter('camera_frame_fallback').value
        self.use_msg_timestamp = bool(self.get_parameter('use_msg_timestamp').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)

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

        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.depth_local_search_radius = int(self.get_parameter('depth_local_search_radius').value)

        self.handle_z_nominal_m = float(self.get_parameter('handle_z_nominal_m').value)
        self.handle_length_m = float(self.get_parameter('handle_length_m').value)
        self.axis_sample_count = int(self.get_parameter('axis_sample_count').value)
        self.standoff_m = float(self.get_parameter('standoff_m').value)

        self.min_yellow_pixels = int(self.get_parameter('min_yellow_pixels').value)
        self.yellow_axis_min_linearity = float(self.get_parameter('yellow_axis_min_linearity').value)
        self.yellow_axis_max_mean_perp_px = float(self.get_parameter('yellow_axis_max_mean_perp_px').value)
        self.yellow_axis_half_width_px = float(self.get_parameter('yellow_axis_half_width_px').value)

        self.use_constraint_candidate_search = bool(self.get_parameter('use_constraint_candidate_search').value)
        self.constraint_perp_search_px = float(self.get_parameter('constraint_perp_search_px').value)
        self.constraint_perp_step_px = float(self.get_parameter('constraint_perp_step_px').value)
        self.constraint_max_length_error_m = float(self.get_parameter('constraint_max_length_error_m').value)
        self.constraint_length_weight = float(self.get_parameter('constraint_length_weight').value)
        self.constraint_yellow_count_weight = float(self.get_parameter('constraint_yellow_count_weight').value)
        self.constraint_green_count_weight = float(self.get_parameter('constraint_green_count_weight').value)
        self.constraint_offset_weight = float(self.get_parameter('constraint_offset_weight').value)

        self.green_corridor_half_width_px = float(self.get_parameter('green_corridor_half_width_px').value)
        self.green_margin_px = float(self.get_parameter('green_margin_px').value)
        self.pair_min_green_pixels = int(self.get_parameter('pair_min_green_pixels').value)
        self.yellow_split_gap_px = float(self.get_parameter('yellow_split_gap_px').value)
        self.min_side_yellow_pixels = int(self.get_parameter('min_side_yellow_pixels').value)
        self.green_mid_ratio_max = float(self.get_parameter('green_mid_ratio_max').value)

        self.handle_mask_line_thickness_px = int(self.get_parameter('handle_mask_line_thickness_px').value)
        self.handle_mask_dilate_px = int(self.get_parameter('handle_mask_dilate_px').value)
        self.pc_max_points = int(self.get_parameter('pc_max_points').value)

        self.probe_px_min = int(self.get_parameter('probe_px_min').value)
        self.probe_px_max = int(self.get_parameter('probe_px_max').value)
        self.probe_px_step = int(self.get_parameter('probe_px_step').value)
        self.side_z_window_m = float(self.get_parameter('side_z_window_m').value)

        self.ema_alpha_center = float(self.get_parameter('ema_alpha_center').value)
        self.ema_alpha_yaw = float(self.get_parameter('ema_alpha_yaw').value)
        self.basket_score_hysteresis = float(self.get_parameter('basket_score_hysteresis').value)

        self.sync_slop = float(self.get_parameter('sync_slop').value)
        self.debug = bool(self.get_parameter('debug').value)
        self.verbose_failure_logs = bool(self.get_parameter('verbose_failure_logs').value)

        # --------------------------------------------------
        # State
        # --------------------------------------------------
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.filtered_center: Optional[np.ndarray] = None
        self.filtered_handle_yaw: Optional[float] = None
        self.prev_basket_sign: Optional[float] = None

        # --------------------------------------------------
        # Sync subs
        # --------------------------------------------------
        self.sub_color = Subscriber(self, Image, self.color_topic)
        self.sub_depth = Subscriber(self, Image, self.depth_topic)
        self.sub_info = Subscriber(self, CameraInfo, self.camera_info_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth, self.sub_info],
            queue_size=10,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self.synced_callback)

        # --------------------------------------------------
        # Publishers
        # --------------------------------------------------
        self.pub_handle = self.create_publisher(PointStamped, self.output_handle_topic, 10)
        self.pub_goal = self.create_publisher(PoseStamped, self.output_goal_pose_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.output_marker_topic, 10)

        self.get_logger().info('=== Y-G-Y Cart Handle Detector (global yellow axis version) ===')
        self.get_logger().info(f'color_topic         : {self.color_topic}')
        self.get_logger().info(f'depth_topic         : {self.depth_topic}')
        self.get_logger().info(f'camera_info_topic   : {self.camera_info_topic}')
        self.get_logger().info(f'base_frame          : {self.base_frame}')
        self.get_logger().info(f'handle_z_nominal    : {self.handle_z_nominal_m:.3f} m')
        self.get_logger().info(f'handle_length_prior : {self.handle_length_m:.3f} m')
        self.get_logger().info(f'constraint_search   : {self.use_constraint_candidate_search}')

    # =========================================================
    # Main
    # =========================================================
    def synced_callback(self, color_msg: Image, depth_msg: Image, cam_info: CameraInfo):
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'FAIL[color_decode]: {e}')
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'FAIL[depth_decode]: {e}')
            return

        if color.shape[:2] != depth.shape[:2]:
            self.get_logger().warn(f'FAIL[shape_mismatch]: color={color.shape}, depth={depth.shape}')
            self.publish_delete_markers()
            return

        yellow_mask = self.make_color_mask(
            color,
            self.yellow_h_min, self.yellow_h_max,
            self.yellow_s_min, self.yellow_s_max,
            self.yellow_v_min, self.yellow_v_max,
        )
        green_mask = self.make_color_mask(
            color,
            self.green_h_min, self.green_h_max,
            self.green_s_min, self.green_s_max,
            self.green_v_min, self.green_v_max,
        )

        yellow_pixels = int(np.count_nonzero(yellow_mask))
        green_pixels = int(np.count_nonzero(green_mask))

        if yellow_pixels < self.min_yellow_pixels:
            self.fail_log('yellow_mask_small', f'yellow_pixels={yellow_pixels} < {self.min_yellow_pixels}')
            self.publish_delete_markers()
            return

        tf_msg = self.lookup_base_from_camera(color_msg)
        if tf_msg is None:
            self.fail_log('tf_lookup_failed', f'base={self.base_frame}, source={color_msg.header.frame_id or self.camera_frame_fallback}')
            self.publish_delete_markers()
            return
        T_base_from_cam = self.make_transform_matrix(tf_msg)

        detected, fail_reasons = self.detect_handle_from_masks(
            yellow_mask=yellow_mask,
            green_mask=green_mask,
            cam_info=cam_info,
            T_base_from_cam=T_base_from_cam,
        )

        if detected is None:
            self.fail_log('no_valid_ygy_candidate', f'yellow_pixels={yellow_pixels}, green_pixels={green_pixels}')
            if self.verbose_failure_logs:
                for msg in fail_reasons[:20]:
                    self.get_logger().info(msg)
            self.publish_delete_markers()
            return

        handle_mask_2d, handle_mask_pc = self.build_handle_masks(
            yellow_mask=yellow_mask,
            green_pts_uv=detected['green_pts_uv'],
            uv_center=detected['uv_center'],
            uv_dir=detected['uv_dir'],
            s_min=detected['s_min'],
            s_max=detected['s_max'],
            shape_hw=yellow_mask.shape,
        )

        local_pts_base = self.collect_mask_points_base(
            mask=handle_mask_pc,
            depth=depth,
            cam_info=cam_info,
            T_base_from_cam=T_base_from_cam,
            max_points=self.pc_max_points,
        )

        center_xyz = detected['center_xyz']
        dir_xy = detected['dir_xy']
        end0_xyz = detected['end0_xyz']
        end1_xyz = detected['end1_xyz']
        axis_points_base = detected['axis_points_base']
        uv_p0 = detected['uv_p0']
        uv_p1 = detected['uv_p1']
        uv_dir = detected['uv_dir']

        handle_yaw_raw = math.atan2(dir_xy[1], dir_xy[0])
        normal_xy = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float64)

        pos_side_pts, neg_side_pts = self.collect_side_point_clouds(
            uv_p0=uv_p0,
            uv_p1=uv_p1,
            uv_dir=uv_dir,
            depth=depth,
            cam_info=cam_info,
            T_base_from_cam=T_base_from_cam,
            center_xyz=center_xyz,
            dir_xy=dir_xy,
        )

        pos_score = self.side_score(pos_side_pts, center_xyz, normal_xy)
        neg_score = self.side_score(neg_side_pts, center_xyz, -normal_xy)

        score_margin = abs(pos_score - neg_score)
        if pos_score <= 1e-9 and neg_score <= 1e-9:
            self.fail_log('no_side_cloud', 'both pos_score and neg_score are ~0, fallback previous basket_sign')
            basket_sign = self.prev_basket_sign if self.prev_basket_sign is not None else +1.0
        else:
            if score_margin < self.basket_score_hysteresis and self.prev_basket_sign is not None:
                basket_sign = self.prev_basket_sign
            else:
                basket_sign = +1.0 if pos_score >= neg_score else -1.0

        self.prev_basket_sign = basket_sign

        center_xyz, handle_yaw = self.apply_temporal_smoothing(center_xyz, handle_yaw_raw)

        if basket_sign > 0.0:
            basket_side_yaw = self.wrap_to_pi(handle_yaw + math.pi / 2.0)
            robot_side_yaw = self.wrap_to_pi(handle_yaw - math.pi / 2.0)
        else:
            basket_side_yaw = self.wrap_to_pi(handle_yaw - math.pi / 2.0)
            robot_side_yaw = self.wrap_to_pi(handle_yaw + math.pi / 2.0)

        goal_theta = basket_side_yaw
        goal_x = center_xyz[0] + self.standoff_m * math.cos(robot_side_yaw)
        goal_y = center_xyz[1] + self.standoff_m * math.sin(robot_side_yaw)

        out = PointStamped()
        out.header.stamp = color_msg.header.stamp
        out.header.frame_id = self.base_frame
        out.point.x = float(center_xyz[0])
        out.point.y = float(center_xyz[1])
        out.point.z = float(goal_theta)
        self.pub_handle.publish(out)

        goal_pose = PoseStamped()
        goal_pose.header = out.header
        goal_pose.pose.position.x = float(goal_x)
        goal_pose.pose.position.y = float(goal_y)
        goal_pose.pose.position.z = 0.0
        qx, qy, qz, qw = self.yaw_to_quaternion(goal_theta)
        goal_pose.pose.orientation.x = qx
        goal_pose.pose.orientation.y = qy
        goal_pose.pose.orientation.z = qz
        goal_pose.pose.orientation.w = qw
        self.pub_goal.publish(goal_pose)

        self.publish_markers(
            stamp=out.header.stamp,
            axis_points_base=axis_points_base,
            pos_side_pts=pos_side_pts,
            neg_side_pts=neg_side_pts,
            local_pts_base=local_pts_base,
            center_xyz=center_xyz,
            end0_xyz=end0_xyz,
            end1_xyz=end1_xyz,
            goal_x=goal_x,
            goal_y=goal_y,
            goal_theta=goal_theta,
            left_anchor=detected['left_anchor_xyz'],
            right_anchor=detected['right_anchor_xyz'],
            green_anchor=detected['green_anchor_xyz'],
        )

        if self.debug:
            self.get_logger().info(
                '[YGY HANDLE RESULT]\n'
                f'  yellow_pixels      = {yellow_pixels}\n'
                f'  green_pixels       = {green_pixels}\n'
                f'  score_2d           = {detected["score_2d"]:.4f}\n'
                f'  projected_length_m = {detected["projected_length_m"]:.4f}\n'
                f'  length_error_m     = {detected["length_error_m"]:.4f}\n'
                f'  forced_length_m    = {detected["forced_length_m"]:.4f}\n'
                f'  constraint_score   = {detected["constraint_score"]:.4f}\n'
                f'  constraint_perp_px = {detected["constraint_perp_offset_px"]:.2f}\n'
                f'  center_s           = {detected["center_s"]:.2f}\n'
                f'  green_s            = {detected["green_s"]:.2f}\n'
                f'  flipped_uv_axis    = {detected["flipped_uv_axis"]}\n'
                f'  center_uv_shift_m  = {detected["center_uv_shift_m"]:.4f}\n'
                f'  local_pc_pts       = {len(local_pts_base)}\n'
                f'  axis_samples       = {len(axis_points_base)}\n'
                f'  pos_pts            = {len(pos_side_pts)}\n'
                f'  neg_pts            = {len(neg_side_pts)}\n'
                f'  pos_score          = {pos_score:.4f}\n'
                f'  neg_score          = {neg_score:.4f}\n'
                f'  basket_sign        = {basket_sign:+.0f}\n'
                f'  handle_ctr         = ({center_xyz[0]:.4f}, {center_xyz[1]:.4f}, {center_xyz[2]:.4f})\n'
                f'  handle_yaw         = {handle_yaw:.6f} rad ({math.degrees(handle_yaw):.2f} deg)\n'
                f'  goal_theta         = {goal_theta:.6f} rad ({math.degrees(goal_theta):.2f} deg)\n'
                f'  goal_pos           = ({goal_x:.4f}, {goal_y:.4f})'
            )

    # =========================================================
    # Detection from masks
    # =========================================================
    def detect_handle_from_masks(
        self,
        yellow_mask: np.ndarray,
        green_mask: np.ndarray,
        cam_info: CameraInfo,
        T_base_from_cam: np.ndarray,
    ) -> Tuple[Optional[dict], List[str]]:
        fail_reasons: List[str] = []

        # 1) Find a rough global yellow axis. This only gives the approximate
        #    orientation in image coordinates. The final center/axis corridor is
        #    selected below by physical constraints.
        axis_fit = self.fit_axis_2d_from_mask(yellow_mask)
        if axis_fit is None:
            fail_reasons.append('[yellow-axis] reject: fit_axis_2d_from_mask failed')
            return None, fail_reasons

        uv_center, uv_dir, _, _, linearity, mean_perp = axis_fit
        if linearity < self.yellow_axis_min_linearity:
            fail_reasons.append(f'[yellow-axis] reject: linearity={linearity:.3f}')
            return None, fail_reasons
        if mean_perp > self.yellow_axis_max_mean_perp_px:
            fail_reasons.append(f'[yellow-axis] reject: mean_perp={mean_perp:.3f}')
            return None, fail_reasons

        ys, xs = np.where(yellow_mask > 0)
        pts_y = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)

        normal_uv = np.array([-uv_dir[1], uv_dir[0]], dtype=np.float64)
        n_norm = np.linalg.norm(normal_uv)
        if n_norm < 1e-9:
            fail_reasons.append('[yellow-axis] reject: invalid normal_uv')
            return None, fail_reasons
        normal_uv /= n_norm

        rel_y = pts_y - uv_center[None, :]
        s_y = rel_y @ uv_dir
        perp_y = rel_y @ normal_uv

        # 2) Search parallel corridors near the rough yellow axis and choose the
        #    corridor whose z=handle_z_nominal projection best matches the known
        #    physical handle length. This is the "find a similar object under my
        #    constraints, then force it to my constraints" step.
        if self.use_constraint_candidate_search:
            candidate = self.find_constraint_matched_ygy_candidate(
                yellow_mask=yellow_mask,
                green_mask=green_mask,
                pts_y=pts_y,
                s_y=s_y,
                perp_y=perp_y,
                uv_center=uv_center,
                uv_dir=uv_dir,
                normal_uv=normal_uv,
                cam_info=cam_info,
                T_base_from_cam=T_base_from_cam,
            )
        else:
            candidate = self.find_constraint_matched_ygy_candidate(
                yellow_mask=yellow_mask,
                green_mask=green_mask,
                pts_y=pts_y,
                s_y=s_y,
                perp_y=perp_y,
                uv_center=uv_center,
                uv_dir=uv_dir,
                normal_uv=normal_uv,
                cam_info=cam_info,
                T_base_from_cam=T_base_from_cam,
                forced_offsets=[0.0],
            )

        if candidate is None:
            fail_reasons.append('[constraint-search] reject: no valid Y-G-Y candidate matched constraints')
            return None, fail_reasons

        # Unpack selected candidate. From this point on, all operations use the
        # selected corridor, not necessarily the mean global yellow axis.
        uv_center_sel = candidate['uv_center_sel']
        uv_dir = candidate['uv_dir']
        normal_uv = candidate['normal_uv']
        s_min = candidate['s_min']
        s_max = candidate['s_max']
        center_s = candidate['center_s']
        green_s = candidate['green_s']
        green_pts_uv = candidate['green_pts_uv']
        left_anchor_uv = candidate['left_anchor_uv']
        right_anchor_uv = candidate['right_anchor_uv']
        green_anchor_uv = candidate['green_anchor_uv']
        uv_p0 = candidate['uv_p0']
        uv_p1 = candidate['uv_p1']
        axis_points_base = candidate['axis_points_base']
        raw_center_xyz = candidate['raw_center_xyz']
        dir_xy = candidate['dir_xy']
        raw_end0_xyz = candidate['raw_end0_xyz']
        raw_end1_xyz = candidate['raw_end1_xyz']
        raw_len = candidate['projected_length_m']
        raw_len_err = candidate['length_error_m']
        green_mid_ratio = candidate['green_mid_ratio']
        perp_offset_px = candidate['perp_offset_px']
        constraint_score = candidate['constraint_score']

        # 3) Normalize UV direction so that left/right anchors agree with uv_dir.
        #    If direction is flipped, scalar coordinates must be flipped too.
        flipped_uv_axis = False
        if float(np.dot(right_anchor_uv - left_anchor_uv, uv_dir)) < 0.0:
            old_s_min = s_min
            old_s_max = s_max
            uv_dir = -uv_dir
            s_min = -old_s_max
            s_max = -old_s_min
            center_s = -center_s
            green_s = -green_s
            uv_p0 = uv_center_sel + s_min * uv_dir
            uv_p1 = uv_center_sel + s_max * uv_dir
            flipped_uv_axis = True

            # Re-project after flipping for consistent debug/markers.
            axis_points_base = self.project_axis_to_base_plane(
                uv_p0, uv_p1,
                cam_info, T_base_from_cam,
                self.handle_z_nominal_m,
                self.axis_sample_count,
            )
            if len(axis_points_base) < 5:
                fail_reasons.append('[proj-axis] reject: projected axis points too few after flip')
                return None, fail_reasons
            raw_center_xyz, dir_xy, raw_end0_xyz, raw_end1_xyz = self.fit_xy_line_from_points(axis_points_base)
            if dir_xy is None:
                fail_reasons.append('[proj-axis] reject: fit_xy_line_from_points failed after flip')
                return None, fail_reasons
            raw_len = float(np.linalg.norm(raw_end1_xyz[:2] - raw_end0_xyz[:2]))
            raw_len_err = abs(raw_len - self.handle_length_m)

        left_anchor_xyz = self.project_uv_to_base_plane(
            float(left_anchor_uv[0]), float(left_anchor_uv[1]),
            cam_info, T_base_from_cam,
            self.handle_z_nominal_m,
        )
        right_anchor_xyz = self.project_uv_to_base_plane(
            float(right_anchor_uv[0]), float(right_anchor_uv[1]),
            cam_info, T_base_from_cam,
            self.handle_z_nominal_m,
        )
        green_anchor_xyz = self.project_uv_to_base_plane(
            float(green_anchor_uv[0]), float(green_anchor_uv[1]),
            cam_info, T_base_from_cam,
            self.handle_z_nominal_m,
        )

        if left_anchor_xyz is None:
            left_anchor_xyz = raw_end0_xyz.copy()
        if right_anchor_xyz is None:
            right_anchor_xyz = raw_end1_xyz.copy()
        if green_anchor_xyz is None:
            green_anchor_xyz = 0.5 * (left_anchor_xyz + right_anchor_xyz)

        if float(np.dot(right_anchor_xyz[:2] - left_anchor_xyz[:2], dir_xy)) < 0.0:
            dir_xy = -dir_xy

        # 4) Hard-force final physical constraints.
        #    - z is exactly handle_z_nominal_m.
        #    - segment length is exactly handle_length_m.
        #    - center is taken from the selected constraint-matched corridor.
        center_uv = uv_center_sel + center_s * uv_dir
        center_from_uv_xyz = self.project_uv_to_base_plane(
            float(center_uv[0]), float(center_uv[1]),
            cam_info, T_base_from_cam,
            self.handle_z_nominal_m,
        )

        # Use the projected midpoint of the selected corridor as the physical
        # handle center. Since the corridor itself was selected by length/z prior,
        # this can move the center in x if another parallel corridor better fits
        # the 0.61 m handle prior.
        center_xyz = 0.5 * (raw_end0_xyz + raw_end1_xyz)
        center_xyz[2] = self.handle_z_nominal_m

        center_uv_shift_m = 0.0
        if center_from_uv_xyz is not None:
            center_uv_shift_m = float(np.linalg.norm(center_xyz[:2] - center_from_uv_xyz[:2]))

        half_len = 0.5 * self.handle_length_m
        end0_xyz = center_xyz.copy()
        end1_xyz = center_xyz.copy()
        end0_xyz[:2] = center_xyz[:2] - half_len * dir_xy
        end1_xyz[:2] = center_xyz[:2] + half_len * dir_xy
        end0_xyz[2] = self.handle_z_nominal_m
        end1_xyz[2] = self.handle_z_nominal_m

        axis_points_base_fixed = [
            np.array([
                (1.0 - t) * end0_xyz[0] + t * end1_xyz[0],
                (1.0 - t) * end0_xyz[1] + t * end1_xyz[1],
                self.handle_z_nominal_m,
            ], dtype=np.float64)
            for t in np.linspace(0.0, 1.0, self.axis_sample_count)
        ]

        forced_length_m = float(np.linalg.norm(end1_xyz[:2] - end0_xyz[:2]))

        score_2d = (
            4.0 * linearity
            + 2.0 * (1.0 - min(green_mid_ratio, 1.0))
            + 0.001 * candidate['yellow_count']
            + 0.002 * candidate['green_count']
            - 0.03 * mean_perp
            + 0.01 * constraint_score
        )

        detected = {
            'uv_center': uv_center_sel,
            'uv_dir': uv_dir,
            'uv_p0': uv_p0,
            'uv_p1': uv_p1,
            's_min': s_min,
            's_max': s_max,
            'green_pts_uv': green_pts_uv,
            'green_s': green_s,
            'center_s': center_s,
            'flipped_uv_axis': flipped_uv_axis,
            'center_uv_shift_m': center_uv_shift_m,
            'constraint_perp_offset_px': perp_offset_px,
            'constraint_score': constraint_score,
            'score_2d': score_2d,
            'projected_length_m': raw_len,
            'length_error_m': raw_len_err,
            'forced_length_m': forced_length_m,
            'center_xyz': center_xyz,
            'dir_xy': dir_xy,
            'end0_xyz': end0_xyz,
            'end1_xyz': end1_xyz,
            'axis_points_base': axis_points_base_fixed,
            'left_anchor_xyz': left_anchor_xyz,
            'right_anchor_xyz': right_anchor_xyz,
            'green_anchor_xyz': green_anchor_xyz,
        }
        return detected, fail_reasons

    def find_constraint_matched_ygy_candidate(
        self,
        yellow_mask: np.ndarray,
        green_mask: np.ndarray,
        pts_y: np.ndarray,
        s_y: np.ndarray,
        perp_y: np.ndarray,
        uv_center: np.ndarray,
        uv_dir: np.ndarray,
        normal_uv: np.ndarray,
        cam_info: CameraInfo,
        T_base_from_cam: np.ndarray,
        forced_offsets: Optional[List[float]] = None,
    ) -> Optional[dict]:
        """Select the best parallel Y-G-Y axis corridor using physical priors.

        The previous implementation only used the global mask mean line. If that
        line lies on the visible front/top edge of the tape instead of the real
        handle centerline, x can be biased by several cm after ray-plane
        projection. This function searches nearby parallel corridors and selects
        the one whose projection at z=handle_z_nominal_m has length closest to
        handle_length_m while still satisfying the Y-G-Y color evidence.
        """
        if forced_offsets is None:
            max_off = max(0.0, self.constraint_perp_search_px)
            step = max(0.5, self.constraint_perp_step_px)
            offsets = list(np.arange(-max_off, max_off + 0.5 * step, step, dtype=np.float64))
            if 0.0 not in offsets:
                offsets.append(0.0)
            # Try the original center first for deterministic tie-breaking.
            offsets = sorted(offsets, key=lambda x: (abs(float(x)) > 1e-9, abs(float(x))))
        else:
            offsets = [float(x) for x in forced_offsets]

        best = None
        best_score = -1e18

        # Precompute green pixels once.
        gy, gx = np.where(green_mask > 0)
        if len(gx) == 0:
            return None
        pts_g = np.stack([gx.astype(np.float64), gy.astype(np.float64)], axis=1)

        for off in offsets:
            uv_center_sel = uv_center + float(off) * normal_uv

            keep_axis = np.abs(perp_y - float(off)) <= self.yellow_axis_half_width_px
            yellow_count = int(np.count_nonzero(keep_axis))
            if yellow_count < self.min_yellow_pixels:
                continue

            s_axis = s_y[keep_axis]
            if len(s_axis) < self.min_yellow_pixels:
                continue

            # Percentile endpoints keep outliers from controlling the candidate.
            s_min = float(np.percentile(s_axis, 2.0))
            s_max = float(np.percentile(s_axis, 98.0))
            if s_max <= s_min + 1.0:
                continue

            # Green support for this shifted corridor.
            rel_g = pts_g - uv_center_sel[None, :]
            s_g = rel_g @ uv_dir
            perp_g = rel_g @ normal_uv
            keep_g = (
                (s_g >= s_min - self.green_margin_px) &
                (s_g <= s_max + self.green_margin_px) &
                (np.abs(perp_g) <= self.green_corridor_half_width_px)
            )
            green_count = int(np.count_nonzero(keep_g))
            if green_count < self.pair_min_green_pixels:
                continue

            green_pts_uv = pts_g[keep_g]
            green_anchor_uv = np.median(green_pts_uv, axis=0)
            green_s = float(np.median(s_g[keep_g]))

            left_keep = keep_axis & (s_y < (green_s - self.yellow_split_gap_px))
            right_keep = keep_axis & (s_y > (green_s + self.yellow_split_gap_px))
            left_count = int(np.count_nonzero(left_keep))
            right_count = int(np.count_nonzero(right_keep))
            if left_count < self.min_side_yellow_pixels or right_count < self.min_side_yellow_pixels:
                continue

            pts_left = pts_y[left_keep]
            pts_right = pts_y[right_keep]
            s_left = s_y[left_keep]
            s_right = s_y[right_keep]

            left_anchor_uv = np.median(pts_left, axis=0)
            right_anchor_uv = np.median(pts_right, axis=0)
            left_s_med = float(np.median(s_left))
            right_s_med = float(np.median(s_right))
            center_s = 0.5 * (left_s_med + right_s_med)

            half_span = max(0.5 * abs(right_s_med - left_s_med), 1e-6)
            green_mid_ratio = abs(center_s - green_s) / half_span
            if green_mid_ratio > self.green_mid_ratio_max:
                continue

            uv_p0 = uv_center_sel + s_min * uv_dir
            uv_p1 = uv_center_sel + s_max * uv_dir
            axis_points_base = self.project_axis_to_base_plane(
                uv_p0, uv_p1,
                cam_info, T_base_from_cam,
                self.handle_z_nominal_m,
                self.axis_sample_count,
            )
            if len(axis_points_base) < 5:
                continue

            raw_center_xyz, dir_xy, raw_end0_xyz, raw_end1_xyz = self.fit_xy_line_from_points(axis_points_base)
            if dir_xy is None:
                continue

            projected_length_m = float(np.linalg.norm(raw_end1_xyz[:2] - raw_end0_xyz[:2]))
            length_error_m = abs(projected_length_m - self.handle_length_m)
            if length_error_m > self.constraint_max_length_error_m:
                continue

            # Length matching is dominant. Yellow/green evidence breaks ties.
            # Offset penalty prevents jumping to a far parallel line unless it
            # clearly fits the physical 61 cm handle better.
            balance = min(left_count, right_count) / max(max(left_count, right_count), 1)
            score = (
                -self.constraint_length_weight * length_error_m
                + self.constraint_yellow_count_weight * yellow_count
                + self.constraint_green_count_weight * green_count
                + 4.0 * balance
                + 3.0 * (1.0 - min(green_mid_ratio, 1.0))
                - self.constraint_offset_weight * abs(float(off))
            )

            if score > best_score:
                best_score = score
                best = {
                    'uv_center_sel': uv_center_sel,
                    'uv_dir': uv_dir.copy(),
                    'normal_uv': normal_uv.copy(),
                    'perp_offset_px': float(off),
                    's_min': s_min,
                    's_max': s_max,
                    'center_s': center_s,
                    'green_s': green_s,
                    'green_mid_ratio': green_mid_ratio,
                    'green_pts_uv': green_pts_uv,
                    'green_anchor_uv': green_anchor_uv,
                    'left_anchor_uv': left_anchor_uv,
                    'right_anchor_uv': right_anchor_uv,
                    'uv_p0': uv_p0,
                    'uv_p1': uv_p1,
                    'axis_points_base': axis_points_base,
                    'raw_center_xyz': raw_center_xyz,
                    'dir_xy': dir_xy,
                    'raw_end0_xyz': raw_end0_xyz,
                    'raw_end1_xyz': raw_end1_xyz,
                    'projected_length_m': projected_length_m,
                    'length_error_m': length_error_m,
                    'yellow_count': yellow_count,
                    'green_count': green_count,
                    'left_count': left_count,
                    'right_count': right_count,
                    'constraint_score': float(score),
                }

        return best

    # =========================================================
    # Masks
    # =========================================================
    def make_color_mask(
        self,
        bgr: np.ndarray,
        h_min: int, h_max: int,
        s_min: int, s_max: int,
        v_min: int, v_max: int,
    ) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        if self.open_kernel > 0:
            k = np.ones((self.open_kernel, self.open_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if self.close_kernel > 0:
            k = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def build_handle_masks(
        self,
        yellow_mask: np.ndarray,
        green_pts_uv: np.ndarray,
        uv_center: np.ndarray,
        uv_dir: np.ndarray,
        s_min: float,
        s_max: float,
        shape_hw: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = shape_hw
        handle_mask = np.zeros((h, w), dtype=np.uint8)

        ys, xs = np.where(yellow_mask > 0)
        if len(xs) > 0:
            pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
            normal = np.array([-uv_dir[1], uv_dir[0]], dtype=np.float64)
            rel = pts - uv_center[None, :]
            s = rel @ uv_dir
            perp = rel @ normal
            keep = (
                (s >= s_min - self.green_margin_px) &
                (s <= s_max + self.green_margin_px) &
                (np.abs(perp) <= self.yellow_axis_half_width_px * 1.2)
            )
            kept = pts[keep]
            for p in kept:
                u = int(round(float(p[0])))
                v = int(round(float(p[1])))
                if 0 <= u < w and 0 <= v < h:
                    handle_mask[v, u] = 255

        p0 = (
            int(round(float((uv_center + s_min * uv_dir)[0]))),
            int(round(float((uv_center + s_min * uv_dir)[1]))),
        )
        p1 = (
            int(round(float((uv_center + s_max * uv_dir)[0]))),
            int(round(float((uv_center + s_max * uv_dir)[1]))),
        )
        cv2.line(handle_mask, p0, p1, 255, thickness=self.handle_mask_line_thickness_px)

        if green_pts_uv is not None and len(green_pts_uv) > 0:
            for pt in green_pts_uv:
                u = int(round(float(pt[0])))
                v = int(round(float(pt[1])))
                if 0 <= u < w and 0 <= v < h:
                    handle_mask[v, u] = 255

        pc_mask = handle_mask.copy()
        ksize = max(1, self.handle_mask_dilate_px)
        kernel = np.ones((ksize, ksize), np.uint8)
        pc_mask = cv2.dilate(pc_mask, kernel, iterations=1)

        return handle_mask, pc_mask

    # =========================================================
    # 2D helpers
    # =========================================================
    def fit_axis_2d_from_mask(self, mask: np.ndarray):
        ys, xs = np.where(mask > 0)
        if len(xs) < self.min_yellow_pixels:
            return None

        pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        center = np.mean(pts, axis=0)
        X = pts - center
        cov = X.T @ X

        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, np.argmax(vals)]

        n = np.linalg.norm(axis)
        if n < 1e-9:
            return None
        axis /= n

        proj = X @ axis
        p0 = center + np.min(proj) * axis
        p1 = center + np.max(proj) * axis

        vals_sorted = np.sort(vals)
        linearity = float(vals_sorted[-1] / max(vals_sorted[0], 1e-6))
        perp = X - np.outer(X @ axis, axis)
        mean_perp = float(np.mean(np.linalg.norm(perp, axis=1)))

        return center, axis, p0, p1, linearity, mean_perp

    def collect_green_support_pixels(
        self,
        green_mask: np.ndarray,
        midpoint: np.ndarray,
        axis: np.ndarray,
        s_min: float,
        s_max: float,
        half_width: float,
    ) -> Optional[dict]:
        ys, xs = np.where(green_mask > 0)
        if len(xs) == 0:
            return None

        pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        rel = pts - midpoint[None, :]
        s_vals = rel @ axis
        normal = np.array([-axis[1], axis[0]], dtype=np.float64)
        perp_vals = rel @ normal

        keep = (
            (s_vals >= s_min) &
            (s_vals <= s_max) &
            (np.abs(perp_vals) <= half_width)
        )

        if not np.any(keep):
            return None

        kept_pts = pts[keep]
        kept_s = s_vals[keep]
        kept_perp = perp_vals[keep]
        uv_center = np.median(kept_pts, axis=0)

        return {
            'uv_pts': kept_pts,
            'uv_center': uv_center,
            's_vals': kept_s,
            'perp_vals': kept_perp,
            's_median': float(np.median(kept_s)),
        }

    # =========================================================
    # Point cloud collection from confirmed handle ROI
    # =========================================================
    def collect_mask_points_base(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        cam_info: CameraInfo,
        T_base_from_cam: np.ndarray,
        max_points: int,
    ) -> List[np.ndarray]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return []

        idx = self.subsample_indices(len(xs), max_points)
        pts = []
        for k in idx:
            u = int(xs[k])
            v = int(ys[k])
            z = self.read_depth_m_with_local_search(depth, u, v, radius=self.depth_local_search_radius)
            if z is None:
                continue
            p_base = self.pixel_depth_to_base(u, v, z, cam_info, T_base_from_cam)
            if p_base is None:
                continue
            pts.append(p_base)
        return pts

    # =========================================================
    # Side point cloud extraction
    # =========================================================
    def collect_side_point_clouds(
        self,
        uv_p0: np.ndarray,
        uv_p1: np.ndarray,
        uv_dir: np.ndarray,
        depth: np.ndarray,
        cam_info: CameraInfo,
        T_base_from_cam: np.ndarray,
        center_xyz: np.ndarray,
        dir_xy: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        h, w = depth.shape[:2]
        n_uv = np.array([-uv_dir[1], uv_dir[0]], dtype=np.float64)
        n_uv_norm = np.linalg.norm(n_uv)
        if n_uv_norm < 1e-9:
            return [], []
        n_uv /= n_uv_norm

        pos_pts = []
        neg_pts = []

        u0, v0 = float(uv_p0[0]), float(uv_p0[1])
        u1, v1 = float(uv_p1[0]), float(uv_p1[1])

        normal_xy = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float64)

        for t in np.linspace(0.0, 1.0, self.axis_sample_count):
            u_axis = (1.0 - t) * u0 + t * u1
            v_axis = (1.0 - t) * v0 + t * v1

            for dpx in range(self.probe_px_min, self.probe_px_max + 1, self.probe_px_step):
                for sgn in (-1.0, +1.0):
                    u = int(round(u_axis + sgn * dpx * n_uv[0]))
                    v = int(round(v_axis + sgn * dpx * n_uv[1]))
                    if u < 0 or u >= w or v < 0 or v >= h:
                        continue

                    z = self.read_depth_m_with_local_search(depth, u, v, radius=self.depth_local_search_radius)
                    if z is None:
                        continue

                    p_base = self.pixel_depth_to_base(u, v, z, cam_info, T_base_from_cam)
                    if p_base is None:
                        continue

                    if abs(float(p_base[2]) - self.handle_z_nominal_m) > self.side_z_window_m:
                        continue

                    rel_xy = p_base[:2] - center_xyz[:2]
                    signed_lat = float(np.dot(rel_xy, normal_xy))

                    if signed_lat >= 0.0:
                        pos_pts.append(p_base)
                    else:
                        neg_pts.append(p_base)

        return pos_pts, neg_pts

    def side_score(self, pts: List[np.ndarray], center_xyz: np.ndarray, normal_xy: np.ndarray) -> float:
        if len(pts) == 0:
            return 0.0
        arr = np.asarray(pts, dtype=np.float64)
        rel = arr[:, :2] - center_xyz[:2]
        lateral = np.abs(rel @ normal_xy)
        return float(len(pts)) * float(np.mean(lateral) + 1e-6)

    # =========================================================
    # Projection helpers
    # =========================================================
    def project_axis_to_base_plane(
        self,
        uv_p0: np.ndarray,
        uv_p1: np.ndarray,
        cam_info: CameraInfo,
        T_base_from_cam: np.ndarray,
        target_z: float,
        n_samples: int,
    ) -> List[np.ndarray]:
        pts = []
        u0, v0 = float(uv_p0[0]), float(uv_p0[1])
        u1, v1 = float(uv_p1[0]), float(uv_p1[1])

        for t in np.linspace(0.0, 1.0, n_samples):
            u = (1.0 - t) * u0 + t * u1
            v = (1.0 - t) * v0 + t * v1
            p = self.project_uv_to_base_plane(u, v, cam_info, T_base_from_cam, target_z)
            if p is not None:
                pts.append(p)
        return pts

    def project_uv_to_base_plane(
        self,
        u: float,
        v: float,
        cam_info: CameraInfo,
        T_base_from_cam: np.ndarray,
        target_z: float,
    ) -> Optional[np.ndarray]:
        fx = cam_info.k[0]
        fy = cam_info.k[4]
        cx = cam_info.k[2]
        cy = cam_info.k[5]

        x_n = (u - cx) / fx
        y_n = (v - cy) / fy
        ray_cam = np.array([x_n, y_n, 1.0], dtype=np.float64)
        nr = np.linalg.norm(ray_cam)
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

    # =========================================================
    # Fit line in base
    # =========================================================
    @staticmethod
    def fit_xy_line_from_points(points_base: List[np.ndarray]):
        if len(points_base) < 2:
            return None, None, None, None

        pts = np.asarray(points_base, dtype=np.float64)
        center = np.mean(pts, axis=0)

        xy = pts[:, :2] - center[:2]
        cov = xy.T @ xy
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, np.argmax(vals)]

        n = np.linalg.norm(axis)
        if n < 1e-9:
            return None, None, None, None
        axis /= n

        proj = xy @ axis
        end0 = center.copy()
        end1 = center.copy()
        end0[:2] = center[:2] + np.min(proj) * axis
        end1[:2] = center[:2] + np.max(proj) * axis
        return center, axis, end0, end1

    # =========================================================
    # Depth / camera
    # =========================================================
    def read_depth_m(self, depth: np.ndarray, u: int, v: int) -> Optional[float]:
        z = float(depth[v, u]) * self.depth_scale
        if not np.isfinite(z):
            return None
        if z < self.min_depth_m or z > self.max_depth_m:
            return None
        return z

    def read_depth_m_with_local_search(
        self,
        depth: np.ndarray,
        u: int,
        v: int,
        radius: int = 2,
    ) -> Optional[float]:
        h, w = depth.shape[:2]

        z = self.read_depth_m(depth, u, v)
        if z is not None:
            return z

        for r in range(1, radius + 1):
            u0 = max(0, u - r)
            u1 = min(w - 1, u + r)
            v0 = max(0, v - r)
            v1 = min(h - 1, v + r)

            patch = depth[v0:v1 + 1, u0:u1 + 1].astype(np.float32) * self.depth_scale
            valid = np.isfinite(patch) & (patch >= self.min_depth_m) & (patch <= self.max_depth_m)
            if np.any(valid):
                vals = patch[valid]
                return float(np.median(vals))

        return None

    def pixel_depth_to_base(
        self,
        u: int,
        v: int,
        z: float,
        cam_info: CameraInfo,
        T_base_from_cam: np.ndarray,
    ) -> Optional[np.ndarray]:
        fx = cam_info.k[0]
        fy = cam_info.k[4]
        cx = cam_info.k[2]
        cy = cam_info.k[5]

        x = (float(u) - cx) * z / fx
        y = (float(v) - cy) * z / fy
        p_cam = np.array([x, y, z, 1.0], dtype=np.float64)
        p_base = T_base_from_cam @ p_cam
        return p_base[:3].astype(np.float64)

    @staticmethod
    def subsample_indices(n: int, max_points: int) -> np.ndarray:
        if n <= max_points:
            return np.arange(n, dtype=np.int64)
        return np.linspace(0, n - 1, max_points, dtype=np.int64)

    # =========================================================
    # TF
    # =========================================================
    def lookup_base_from_camera(self, msg: Image):
        source_frame = msg.header.frame_id if msg.header.frame_id else self.camera_frame_fallback

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

    def make_transform_matrix(self, transform_stamped):
        q = transform_stamped.transform.rotation
        t = transform_stamped.transform.translation
        rot = self.quat_to_rotmat(q.x, q.y, q.z, q.w)

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
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)]
        ], dtype=np.float64)

    # =========================================================
    # Temporal smoothing
    # =========================================================
    def apply_temporal_smoothing(self, center_xyz: np.ndarray, handle_yaw: float):
        if self.filtered_center is None:
            self.filtered_center = center_xyz.copy()
            self.filtered_handle_yaw = handle_yaw
        else:
            a_c = self.ema_alpha_center
            a_y = self.ema_alpha_yaw
            self.filtered_center = (1.0 - a_c) * self.filtered_center + a_c * center_xyz
            self.filtered_handle_yaw = self.blend_angle(self.filtered_handle_yaw, handle_yaw, a_y)

        return self.filtered_center.copy(), float(self.filtered_handle_yaw)

    @staticmethod
    def blend_angle(prev: float, cur: float, alpha: float) -> float:
        dx = math.cos(prev)
        dy = math.sin(prev)
        cx = math.cos(cur)
        cy = math.sin(cur)
        x = (1.0 - alpha) * dx + alpha * cx
        y = (1.0 - alpha) * dy + alpha * cy
        return math.atan2(y, x)

    # =========================================================
    # Markers
    # =========================================================
    def publish_delete_markers(self):
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.action = Marker.DELETEALL
        ma.markers.append(m)
        self.pub_markers.publish(ma)

    def publish_markers(
        self,
        stamp,
        axis_points_base: List[np.ndarray],
        pos_side_pts: List[np.ndarray],
        neg_side_pts: List[np.ndarray],
        local_pts_base: List[np.ndarray],
        center_xyz: np.ndarray,
        end0_xyz: np.ndarray,
        end1_xyz: np.ndarray,
        goal_x: float,
        goal_y: float,
        goal_theta: float,
        left_anchor: np.ndarray,
        right_anchor: np.ndarray,
        green_anchor: np.ndarray,
    ):
        markers = MarkerArray()

        mdel = Marker()
        mdel.header.frame_id = self.base_frame
        mdel.header.stamp = stamp
        mdel.action = Marker.DELETEALL
        markers.markers.append(mdel)

        for i, p in enumerate(axis_points_base[:80]):
            markers.markers.append(self.make_sphere_marker(1000 + i, 'axis_samples', stamp, p, (1.0, 1.0, 0.0), 0.012))

        for i, p in enumerate(local_pts_base[:200]):
            markers.markers.append(self.make_sphere_marker(1500 + i, 'local_pc', stamp, p, (0.7, 0.7, 0.7), 0.008))

        for i, p in enumerate(pos_side_pts[:150]):
            markers.markers.append(self.make_sphere_marker(2000 + i, 'pos_side', stamp, p, (0.0, 1.0, 1.0), 0.010))

        for i, p in enumerate(neg_side_pts[:150]):
            markers.markers.append(self.make_sphere_marker(3000 + i, 'neg_side', stamp, p, (1.0, 0.0, 1.0), 0.010))

        markers.markers.append(self.make_sphere_marker(1, 'center', stamp, center_xyz, (1.0, 0.5, 0.0), 0.035))
        markers.markers.append(self.make_sphere_marker(2, 'endpoints', stamp, end0_xyz, (1.0, 1.0, 0.0), 0.025))
        markers.markers.append(self.make_sphere_marker(3, 'endpoints', stamp, end1_xyz, (0.0, 1.0, 0.0), 0.025))

        markers.markers.append(self.make_sphere_marker(4, 'anchors', stamp, left_anchor, (1.0, 1.0, 0.0), 0.020))
        markers.markers.append(self.make_sphere_marker(5, 'anchors', stamp, right_anchor, (1.0, 1.0, 0.0), 0.020))
        markers.markers.append(self.make_sphere_marker(6, 'anchors', stamp, green_anchor, (0.0, 1.0, 0.0), 0.028))

        markers.markers.append(
            self.make_arrow_marker(
                10, 'handle_axis', stamp,
                end0_xyz, end1_xyz,
                (0.0, 1.0, 1.0)
            )
        )

        markers.markers.append(
            self.make_arrow_marker(
                11, 'goal_heading', stamp,
                np.array([goal_x, goal_y, 0.05], dtype=np.float64),
                np.array([goal_x + 0.25 * math.cos(goal_theta), goal_y + 0.25 * math.sin(goal_theta), 0.05], dtype=np.float64),
                (1.0, 0.5, 0.0)
            )
        )

        self.pub_markers.publish(markers)

    def make_sphere_marker(self, marker_id, ns, stamp, xyz, rgb, scale=0.03):
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

    def make_arrow_marker(self, marker_id, ns, stamp, start_xyz, end_xyz, rgb):
        mk = Marker()
        mk.header.frame_id = self.base_frame
        mk.header.stamp = stamp
        mk.ns = ns
        mk.id = marker_id
        mk.type = Marker.ARROW
        mk.action = Marker.ADD
        mk.pose.orientation.w = 1.0
        mk.scale.x = 0.012
        mk.scale.y = 0.025
        mk.scale.z = 0.035
        mk.color.a = 1.0
        mk.color.r = float(rgb[0])
        mk.color.g = float(rgb[1])
        mk.color.b = float(rgb[2])

        p0 = Point()
        p0.x, p0.y, p0.z = float(start_xyz[0]), float(start_xyz[1]), float(start_xyz[2])
        p1 = Point()
        p1.x, p1.y, p1.z = float(end_xyz[0]), float(end_xyz[1]), float(end_xyz[2])
        mk.points = [p0, p1]
        return mk

    # =========================================================
    # Utils
    # =========================================================
    def fail_log(self, stage: str, detail: str):
        if self.debug:
            self.get_logger().info(f'FAIL[{stage}]: {detail}')

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def yaw_to_quaternion(yaw: float):
        half = yaw * 0.5
        return (0.0, 0.0, math.sin(half), math.cos(half))


def main(args=None):
    rclpy.init(args=args)
    node = YGYCartHandleDetector()
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