#!/usr/bin/env python3
from typing import Dict, List, Optional, Tuple
import itertools
import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PointStamped, PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


class CartHandleGoalPoseSelectorZed(Node):
    def __init__(self):
        super().__init__('cart_handle_goal_pose_selector_zed')

        # ==================================================
        # Fixed hand-eye matrix (camera -> head)
        # ==================================================
        self.T_cam_to_head = np.array([
            [-0.035023,  0.029973,  0.998937,  0.048565   ],
            [-0.999381,  0.002159, -0.035103,  0.02498203 ],
            [-0.003208, -0.999548,  0.029879, -0.0109594  ],
            [ 0.0,       0.0,       0.0,       1.0        ]
        ], dtype=np.float64)

        # ==================================================
        # Parameters
        # ==================================================
        self.declare_parameter('input_points_topic', '/cart_handle/blue_points_zed')

        # 최종 출력: x=d0.x, y=d0.y, z=goal_theta
        self.declare_parameter('output_topic', '/cart_handle/handle_pose_base')

        # debug / viz 용
        self.declare_parameter('output_marker_topic', '/cart_handle/goal_pose_markers')
        self.declare_parameter('output_goal_pose_topic', '/cart_handle/goal_pose_base')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('head_frame', 'head_link2')

        self.declare_parameter('use_msg_timestamp', False)
        self.declare_parameter('tf_timeout_sec', 0.2)

        # frame grouping
        self.declare_parameter('frame_finalize_delay_sec', 0.15)
        self.declare_parameter('process_timer_sec', 0.02)

        # z constraint
        self.declare_parameter('z_min_m', 0.80)
        self.declare_parameter('z_max_m', 1.00)

        # point spacing rule
        self.declare_parameter('dist_d0_d1_m', 0.05)
        self.declare_parameter('dist_d1_d2_m', 0.17)
        self.declare_parameter('dist_d0_d2_m', 0.22)

        self.declare_parameter('dist_tol_m', 0.05)
        self.declare_parameter('line_tol_m', 0.03)

        # robot goal offset from d0 toward basket-less side
        self.declare_parameter('standoff_m', 0.45)

        self.declare_parameter('min_points_to_consider', 3)
        self.declare_parameter('debug', True)
        self.declare_parameter('print_tf_matrix', False)

        self.input_points_topic = self.get_parameter('input_points_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.output_marker_topic = self.get_parameter('output_marker_topic').value
        self.output_goal_pose_topic = self.get_parameter('output_goal_pose_topic').value

        self.base_frame = self.get_parameter('base_frame').value
        self.head_frame = self.get_parameter('head_frame').value

        self.use_msg_timestamp = bool(self.get_parameter('use_msg_timestamp').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)

        self.frame_finalize_delay_sec = float(self.get_parameter('frame_finalize_delay_sec').value)
        self.process_timer_sec = float(self.get_parameter('process_timer_sec').value)

        self.z_min_m = float(self.get_parameter('z_min_m').value)
        self.z_max_m = float(self.get_parameter('z_max_m').value)

        self.dist_d0_d1_m = float(self.get_parameter('dist_d0_d1_m').value)
        self.dist_d1_d2_m = float(self.get_parameter('dist_d1_d2_m').value)
        self.dist_d0_d2_m = float(self.get_parameter('dist_d0_d2_m').value)

        self.dist_tol_m = float(self.get_parameter('dist_tol_m').value)
        self.line_tol_m = float(self.get_parameter('line_tol_m').value)

        self.standoff_m = float(self.get_parameter('standoff_m').value)

        self.min_points_to_consider = int(self.get_parameter('min_points_to_consider').value)
        self.debug = bool(self.get_parameter('debug').value)
        self.print_tf_matrix = bool(self.get_parameter('print_tf_matrix').value)

        # ==================================================
        # TF
        # ==================================================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ==================================================
        # Frame buffers
        # ==================================================
        self.frame_points: Dict[Tuple[int, int], List[PointStamped]] = {}

        # ==================================================
        # Sub / Pub
        # ==================================================
        self.sub_points = self.create_subscription(
            PointStamped,
            self.input_points_topic,
            self.points_callback,
            100
        )

        # 최종 출력 하나만 사용
        self.pub = self.create_publisher(PointStamped, self.output_topic, 10)

        # 디버그용 유지
        self.pub_goal_pose = self.create_publisher(PoseStamped, self.output_goal_pose_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.output_marker_topic, 10)

        self.timer = self.create_timer(self.process_timer_sec, self.process_pending_frames)

        self.get_logger().info('========================================')
        self.get_logger().info('Cart Handle Goal Pose Selector ZED Initialized')
        self.get_logger().info(f'input_points_topic : {self.input_points_topic}')
        self.get_logger().info(f'output_topic       : {self.output_topic}')
        self.get_logger().info(f'output_goal_pose   : {self.output_goal_pose_topic}')
        self.get_logger().info(f'output_marker_topic: {self.output_marker_topic}')
        self.get_logger().info(f'base_frame         : {self.base_frame}')
        self.get_logger().info(f'head_frame         : {self.head_frame}')
        self.get_logger().info(f'z range            : [{self.z_min_m:.3f}, {self.z_max_m:.3f}]')
        self.get_logger().info(
            f'target distances   : d0-d1={self.dist_d0_d1_m:.3f}, '
            f'd1-d2={self.dist_d1_d2_m:.3f}, d0-d2={self.dist_d0_d2_m:.3f}'
        )
        self.get_logger().info(f'standoff_m         : {self.standoff_m:.3f}')
        self.get_logger().info('========================================')

    # ==================================================
    # Callbacks
    # ==================================================
    def points_callback(self, msg: PointStamped):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)

        base_msg = self.transform_camera_point_to_base(msg)
        if base_msg is None:
            return

        if key not in self.frame_points:
            self.frame_points[key] = []
        self.frame_points[key].append(base_msg)

        if self.debug:
            self.get_logger().info(
                f'[BUFFER] stamp={key[0]}.{key[1]:09d} '
                f'append base_xyz=({base_msg.point.x:.4f}, {base_msg.point.y:.4f}, {base_msg.point.z:.4f}) '
                f'count={len(self.frame_points[key])}'
            )

    def process_pending_frames(self):
        if not self.frame_points:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        keys_to_process = []

        for key in self.frame_points.keys():
            stamp_sec = float(key[0]) + float(key[1]) * 1e-9
            age = now_sec - stamp_sec
            if age >= self.frame_finalize_delay_sec:
                keys_to_process.append(key)

        for key in sorted(keys_to_process):
            points = self.frame_points.pop(key, [])
            self.process_frame_group(key, points)

    # ==================================================
    # Main per-frame processing
    # ==================================================
    def process_frame_group(self, key: Tuple[int, int], points_base: List[PointStamped]):
        stamp_str = f'{key[0]}.{key[1]:09d}'

        if self.debug:
            self.get_logger().info(f'\n[FRAME PROCESS] stamp={stamp_str} total_points={len(points_base)}')

        if len(points_base) < self.min_points_to_consider:
            if self.debug:
                self.get_logger().warn(
                    f'[FRAME PROCESS] skip: not enough points ({len(points_base)} < {self.min_points_to_consider})'
                )
            return

        z_filtered = []
        for i, p in enumerate(points_base):
            if self.z_min_m <= p.point.z <= self.z_max_m:
                z_filtered.append((i, p))

        if self.debug:
            self.get_logger().info(
                f'[FRAME PROCESS] z_filtered={len(z_filtered)} / {len(points_base)} '
                f'within [{self.z_min_m:.3f}, {self.z_max_m:.3f}]'
            )

        if len(z_filtered) < 3:
            if self.debug:
                self.get_logger().warn('[FRAME PROCESS] skip: fewer than 3 points after z filtering')
            return

        selected = self.select_best_triplet([p for _, p in z_filtered])
        if selected is None:
            if self.debug:
                self.get_logger().warn('[FRAME PROCESS] no valid triplet found')
            return

        d0, d1, d2 = selected

        # handle vector: d0 -> d2
        handle_yaw = self.compute_yaw(d0, d2)

        # basket is always on the RIGHT of d0->d2
        basket_side_yaw = self.wrap_to_pi(handle_yaw - math.pi / 2.0)

        # basket-less side = LEFT normal
        robot_side_yaw = self.wrap_to_pi(handle_yaw + math.pi / 2.0)

        # robot goal position: move to basket-less side
        goal_x = d0.point.x + self.standoff_m * math.cos(robot_side_yaw)
        goal_y = d0.point.y + self.standoff_m * math.sin(robot_side_yaw)

        # robot should face the handle from that side
        goal_theta = basket_side_yaw

        self.publish_results(
            d0=d0,
            d1=d1,
            d2=d2,
            handle_yaw=handle_yaw,
            goal_x=goal_x,
            goal_y=goal_y,
            goal_theta=goal_theta,
            robot_side_yaw=robot_side_yaw
        )

        self.get_logger().info(
            '[CART HANDLE RESULT]\n'
            f'  d0(center) = ({d0.point.x:.4f}, {d0.point.y:.4f}, {d0.point.z:.4f})\n'
            f'  d1         = ({d1.point.x:.4f}, {d1.point.y:.4f}, {d1.point.z:.4f})\n'
            f'  d2         = ({d2.point.x:.4f}, {d2.point.y:.4f}, {d2.point.z:.4f})\n'
            f'  handle_yaw = {handle_yaw:.6f} rad ({math.degrees(handle_yaw):.2f} deg)\n'
            f'  goal_theta = {goal_theta:.6f} rad ({math.degrees(goal_theta):.2f} deg)'
        )

    def select_best_triplet(self, points: List[PointStamped]) -> Optional[Tuple[PointStamped, PointStamped, PointStamped]]:
        best_score = float('inf')
        best_triplet = None

        for triplet in itertools.combinations(points, 3):
            p0, p1, p2 = triplet
            cand = [p0, p1, p2]

            # 세 점 중 어떤 점이 중간점(d1)인지 모두 시험
            for mid_idx in range(3):
                d1 = cand[mid_idx]
                others = [cand[i] for i in range(3) if i != mid_idx]
                a, b = others[0], others[1]

                dist_a = self.distance_3d(d1, a)
                dist_b = self.distance_3d(d1, b)

                # d1에서 5cm쪽을 d0, 17cm쪽을 d2로 배정
                err_case1 = abs(dist_a - self.dist_d0_d1_m) + abs(dist_b - self.dist_d1_d2_m)
                err_case2 = abs(dist_b - self.dist_d0_d1_m) + abs(dist_a - self.dist_d1_d2_m)

                if err_case1 <= err_case2:
                    d0, d2 = a, b
                    err_01 = abs(dist_a - self.dist_d0_d1_m)
                    err_12 = abs(dist_b - self.dist_d1_d2_m)
                else:
                    d0, d2 = b, a
                    err_01 = abs(dist_b - self.dist_d0_d1_m)
                    err_12 = abs(dist_a - self.dist_d1_d2_m)

                dist_02 = self.distance_3d(d0, d2)
                err_02 = abs(dist_02 - self.dist_d0_d2_m)

                if err_01 > self.dist_tol_m or err_12 > self.dist_tol_m or err_02 > self.dist_tol_m:
                    continue

                line_err = self.middle_point_line_error(d0, d1, d2)
                if line_err > self.line_tol_m:
                    continue

                score = 3.0 * (err_01 + err_12 + err_02) + 2.0 * line_err

                if self.debug:
                    self.get_logger().info(
                        f'[TRIPLET CAND] '
                        f'd0-d1={self.distance_3d(d0,d1):.4f}, '
                        f'd1-d2={self.distance_3d(d1,d2):.4f}, '
                        f'd0-d2={dist_02:.4f}, '
                        f'line_err={line_err:.4f}, '
                        f'score={score:.4f}'
                    )

                if score < best_score:
                    best_score = score
                    best_triplet = (d0, d1, d2)

        return best_triplet

    # ==================================================
    # Geometry helpers
    # ==================================================
    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def distance_3d(p1: PointStamped, p2: PointStamped) -> float:
        dx = p1.point.x - p2.point.x
        dy = p1.point.y - p2.point.y
        dz = p1.point.z - p2.point.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def middle_point_line_error(self, p0: PointStamped, p1: PointStamped, p2: PointStamped) -> float:
        a = np.array([p0.point.x, p0.point.y, p0.point.z], dtype=np.float64)
        b = np.array([p1.point.x, p1.point.y, p1.point.z], dtype=np.float64)
        c = np.array([p2.point.x, p2.point.y, p2.point.z], dtype=np.float64)

        ac = c - a
        ac_norm = np.linalg.norm(ac)
        if ac_norm < 1e-9:
            return 1e9

        proj_len = np.dot((b - a), ac) / ac_norm
        proj = a + (proj_len / ac_norm) * ac
        return float(np.linalg.norm(b - proj))

    @staticmethod
    def compute_yaw(p_from: PointStamped, p_to: PointStamped) -> float:
        dx = p_to.point.x - p_from.point.x
        dy = p_to.point.y - p_from.point.y
        return math.atan2(dy, dx)

    @staticmethod
    def yaw_to_quaternion(yaw: float):
        half = yaw * 0.5
        qz = math.sin(half)
        qw = math.cos(half)
        return (0.0, 0.0, qz, qw)

    # ==================================================
    # Transform: camera -> head -> base
    # ==================================================
    def transform_camera_point_to_base(self, msg: PointStamped) -> Optional[PointStamped]:
        p_cam = np.array([msg.point.x, msg.point.y, msg.point.z, 1.0], dtype=np.float64)

        try:
            p_head = self.T_cam_to_head @ p_cam
        except Exception as e:
            self.get_logger().error(f'[TRANSFORM] camera->head failed: {repr(e)}')
            return None

        tf_msg = self.lookup_base_from_head(msg)
        if tf_msg is None:
            return None

        try:
            T_head_to_base = self.make_transform_matrix(tf_msg)
            p_base = T_head_to_base @ p_head
        except Exception as e:
            self.get_logger().error(f'[TRANSFORM] head->base failed: {repr(e)}')
            return None

        out_msg = PointStamped()
        out_msg.header = msg.header
        out_msg.header.frame_id = self.base_frame
        out_msg.point.x = float(p_base[0])
        out_msg.point.y = float(p_base[1])
        out_msg.point.z = float(p_base[2])

        return out_msg

    def lookup_base_from_head(self, msg: PointStamped):
        if self.use_msg_timestamp:
            try:
                target_time = Time.from_msg(msg.header.stamp)
                return self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.head_frame,
                    target_time,
                    timeout=Duration(seconds=self.tf_timeout_sec)
                )
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(f'[TF] msg timestamp extrapolation failed, fallback to latest. detail={e}')
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException) as e:
                self.get_logger().warn(f'[TF] msg timestamp lookup failed: {e}')
            except Exception as e:
                self.get_logger().error(f'[TF] unexpected msg timestamp error: {repr(e)}')

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.head_frame,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_sec)
            )
            if self.print_tf_matrix:
                self.print_tf_debug(tf_msg)
            return tf_msg
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'[TF] latest lookup failed: {e}')
            return None
        except Exception as e:
            self.get_logger().error(f'[TF] unexpected latest lookup error: {repr(e)}')
            return None

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

    def make_transform_matrix(self, transform_stamped):
        q = transform_stamped.transform.rotation
        t = transform_stamped.transform.translation

        rot = self.quat_to_rotmat(q.x, q.y, q.z, q.w)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def print_tf_debug(self, tf_msg):
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        rot = self.quat_to_rotmat(q.x, q.y, q.z, q.w)

        self.get_logger().info(
            '[TF DETAIL] '
            f'translation=({t.x:.4f}, {t.y:.4f}, {t.z:.4f}), '
            f'quaternion=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})'
        )
        self.get_logger().info(
            '[TF MATRIX]\n'
            f'[{rot[0,0]: .4f} {rot[0,1]: .4f} {rot[0,2]: .4f} | {t.x: .4f}]\n'
            f'[{rot[1,0]: .4f} {rot[1,1]: .4f} {rot[1,2]: .4f} | {t.y: .4f}]\n'
            f'[{rot[2,0]: .4f} {rot[2,1]: .4f} {rot[2,2]: .4f} | {t.z: .4f}]'
        )

    # ==================================================
    # Publish
    # ==================================================
    def publish_results(
        self,
        d0: PointStamped,
        d1: PointStamped,
        d2: PointStamped,
        handle_yaw: float,
        goal_x: float,
        goal_y: float,
        goal_theta: float,
        robot_side_yaw: float,
    ):
        # --------------------------------------------------
        # 최종 출력:
        #   x = d0.x
        #   y = d0.y
        #   z = goal_theta
        # --------------------------------------------------
        out = PointStamped()
        out.header = d0.header
        out.header.frame_id = self.base_frame
        out.point.x = float(d0.point.x)
        out.point.y = float(d0.point.y)
        out.point.z = float(goal_theta)
        self.pub.publish(out)

        # --------------------------------------------------
        # 디버그용 goal pose
        # --------------------------------------------------
        goal_pose = PoseStamped()
        goal_pose.header = d0.header
        goal_pose.header.frame_id = self.base_frame
        goal_pose.pose.position.x = float(goal_x)
        goal_pose.pose.position.y = float(goal_y)
        goal_pose.pose.position.z = 0.0

        qx, qy, qz, qw = self.yaw_to_quaternion(goal_theta)
        goal_pose.pose.orientation.x = qx
        goal_pose.pose.orientation.y = qy
        goal_pose.pose.orientation.z = qz
        goal_pose.pose.orientation.w = qw
        self.pub_goal_pose.publish(goal_pose)

        # --------------------------------------------------
        # RViz markers
        # --------------------------------------------------
        markers = MarkerArray()

        pts = [d0, d1, d2]
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        for i, (p, c) in enumerate(zip(pts, colors)):
            mk = Marker()
            mk.header.frame_id = self.base_frame
            mk.header.stamp = d0.header.stamp
            mk.ns = 'cart_handle_points'
            mk.id = i
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.pose.position.x = p.point.x
            mk.pose.position.y = p.point.y
            mk.pose.position.z = p.point.z
            mk.pose.orientation.w = 1.0
            mk.scale.x = 0.03
            mk.scale.y = 0.03
            mk.scale.z = 0.03
            mk.color.a = 1.0
            mk.color.r = c[0]
            mk.color.g = c[1]
            mk.color.b = c[2]
            markers.markers.append(mk)

        line = Marker()
        line.header.frame_id = self.base_frame
        line.header.stamp = d0.header.stamp
        line.ns = 'cart_handle_line'
        line.id = 10
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.01
        line.color.a = 1.0
        line.color.r = 1.0
        line.color.g = 1.0
        line.color.b = 0.0
        line.points = [d0.point, d1.point, d2.point]
        markers.markers.append(line)

        markers.markers.append(
            self.make_arrow_marker(
                marker_id=20,
                ns='handle_vec',
                stamp=d0.header.stamp,
                start_xyz=(d0.point.x, d0.point.y, d0.point.z),
                end_xyz=(
                    d0.point.x + 0.20 * math.cos(handle_yaw),
                    d0.point.y + 0.20 * math.sin(handle_yaw),
                    d0.point.z
                ),
                rgb=(0.0, 1.0, 1.0)
            )
        )

        markers.markers.append(
            self.make_arrow_marker(
                marker_id=21,
                ns='robot_side_vec',
                stamp=d0.header.stamp,
                start_xyz=(d0.point.x, d0.point.y, d0.point.z),
                end_xyz=(
                    d0.point.x + 0.20 * math.cos(robot_side_yaw),
                    d0.point.y + 0.20 * math.sin(robot_side_yaw),
                    d0.point.z
                ),
                rgb=(1.0, 0.0, 1.0)
            )
        )

        goal_sphere = Marker()
        goal_sphere.header.frame_id = self.base_frame
        goal_sphere.header.stamp = d0.header.stamp
        goal_sphere.ns = 'goal_pose'
        goal_sphere.id = 30
        goal_sphere.type = Marker.SPHERE
        goal_sphere.action = Marker.ADD
        goal_sphere.pose.position.x = goal_x
        goal_sphere.pose.position.y = goal_y
        goal_sphere.pose.position.z = 0.05
        goal_sphere.pose.orientation.w = 1.0
        goal_sphere.scale.x = 0.06
        goal_sphere.scale.y = 0.06
        goal_sphere.scale.z = 0.06
        goal_sphere.color.a = 1.0
        goal_sphere.color.r = 1.0
        goal_sphere.color.g = 1.0
        goal_sphere.color.b = 1.0
        markers.markers.append(goal_sphere)

        markers.markers.append(
            self.make_arrow_marker(
                marker_id=31,
                ns='goal_heading',
                stamp=d0.header.stamp,
                start_xyz=(goal_x, goal_y, 0.05),
                end_xyz=(
                    goal_x + 0.25 * math.cos(goal_theta),
                    goal_y + 0.25 * math.sin(goal_theta),
                    0.05
                ),
                rgb=(1.0, 0.5, 0.0)
            )
        )

        self.pub_markers.publish(markers)

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
        mk.color.r = rgb[0]
        mk.color.g = rgb[1]
        mk.color.b = rgb[2]

        p0 = Point()
        p0.x, p0.y, p0.z = start_xyz
        p1 = Point()
        p1.x, p1.y, p1.z = end_xyz
        mk.points = [p0, p1]
        return mk


def main():
    rclpy.init()
    node = CartHandleGoalPoseSelectorZed()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()