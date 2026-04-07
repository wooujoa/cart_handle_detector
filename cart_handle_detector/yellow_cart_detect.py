#!/usr/bin/env python3
from typing import List, Optional, Tuple
import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PointStamped, PoseStamped, Point
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


class YellowCartDetectPose(Node):
    def __init__(self):
        super().__init__('yellow_cart_detect_pose')

        self.T_cam_to_head = np.array([
            [-0.035023,  0.029973,  0.998937,  0.048565   ],
            [-0.999381,  0.002159, -0.035103,  0.02498203 ],
            [-0.003208, -0.999548,  0.029879, -0.0109594  ],
            [ 0.0,       0.0,       0.0,       1.0        ]
        ], dtype=np.float64)

        self.declare_parameter('camera_info_topic', '/zedm/zed_node/left/camera_info')
        self.declare_parameter('center_green_px_topic', '/cart_handle/center_green_px_zed')
        self.declare_parameter('end_green_px_topic', '/cart_handle/end_green_px_zed')
        self.declare_parameter('yellow_axis_p0_px_topic', '/cart_handle/yellow_axis_p0_px_zed')
        self.declare_parameter('yellow_axis_p1_px_topic', '/cart_handle/yellow_axis_p1_px_zed')

        self.declare_parameter('output_topic', '/cart_handle/handle_pose_base')
        self.declare_parameter('output_marker_topic', '/cart_handle/goal_pose_markers')
        self.declare_parameter('output_goal_pose_topic', '/cart_handle/goal_pose_base')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('head_frame', 'head_link2')
        self.declare_parameter('use_msg_timestamp', False)
        self.declare_parameter('tf_timeout_sec', 0.2)

        self.declare_parameter('handle_z_nominal_m', 0.93)
        self.declare_parameter('sample_count', 21)
        self.declare_parameter('standoff_m', 0.45)

        self.declare_parameter('debug', True)
        self.declare_parameter('print_tf_matrix', False)

        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.center_green_px_topic = self.get_parameter('center_green_px_topic').value
        self.end_green_px_topic = self.get_parameter('end_green_px_topic').value
        self.yellow_axis_p0_px_topic = self.get_parameter('yellow_axis_p0_px_topic').value
        self.yellow_axis_p1_px_topic = self.get_parameter('yellow_axis_p1_px_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.output_marker_topic = self.get_parameter('output_marker_topic').value
        self.output_goal_pose_topic = self.get_parameter('output_goal_pose_topic').value
        self.base_frame = self.get_parameter('base_frame').value
        self.head_frame = self.get_parameter('head_frame').value
        self.use_msg_timestamp = bool(self.get_parameter('use_msg_timestamp').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.handle_z_nominal_m = float(self.get_parameter('handle_z_nominal_m').value)
        self.sample_count = max(5, int(self.get_parameter('sample_count').value))
        self.standoff_m = float(self.get_parameter('standoff_m').value)
        self.debug = bool(self.get_parameter('debug').value)
        self.print_tf_matrix = bool(self.get_parameter('print_tf_matrix').value)

        self.camera_info: Optional[CameraInfo] = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.center_green_px_msg: Optional[PointStamped] = None
        self.end_green_px_msg: Optional[PointStamped] = None
        self.yellow_axis_p0_px_msg: Optional[PointStamped] = None
        self.yellow_axis_p1_px_msg: Optional[PointStamped] = None
        self.last_processed_key: Optional[Tuple[int, int]] = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data)
        self.sub_center = self.create_subscription(PointStamped, self.center_green_px_topic, self.center_callback, 100)
        self.sub_end = self.create_subscription(PointStamped, self.end_green_px_topic, self.end_callback, 100)
        self.sub_y0 = self.create_subscription(PointStamped, self.yellow_axis_p0_px_topic, self.y0_callback, 100)
        self.sub_y1 = self.create_subscription(PointStamped, self.yellow_axis_p1_px_topic, self.y1_callback, 100)

        self.pub = self.create_publisher(PointStamped, self.output_topic, 10)
        self.pub_goal_pose = self.create_publisher(PoseStamped, self.output_goal_pose_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.output_marker_topic, 10)

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def center_callback(self, msg: PointStamped):
        self.center_green_px_msg = msg
        self.try_process()

    def end_callback(self, msg: PointStamped):
        self.end_green_px_msg = msg
        self.try_process()

    def y0_callback(self, msg: PointStamped):
        self.yellow_axis_p0_px_msg = msg
        self.try_process()

    def y1_callback(self, msg: PointStamped):
        self.yellow_axis_p1_px_msg = msg
        self.try_process()

    def try_process(self):
        if self.camera_info is None:
            return
        if self.center_green_px_msg is None or self.end_green_px_msg is None:
            return
        if self.yellow_axis_p0_px_msg is None or self.yellow_axis_p1_px_msg is None:
            return

        key_center = self.stamp_key(self.center_green_px_msg)
        key_end = self.stamp_key(self.end_green_px_msg)
        key_y0 = self.stamp_key(self.yellow_axis_p0_px_msg)
        key_y1 = self.stamp_key(self.yellow_axis_p1_px_msg)
        if not (key_center == key_end == key_y0 == key_y1):
            return
        if self.last_processed_key == key_center:
            return

        tf_msg = self.lookup_base_from_head(self.center_green_px_msg)
        if tf_msg is None:
            return
        T_head_to_base = self.make_transform_matrix(tf_msg)
        if self.print_tf_matrix:
            self.get_logger().info(f'T_head_to_base=\n{T_head_to_base}')

        line_points_base = self.project_line_to_base_plane(
            self.yellow_axis_p0_px_msg,
            self.yellow_axis_p1_px_msg,
            T_head_to_base,
            self.handle_z_nominal_m,
            self.sample_count,
        )
        if len(line_points_base) < 5:
            if self.debug:
                self.get_logger().warn('[plane pose] not enough valid projected line points')
            return

        green_center_base = self.project_pixel_to_base_plane(self.center_green_px_msg, T_head_to_base, self.handle_z_nominal_m)
        green_end_base = self.project_pixel_to_base_plane(self.end_green_px_msg, T_head_to_base, self.handle_z_nominal_m)
        if green_center_base is None or green_end_base is None:
            if self.debug:
                self.get_logger().warn('[plane pose] failed to project green pixels to z-plane')
            return

        center_xyz, dir_xy, end0_xyz, end1_xyz = self.fit_xy_line_from_points(line_points_base)
        if center_xyz is None or dir_xy is None or end0_xyz is None or end1_xyz is None:
            return

        green_vec_xy = np.array([
            green_end_base[0] - green_center_base[0],
            green_end_base[1] - green_center_base[1],
        ], dtype=np.float64)
        ng = np.linalg.norm(green_vec_xy)
        if ng < 1e-9:
            return
        green_vec_xy /= ng

        if float(np.dot(dir_xy, green_vec_xy)) < 0.0:
            dir_xy = -dir_xy
            end0_xyz, end1_xyz = end1_xyz, end0_xyz

        handle_yaw = math.atan2(dir_xy[1], dir_xy[0])
        basket_side_yaw = self.wrap_to_pi(handle_yaw - math.pi / 2.0)
        robot_side_yaw = self.wrap_to_pi(handle_yaw + math.pi / 2.0)

        goal_x = center_xyz[0] + self.standoff_m * math.cos(robot_side_yaw)
        goal_y = center_xyz[1] + self.standoff_m * math.sin(robot_side_yaw)
        goal_theta = basket_side_yaw

        out = PointStamped()
        out.header = self.center_green_px_msg.header
        out.header.frame_id = self.base_frame
        out.point.x = float(center_xyz[0])
        out.point.y = float(center_xyz[1])
        out.point.z = float(goal_theta)
        self.pub.publish(out)

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
        self.pub_goal_pose.publish(goal_pose)

        self.publish_markers(
            stamp=out.header.stamp,
            line_points_base=line_points_base,
            green_center_base=green_center_base,
            green_end_base=green_end_base,
            center_xyz=center_xyz,
            end0_xyz=end0_xyz,
            end1_xyz=end1_xyz,
            handle_yaw=handle_yaw,
            goal_x=goal_x,
            goal_y=goal_y,
            goal_theta=goal_theta,
        )

        if self.debug:
            self.get_logger().info(
                '[PLANE HANDLE RESULT]\n'
                f'  samples_n    = {len(line_points_base)}\n'
                f'  end0         = ({end0_xyz[0]:.4f}, {end0_xyz[1]:.4f}, {end0_xyz[2]:.4f})\n'
                f'  end1         = ({end1_xyz[0]:.4f}, {end1_xyz[1]:.4f}, {end1_xyz[2]:.4f})\n'
                f'  center_g     = ({green_center_base[0]:.4f}, {green_center_base[1]:.4f}, {green_center_base[2]:.4f})\n'
                f'  end_g        = ({green_end_base[0]:.4f}, {green_end_base[1]:.4f}, {green_end_base[2]:.4f})\n'
                f'  handle_ctr   = ({center_xyz[0]:.4f}, {center_xyz[1]:.4f}, {center_xyz[2]:.4f})\n'
                f'  handle_yaw   = {handle_yaw:.6f} rad ({math.degrees(handle_yaw):.2f} deg)\n'
                f'  goal_theta   = {goal_theta:.6f} rad ({math.degrees(goal_theta):.2f} deg)\n'
                f'  goal_pos     = ({goal_x:.4f}, {goal_y:.4f})'
            )

        self.last_processed_key = key_center

    def project_line_to_base_plane(
        self,
        p0_msg: PointStamped,
        p1_msg: PointStamped,
        T_head_to_base: np.ndarray,
        target_z: float,
        n_samples: int,
    ) -> List[np.ndarray]:
        u0 = float(p0_msg.point.x)
        v0 = float(p0_msg.point.y)
        u1 = float(p1_msg.point.x)
        v1 = float(p1_msg.point.y)

        points_base: List[np.ndarray] = []
        for t in np.linspace(0.0, 1.0, n_samples):
            u = (1.0 - t) * u0 + t * u1
            v = (1.0 - t) * v0 + t * v1
            p = self.project_uv_to_base_plane(u, v, T_head_to_base, target_z)
            if p is not None:
                points_base.append(p)
        return points_base

    def project_pixel_to_base_plane(self, px_msg: PointStamped, T_head_to_base: np.ndarray, target_z: float) -> Optional[np.ndarray]:
        return self.project_uv_to_base_plane(float(px_msg.point.x), float(px_msg.point.y), T_head_to_base, target_z)

    def project_uv_to_base_plane(self, u: float, v: float, T_head_to_base: np.ndarray, target_z: float) -> Optional[np.ndarray]:
        x_n = (u - self.cx) / self.fx
        y_n = (v - self.cy) / self.fy
        dir_cam = np.array([x_n, y_n, 1.0], dtype=np.float64)
        nd = np.linalg.norm(dir_cam)
        if nd < 1e-12:
            return None
        dir_cam /= nd

        origin_cam = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        origin_head = self.T_cam_to_head @ origin_cam
        dir_head = self.T_cam_to_head[:3, :3] @ dir_cam

        origin_base_h = T_head_to_base @ origin_head
        origin_base = origin_base_h[:3]
        dir_base = T_head_to_base[:3, :3] @ dir_head

        dz = float(dir_base[2])
        if abs(dz) < 1e-9:
            return None

        scale = (target_z - float(origin_base[2])) / dz
        if scale <= 0.0:
            return None

        p = origin_base + scale * dir_base
        return np.array([float(p[0]), float(p[1]), float(target_z)], dtype=np.float64)

    @staticmethod
    def fit_xy_line_from_points(points_base: List[np.ndarray]):
        if len(points_base) < 2:
            return None, None, None, None
        pts = np.asarray(points_base, dtype=np.float64)
        center = np.mean(pts, axis=0)
        xy = pts[:, :2] - center[:2]
        cov = xy.T @ xy
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)
        axis = vecs[:, order[-1]]
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
            except Exception:
                pass
        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                self.head_frame,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_sec)
            )
        except Exception:
            return None

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

    @staticmethod
    def stamp_key(msg) -> Tuple[int, int]:
        return (msg.header.stamp.sec, msg.header.stamp.nanosec)

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def yaw_to_quaternion(yaw: float):
        half = yaw * 0.5
        return (0.0, 0.0, math.sin(half), math.cos(half))

    def publish_markers(
        self,
        stamp,
        line_points_base: List[np.ndarray],
        green_center_base: np.ndarray,
        green_end_base: np.ndarray,
        center_xyz: np.ndarray,
        end0_xyz: np.ndarray,
        end1_xyz: np.ndarray,
        handle_yaw: float,
        goal_x: float,
        goal_y: float,
        goal_theta: float,
    ):
        markers = MarkerArray()
        for i, p in enumerate(line_points_base[:50]):
            markers.markers.append(self.make_sphere_marker(100 + i, 'line_samples', stamp, p, (1.0, 1.0, 0.0), scale=0.012))
        markers.markers.append(self.make_sphere_marker(0, 'line_endpoints', stamp, end0_xyz, (1.0, 1.0, 0.0), scale=0.03))
        markers.markers.append(self.make_sphere_marker(1, 'line_endpoints', stamp, end1_xyz, (0.0, 1.0, 1.0), scale=0.03))
        markers.markers.append(self.make_sphere_marker(2, 'green_pts', stamp, green_center_base, (1.0, 0.0, 0.0), scale=0.03))
        markers.markers.append(self.make_sphere_marker(3, 'green_pts', stamp, green_end_base, (0.0, 1.0, 0.0), scale=0.03))
        markers.markers.append(self.make_sphere_marker(4, 'handle_center', stamp, center_xyz, (1.0, 0.5, 0.0), scale=0.035))

        markers.markers.append(
            self.make_arrow_marker(
                20, 'handle_vec', stamp,
                center_xyz,
                (center_xyz[0] + 0.25 * math.cos(handle_yaw), center_xyz[1] + 0.25 * math.sin(handle_yaw), center_xyz[2]),
                (0.0, 1.0, 1.0),
            )
        )
        markers.markers.append(
            self.make_arrow_marker(
                21, 'goal_heading', stamp,
                (goal_x, goal_y, 0.05),
                (goal_x + 0.25 * math.cos(goal_theta), goal_y + 0.25 * math.sin(goal_theta), 0.05),
                (1.0, 0.5, 0.0),
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
        mk.color.r = rgb[0]
        mk.color.g = rgb[1]
        mk.color.b = rgb[2]
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
        mk.color.r = rgb[0]
        mk.color.g = rgb[1]
        mk.color.b = rgb[2]
        p0 = Point(); p0.x, p0.y, p0.z = start_xyz
        p1 = Point(); p1.x, p1.y, p1.z = end_xyz
        mk.points = [p0, p1]
        return mk


def main(args=None):
    rclpy.init(args=args)
    node = YellowCartDetectPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()