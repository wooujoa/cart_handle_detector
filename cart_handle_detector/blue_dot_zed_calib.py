#!/usr/bin/env python3
from typing import Optional

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PointStamped
import tf2_ros


class BluePointBaseTransformer(Node):
    def __init__(self):
        super().__init__('blue_point_base_transformer')

        # ==================================================
        # Fixed hand-eye matrix (camera -> head)
        # ==================================================
        """
        self.T_cam_to_head = np.array([
            [ 0.0,  0.0,  1.0,  0.0238122 ],
            [-1.0,  0.0,  0.0,  0.02498203],
            [ 0.0, -1.0,  0.0, -0.0109594 ],
            [ 0.0,  0.0,  0.0,  1.0       ]
        ], dtype=np.float64)
        
        """
        self.T_cam_to_head = np.array([
            [-0.035023,  0.029973,  0.998937,  0.048565   ],
            [-0.999381,  0.002159, -0.035103,  0.02498203 ],
            [-0.003208, -0.999548,  0.029879, -0.0109594  ],
            [ 0.0,       0.0,       0.0,       1.0        ]
        ], dtype=np.float64)

        
        # ==================================================
        # Parameters
        # ==================================================
        self.declare_parameter('input_topic', '/cart_handle/blue_points_zed')
        self.declare_parameter('output_topic', '/cart_handle/blue_points_base')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('head_frame', 'head_link2')

        self.declare_parameter('use_msg_timestamp', False)
        self.declare_parameter('tf_timeout_sec', 0.2)

        self.declare_parameter('debug', True)
        self.declare_parameter('print_tf_matrix', False)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value

        self.base_frame = self.get_parameter('base_frame').value
        self.head_frame = self.get_parameter('head_frame').value

        self.use_msg_timestamp = bool(self.get_parameter('use_msg_timestamp').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)

        self.debug = bool(self.get_parameter('debug').value)
        self.print_tf_matrix = bool(self.get_parameter('print_tf_matrix').value)

        # ==================================================
        # TF
        # ==================================================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ==================================================
        # Pub / Sub
        # ==================================================
        self.sub = self.create_subscription(
            PointStamped,
            self.input_topic,
            self.point_callback,
            100
        )

        self.pub = self.create_publisher(
            PointStamped,
            self.output_topic,
            100
        )

        self.get_logger().info('========================================')
        self.get_logger().info('Blue Point Base Transformer Initialized')
        self.get_logger().info(f'input_topic       : {self.input_topic}')
        self.get_logger().info(f'output_topic      : {self.output_topic}')
        self.get_logger().info(f'base_frame        : {self.base_frame}')
        self.get_logger().info(f'head_frame        : {self.head_frame}')
        self.get_logger().info(f'use_msg_timestamp : {self.use_msg_timestamp}')
        self.get_logger().info(f'tf_timeout_sec    : {self.tf_timeout_sec}')
        self.get_logger().info(f'debug             : {self.debug}')
        self.get_logger().info('========================================')

    def point_callback(self, msg: PointStamped):
        out_msg = self.transform_camera_point_to_base(msg)
        if out_msg is None:
            return

        self.pub.publish(out_msg)

    def transform_camera_point_to_base(self, msg: PointStamped) -> Optional[PointStamped]:
        # 입력 xyz는 camera optical frame 기준 좌표값이라고 가정
        p_cam = np.array([msg.point.x, msg.point.y, msg.point.z, 1.0], dtype=np.float64)

        # camera -> head : 고정 calib 행렬 사용
        try:
            p_head = self.T_cam_to_head @ p_cam
        except Exception as e:
            self.get_logger().error(f'[TRANSFORM] camera->head failed: {repr(e)}')
            return None

        # head -> base : TF 사용
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

        if self.debug:
            self.get_logger().info(
                f'[TRANSFORM] cam=({p_cam[0]:.4f}, {p_cam[1]:.4f}, {p_cam[2]:.4f}) '
                f'-> head=({p_head[0]:.4f}, {p_head[1]:.4f}, {p_head[2]:.4f}) '
                f'-> base=({out_msg.point.x:.4f}, {out_msg.point.y:.4f}, {out_msg.point.z:.4f})'
            )

        return out_msg

    def lookup_base_from_head(self, msg: PointStamped):
        if self.use_msg_timestamp:
            try:
                target_time = Time.from_msg(msg.header.stamp)
                tf_msg = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.head_frame,
                    target_time,
                    timeout=Duration(seconds=self.tf_timeout_sec)
                )
                return tf_msg
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(
                    f'[TF] msg timestamp extrapolation failed, fallback to latest. detail={e}'
                )
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


def main(args=None):
    rclpy.init(args=args)
    node = BluePointBaseTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()