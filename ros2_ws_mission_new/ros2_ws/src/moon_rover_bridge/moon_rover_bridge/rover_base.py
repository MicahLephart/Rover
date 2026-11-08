"""Shared logic for a ROS 2 vehicle node that mirrors and drives a Gazebo rover in moon.sdf.

Gazebo -> ROS 2
    Reads the rover's own pose stream (/model/<model>/pose, bridged to tf2_msgs/TFMessage),
    picks out the model and its parts (links), and republishes:
      * TF:  map -> <model>,  <model> -> <model>/<part>
      * ~/pose             (geometry_msgs/PoseStamped, model in map frame)
      * ~/parts/<part>     (geometry_msgs/PoseStamped, one per part)
    If the `parts` parameter is empty, every link found in the stream is tracked.

ROS 2 -> Gazebo (diff drive, mirroring the DiffDrive plugin in moon.sdf)
    * Publishes geometry_msgs/Twist on /model/<model>/cmd_vel (bridged to the DiffDrive plugin).
    * ~/cmd_vel      (Twist)        manual drive, forwarded (clamped) to the Gazebo rover.
    * ~/goal_pose    (PoseStamped)  go-to-point controller (same law as demo.py), map frame.
    * ~/set_pose     (PoseStamped)  teleport via the /world/<world>/set_pose service
                                    (needs a service bridge; ignored with a warning if absent).
"""

import math
import re
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster

try:  # only needed for the optional teleport service
    from ros_gz_interfaces.msg import Entity
    from ros_gz_interfaces.srv import SetEntityPose
    _HAVE_SET_POSE = True
except ImportError:  # pragma: no cover
    _HAVE_SET_POSE = False


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _topic_safe(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]', '_', name)


class RoverBridgeNode(Node):
    DEFAULT_MODEL = 'rover'
    DEFAULT_PARTS: List[str] = []   # empty = track every link in the pose stream

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.declare_parameter('world_name', 'moon')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('model_name', self.DEFAULT_MODEL)
        self.declare_parameter('parts', list(self.DEFAULT_PARTS))
        self.declare_parameter('world_offset_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('parts_relative_to_model', True)
        # diff-drive controller limits (match the DiffDrive plugin in moon.sdf)
        self.declare_parameter('max_linear', 0.5)
        self.declare_parameter('max_angular', 1.0)
        self.declare_parameter('kv', 0.5)
        self.declare_parameter('ktheta', 1.0)
        self.declare_parameter('goal_tolerance', 0.15)

        gp = self.get_parameter
        self.world = gp('world_name').value
        self.map_frame = gp('map_frame').value
        self.model = gp('model_name').value
        self.parts: List[str] = list(gp('parts').value)
        self.track_all_parts = len(self.parts) == 0
        self.offset = [float(v) for v in gp('world_offset_xyz').value]
        self.parts_rel_model = gp('parts_relative_to_model').value
        self.max_lin = float(gp('max_linear').value)
        self.max_ang = float(gp('max_angular').value)
        self.kv = float(gp('kv').value)
        self.ktheta = float(gp('ktheta').value)
        self.goal_tol = float(gp('goal_tolerance').value)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.model_pub = self.create_publisher(PoseStamped, '~/pose', 10)
        self.part_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        for p in self.parts:
            self._part_pub(p)

        # ---- Gazebo pose stream for this rover only (avoids link-name clashes between rovers)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.pose_topic = f'/model/{self.model}/pose'
        self.create_subscription(TFMessage, self.pose_topic, self._on_poses, qos)

        # ---- diff drive output (bridged to the DiffDrive plugin)
        self.cmd_pub = self.create_publisher(Twist, f'/model/{self.model}/cmd_vel', 10)
        self.create_subscription(Twist, '~/cmd_vel', self._on_cmd_vel, 10)

        # ---- go-to-point controller
        self.model_pose: Optional[PoseStamped] = None   # map frame
        self.goal: Optional[PoseStamped] = None
        self.create_subscription(PoseStamped, '~/goal_pose', self._on_goal, 10)
        self.create_timer(0.1, self._control_step)

        # ---- optional teleport
        self.set_pose_client = None
        if _HAVE_SET_POSE:
            self.set_pose_client = self.create_client(SetEntityPose, f'/world/{self.world}/set_pose')
            self.create_subscription(PoseStamped, '~/set_pose', self._on_set_pose, 10)

        self._seen_model = False
        self._warn_timer = self.create_timer(5.0, self._warn_if_silent)
        self.get_logger().info(
            f"Rover '{self.model}': pose {self.pose_topic}, parts "
            f"{'ALL links' if self.track_all_parts else self.parts}")

    # ------------------------------------------------------------------ helpers
    def _part_pub(self, part: str):
        if part not in self.part_pubs:
            self.part_pubs[part] = self.create_publisher(
                PoseStamped, f'~/parts/{_topic_safe(part)}', 10)
        return self.part_pubs[part]

    def _classify(self, child: str):
        """('model', None) | ('part', link_name) | (None, None) for a name in this rover's stream."""
        pieces = [p for p in child.replace('/', '::').split('::') if p]
        if not pieces:
            return None, None
        if len(pieces) == 1 and pieces[0] == self.model:
            return 'model', None
        link = pieces[-1]
        if link == self.model:
            return None, None
        if self.track_all_parts or link in self.parts:
            return 'part', link
        return None, None

    # ------------------------------------------------------------------ Gazebo -> ROS
    def _on_poses(self, msg: TFMessage):
        now = self.get_clock().now().to_msg()
        out: List[TransformStamped] = []

        for tf in msg.transforms:
            kind, part = self._classify(tf.child_frame_id)
            if kind is None:
                continue

            t = TransformStamped()
            t.header.stamp = now
            t.transform = tf.transform
            ps = PoseStamped()
            ps.header.stamp = now
            ps.pose.position.x = tf.transform.translation.x
            ps.pose.position.y = tf.transform.translation.y
            ps.pose.position.z = tf.transform.translation.z
            ps.pose.orientation = tf.transform.rotation

            in_map = (kind == 'model') or not self.parts_rel_model
            if in_map:
                for axis, off in zip('xyz', self.offset):
                    setattr(t.transform.translation, axis, getattr(t.transform.translation, axis) + off)
                    setattr(ps.pose.position, axis, getattr(ps.pose.position, axis) + off)

            if kind == 'model':
                self._seen_model = True
                t.header.frame_id, t.child_frame_id = self.map_frame, self.model
                ps.header.frame_id = self.map_frame
                self.model_pose = ps
                self.model_pub.publish(ps)
            else:
                parent = self.map_frame if not self.parts_rel_model else self.model
                t.header.frame_id, t.child_frame_id = parent, f'{self.model}/{part}'
                ps.header.frame_id = parent
                self._part_pub(part).publish(ps)
            out.append(t)

        if out:
            self.tf_broadcaster.sendTransform(out)

    def _warn_if_silent(self):
        if self._seen_model:
            self._warn_timer.cancel()
            return
        self.get_logger().warn(
            f"No pose on {self.pose_topic} yet. Is Gazebo running, is the model named "
            f"'{self.model}' in moon.sdf, and is the bridge up?")

    # ------------------------------------------------------------------ ROS -> Gazebo (drive)
    def _clamp(self, v: float, lim: float) -> float:
        return max(-lim, min(lim, v))

    def _send(self, lin: float, ang: float):
        m = Twist()
        m.linear.x = self._clamp(lin, self.max_lin)
        m.angular.z = self._clamp(ang, self.max_ang)
        self.cmd_pub.publish(m)

    def _on_cmd_vel(self, msg: Twist):
        self.goal = None                      # manual command overrides a pending goal
        self._send(msg.linear.x, msg.angular.z)

    def _on_goal(self, msg: PoseStamped):
        if msg.header.frame_id not in ('', self.map_frame):
            self.get_logger().warn(f"goal_pose must be in '{self.map_frame}'. Ignored.")
            return
        self.goal = msg
        self.get_logger().info(
            f'New goal ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

    def _control_step(self):
        if self.goal is None or self.model_pose is None:
            return
        p, g = self.model_pose.pose, self.goal.pose
        dx, dy = g.position.x - p.position.x, g.position.y - p.position.y
        dist = math.hypot(dx, dy)
        if dist < self.goal_tol:
            self._send(0.0, 0.0)
            self.get_logger().info('Goal reached')
            self.goal = None
            return
        err = wrap_pi(math.atan2(dy, dx) - yaw_from_quat(p.orientation))
        # turn in place when badly misaligned, otherwise drive and steer
        lin = self.kv * dist * max(0.0, math.cos(err)) if abs(err) < math.pi / 2 else 0.0
        self._send(lin, self.ktheta * err)

    # ------------------------------------------------------------------ ROS -> Gazebo (teleport)
    def _on_set_pose(self, msg: PoseStamped):
        if msg.header.frame_id not in ('', self.map_frame):
            self.get_logger().warn(f"set_pose must be in '{self.map_frame}'. Ignored.")
            return
        if not self.set_pose_client.service_is_ready():
            self.get_logger().warn('Gazebo set_pose service not available (no service bridge running).')
            return
        req = SetEntityPose.Request()
        req.entity.name = self.model
        req.entity.type = Entity.MODEL
        req.pose = msg.pose
        req.pose.position.x -= self.offset[0]
        req.pose.position.y -= self.offset[1]
        req.pose.position.z -= self.offset[2]
        self.set_pose_client.call_async(req).add_done_callback(self._on_set_pose_done)

    def _on_set_pose_done(self, future):
        try:
            if not future.result().success:
                self.get_logger().warn(f"Gazebo refused to move '{self.model}'.")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'set_pose failed: {exc}')
