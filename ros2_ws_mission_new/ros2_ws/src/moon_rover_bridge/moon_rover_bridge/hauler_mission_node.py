"""Hauler mission node: runs HaulerMission (mission_core.py) on the Gazebo hauler.

When the collector requests an exchange, the hauler drives from base to a spot beside the
collector, turns to the collector's heading (aligned), waits while the collector runs its
exchange buffer, then returns to base. The container swap/unload is visual only.

    in : /hauler_rover_node/pose            PoseStamped
         /collector_mission_node/status     String(JSON)
    out: /hauler_rover_node/goal_pose       PoseStamped  (drive to a point)
         /hauler_rover_node/cmd_vel         Twist        (turn in place to align / stop)
         ~/status                           String(JSON) 2 Hz (collector reads aligned_request_id)
"""

import json

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String

from moon_rover_bridge.mission_core import HaulerConfig, HaulerMission, hauler_config_from_params
from moon_rover_bridge.rover_base import yaw_from_quat
from moon_rover_bridge.terrain_map import HAZARD_DEFAULTS, hazard_map_from_params


class HaulerMissionNode(Node):

    def __init__(self):
        super().__init__('hauler_mission_node')
        d = HaulerConfig()
        defaults = dict(HAZARD_DEFAULTS)          # same terrain/hazard params as the collector
        defaults.update({
            'base_x': d.base_x,
            'base_y': d.base_y,
            'align_offset_m': d.align_offset_m,
            'position_tolerance_m': d.position_tolerance_m,
            'yaw_tolerance_rad': d.yaw_tolerance_rad,
            'waypoint_tolerance_m': d.waypoint_tolerance_m,
            'collector_keepout_m': d.collector_keepout_m,
            'yaw_gain': 1.0,                 # turn-in-place P gain (vehicle node clamps it)
            'vehicle_node': 'hauler_rover_node',
            'collector_status_topic': '/collector_mission_node/status',
            'map_frame': 'map',
            'goal_republish_s': 2.0,
        })
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        p = {k: self.get_parameter(k).value for k in defaults}

        hazard_map = hazard_map_from_params(p)
        self.mission = HaulerMission(hauler_config_from_params(p), hazard_map)
        if hazard_map is not None and not hazard_map.is_free(p['base_x'], p['base_y']):
            self.get_logger().warn('Hauler base is inside a hazard / safety margin on the map!')
        self.yaw_gain = float(p['yaw_gain'])
        self.map_frame = p['map_frame']
        self.republish_s = float(p['goal_republish_s'])

        veh = p['vehicle_node']
        self.goal_pub = self.create_publisher(PoseStamped, f'/{veh}/goal_pose', 10)
        self.cmd_pub = self.create_publisher(Twist, f'/{veh}/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '~/status', 10)

        self.pose = None
        self.collector = None
        self.create_subscription(PoseStamped, f'/{veh}/pose', self._on_pose, 10)
        self.create_subscription(String, p['collector_status_topic'], self._on_collector, 10)

        self._sent_goal = None
        self._last_goal_pub = -1e9
        self._n_logged = 0
        self.create_timer(0.1, self._tick)
        self.create_timer(0.5, self._publish_status)
        self.get_logger().info(f'Hauler ready at base ({p["base_x"]}, {p["base_y"]})')

    def _on_pose(self, msg: PoseStamped):
        self.pose = msg

    def _on_collector(self, msg: String):
        try:
            self.collector = json.loads(msg.data)
        except ValueError:
            self.get_logger().warn('Bad collector status JSON ignored')

    def _tick(self):
        if self.pose is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        p = self.pose.pose
        cmd = self.mission.update(p.position.x, p.position.y, yaw_from_quat(p.orientation),
                                  self.collector)
        for state, detail in self.mission.events[self._n_logged:]:
            self.get_logger().info(f'hauler -> {state:<13} {detail}')
        self._n_logged = len(self.mission.events)

        if cmd.kind == 'goal':
            goal = (cmd.x, cmd.y)
            if goal != self._sent_goal or now - self._last_goal_pub >= self.republish_s:
                g = PoseStamped()
                g.header.frame_id = self.map_frame
                g.header.stamp = self.get_clock().now().to_msg()
                g.pose.position.x, g.pose.position.y = float(cmd.x), float(cmd.y)
                g.pose.orientation.w = 1.0
                self.goal_pub.publish(g)
                self._sent_goal, self._last_goal_pub = goal, now
        elif cmd.kind == 'rotate':
            t = Twist()
            t.angular.z = self.yaw_gain * cmd.yaw_error
            self.cmd_pub.publish(t)                  # also cancels any goal in the vehicle node
            self._sent_goal = None
        elif cmd.kind == 'stop':
            self.cmd_pub.publish(Twist())
            self._sent_goal = None

    def _publish_status(self):
        self.status_pub.publish(String(data=json.dumps(self.mission.status())))


def main(args=None):
    rclpy.init(args=args)
    node = HaulerMissionNode()
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
