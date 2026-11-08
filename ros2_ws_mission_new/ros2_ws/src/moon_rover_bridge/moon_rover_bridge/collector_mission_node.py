"""Collector mission node: runs CollectorMission (mission_core.py) on the Gazebo collector.

Talks to the existing vehicle node (collector_rover_node) - it does not drive wheels itself:
    in : /collector_rover_node/pose        PoseStamped  (map frame)
         /hauler_mission_node/status       String(JSON) (is the hauler aligned with us?)
    out: /collector_rover_node/goal_pose   PoseStamped  (next site)
         /collector_rover_node/cmd_vel     Twist        (zero = stop at site)
         ~/status                          String(JSON) full snapshot, 2 Hz (hauler reads this)
         ~/state                           String       current state name
         ~/active_time_s, ~/idle_time_s    Float64      for rqt_plot / ros2 topic echo
         ~/planned_path                    nav_msgs/Path the hazard-free path to the next site (RViz)
Results (event CSV + summary JSON) are written to results_dir when the 24 h are up,
or as *_partial when you Ctrl-C early.

All timing uses the node clock -> run with use_sim_time:=true so it follows Gazebo.
"""

import json
import os
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Float64, String

from moon_rover_bridge.mission_core import (
    CollectorConfig, CollectorMission, CollectorState, collector_config_from_params,
    write_results)
from moon_rover_bridge.rover_base import yaw_from_quat
from moon_rover_bridge.terrain_map import HAZARD_DEFAULTS, hazard_map_from_params


class CollectorMissionNode(Node):

    def __init__(self):
        super().__init__('collector_mission_node')
        d = CollectorConfig()
        defaults = dict(HAZARD_DEFAULTS)          # terrain / field params (shared with hauler)
        defaults.update({
            'scenario_name': 'baseline',
            'random_seed': 42,
            'first_site_radius_m': d.first_site_radius_m,
            'next_site_radius_m': d.next_site_radius_m,
            'field_half_width_m': d.field_half_width_m,
            'boundary_margin_m': d.boundary_margin_m,
            'collection_duration_s': d.collection_duration_s,
            'mission_duration_s': d.mission_duration_s,
            'exchange_buffer_s': d.exchange_buffer_s,
            'exchange_every_n_sites': d.exchange_every_n_sites,
            'arrival_tolerance_m': d.arrival_tolerance_m,
            'waypoint_tolerance_m': d.waypoint_tolerance_m,
            'max_detour_factor': d.max_detour_factor,
            'results_dir': '',
            'vehicle_node': 'collector_rover_node',
            'hauler_status_topic': '/hauler_mission_node/status',
            'map_frame': 'map',
            'goal_republish_s': 2.0,
        })
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        p = {k: self.get_parameter(k).value for k in defaults}

        hazard_map = hazard_map_from_params(p)
        self.mission = CollectorMission(collector_config_from_params(p), hazard_map)
        self.scenario = p['scenario_name']
        self.map_frame = p['map_frame']
        self.republish_s = float(p['goal_republish_s'])
        self.results_dir = p['results_dir'] or os.path.join(
            os.path.expanduser('~'), '.ros', 'mission_results')
        self.tag = f"{self.scenario}_{time.strftime('%Y%m%d_%H%M%S')}"

        veh = p['vehicle_node']
        self.goal_pub = self.create_publisher(PoseStamped, f'/{veh}/goal_pose', 10)
        self.stop_pub = self.create_publisher(Twist, f'/{veh}/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.state_pub = self.create_publisher(String, '~/state', 10)
        self.active_pub = self.create_publisher(Float64, '~/active_time_s', 10)
        self.idle_pub = self.create_publisher(Float64, '~/idle_time_s', 10)
        self.path_pub = self.create_publisher(Path, '~/planned_path', 10)
        self._last_path = None

        self.pose = None
        self.hauler_aligned_id = None
        self.create_subscription(PoseStamped, f'/{veh}/pose', self._on_pose, 10)
        self.create_subscription(String, p['hauler_status_topic'], self._on_hauler, 10)

        self._sent_goal = None
        self._last_goal_pub = -1e9
        self._n_logged = 0
        self._results_written = False

        self.create_timer(0.1, self._tick)
        self.create_timer(0.5, self._publish_status)
        self.get_logger().info(
            f"Scenario '{self.scenario}': exchange buffer {p['exchange_buffer_s']} s, "
            f"exchange every {p['exchange_every_n_sites']} site(s), field +/-"
            f"{p['field_half_width_m']} m. Waiting for pose on /{veh}/pose and sim time...")
        if hazard_map is None:
            self.get_logger().warn('No hazard_map_file set: driving STRAIGHT lines (no hazard avoidance).')
        else:
            self.get_logger().info(
                f"Hazard map {p['hazard_map_file']}: z in [{p['hazard_z_min']}, {p['hazard_z_max']}] m, "
                f"slope <= {p['hazard_max_slope_deg']} deg, margin {p['hazard_inflation_m']} m -> "
                f'{100 * hazard_map.free_fraction():.1f} % of the map is drivable')

    # ------------------------------------------------------------------ inputs
    def _on_pose(self, msg: PoseStamped):
        self.pose = msg

    def _on_hauler(self, msg: String):
        try:
            self.hauler_aligned_id = json.loads(msg.data).get('aligned_request_id')
        except (ValueError, AttributeError):
            self.get_logger().warn('Bad hauler status JSON ignored')

    # ------------------------------------------------------------------ main loop
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _tick(self):
        now = self._now()
        if self.pose is None or now <= 0.0:          # no pose yet / sim still paused at t=0
            return
        p = self.pose.pose
        m = self.mission
        m.update(now, p.position.x, p.position.y, yaw_from_quat(p.orientation),
                 self.hauler_aligned_id)
        self._log_new_events()

        if m.goal is not None:
            if m.goal != self._sent_goal or now - self._last_goal_pub >= self.republish_s:
                self._send_goal(*m.goal, now)
        elif self._sent_goal is not None:            # just arrived / mission done -> stop
            self.stop_pub.publish(Twist())
            self._sent_goal = None

        if m.waypoints != self._last_path:
            self._publish_path(p.position.x, p.position.y)

        if m.state == CollectorState.DONE and not self._results_written:
            self._write_results(partial=False)

    def _send_goal(self, x: float, y: float, now: float):
        g = PoseStamped()
        g.header.frame_id = self.map_frame
        g.header.stamp = self.get_clock().now().to_msg()
        g.pose.position.x, g.pose.position.y = float(x), float(y)
        g.pose.orientation.w = 1.0
        self.goal_pub.publish(g)
        self._sent_goal = (x, y)
        self._last_goal_pub = now

    # ------------------------------------------------------------------ outputs
    def _publish_path(self, x: float, y: float):
        self._last_path = list(self.mission.waypoints)
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()
        points = ([(x, y)] + self._last_path) if self._last_path else []   # empty = no path
        for px, py in points:
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            ps.pose.position.x, ps.pose.position.y = float(px), float(py)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_pub.publish(path)

    def _publish_status(self):
        s = self.mission.status()
        self.status_pub.publish(String(data=json.dumps(s)))
        self.state_pub.publish(String(data=s['state']))
        self.active_pub.publish(Float64(data=float(s['active_s'])))
        self.idle_pub.publish(Float64(data=float(s['idle_s'])))

    def _log_new_events(self):
        for e in self.mission.events[self._n_logged:]:
            self.get_logger().info(
                f'[t={e.mission_time_s / 60:7.1f} min] {e.event:<30} -> {e.state:<18} {e.detail}')
        self._n_logged = len(self.mission.events)

    def _write_results(self, partial: bool):
        tag = self.tag + ('_partial' if partial else '')
        csv_path, json_path = write_results(self.mission, self.results_dir, tag)
        self._results_written = True
        self.get_logger().info(f'{self.mission.summary_line()}\nResults: {csv_path}\n         {json_path}')


def main(args=None):
    rclpy.init(args=args)
    node = CollectorMissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not node._results_written and node.mission.t_start is not None:
            node._write_results(partial=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
