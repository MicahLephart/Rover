"""drive_check: why do the wheels turn but the rover not move?

Drives one rover through a short fixed test and compares two things Gazebo already publishes:
    odometry  /model/<model>/odometry   from the DiffDrive plugin = what the WHEELS did
                                        (it is computed from wheel rotation, so it says "moving"
                                         even when the wheels are just slipping)
    truth     /model/<model>/pose       from the PosePublisher   = where the BODY actually is

Run it with the world + vehicle nodes only (NOT the mission nodes, they'd fight over cmd_vel):
    ros2 launch gazebo_worlds moon_launch.xml           # then press play in Gazebo
    ros2 run moon_rover_bridge drive_check --ros-args -p use_sim_time:=true
    ros2 run moon_rover_bridge drive_check --ros-args -p use_sim_time:=true -p model:=hauler_rover

Test (sim time): settle 3 s -> straight 15 s at 0.27 m/s -> stop 2 s -> turn 10 s at 0.5 rad/s -> stop.
At the end it prints the numbers and a verdict.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage

from moon_rover_bridge.rover_base import wrap_pi, yaw_from_quat


def roll_pitch(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    return roll, pitch


class Track:
    """Accumulates distance travelled and yaw turned from a stream of poses."""

    def __init__(self):
        self.n = 0
        self.x = self.y = self.z = self.yaw = None
        self.dist = 0.0
        self.turn = 0.0
        self.max_tilt = 0.0
        self.z_min = float('inf')

    def add(self, x, y, z, q, counting: bool):
        yaw = yaw_from_quat(q)
        if self.x is not None and counting:
            self.dist += math.hypot(x - self.x, y - self.y)
            self.turn += wrap_pi(yaw - self.yaw)
        r, p = roll_pitch(q)
        self.max_tilt = max(self.max_tilt, abs(r), abs(p))
        self.z_min = min(self.z_min, z)
        self.x, self.y, self.z, self.yaw = x, y, z, yaw
        self.n += 1

    def snapshot(self):
        return self.dist, self.turn, self.z


class DriveCheck(Node):
    # (name, duration_s, linear, angular)
    PHASES = [('settle', 3.0, 0.0, 0.0), ('straight', 15.0, None, 0.0), ('stop1', 2.0, 0.0, 0.0),
              ('turn', 10.0, 0.0, None), ('stop2', 2.0, 0.0, 0.0)]

    def __init__(self):
        super().__init__('drive_check')
        self.declare_parameter('model', 'collector_rover')
        self.declare_parameter('speed', 0.27)
        self.declare_parameter('turn_rate', 0.5)
        self.declare_parameter('wheel_radius', 0.31)        # as set in moon.sdf DiffDrive
        self.declare_parameter('wheel_separation', 0.275)   # as set in moon.sdf DiffDrive
        g = self.get_parameter
        self.model = g('model').value
        self.speed = float(g('speed').value)
        self.rate = float(g('turn_rate').value)
        self.r_cfg = float(g('wheel_radius').value)
        self.sep_cfg = float(g('wheel_separation').value)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.cmd_pub = self.create_publisher(Twist, f'/model/{self.model}/cmd_vel', 10)
        self.create_subscription(Odometry, f'/model/{self.model}/odometry', self._on_odom, qos)
        self.create_subscription(TFMessage, f'/model/{self.model}/pose', self._on_truth, qos)

        self.odom, self.truth = Track(), Track()
        self.phase_i = -1
        self.phase_t0 = None
        self.marks = {}                     # phase name -> (odom snapshot, truth snapshot) at start
        self.finished = False
        self.create_timer(0.1, self._tick)
        self.get_logger().info(f"drive_check on '{self.model}': waiting for sim time + topics...")

    # ------------------------------------------------------------------ inputs
    def _phase(self):
        return self.PHASES[self.phase_i][0] if 0 <= self.phase_i < len(self.PHASES) else ''

    def _on_odom(self, m: Odometry):
        p = m.pose.pose
        self.odom.add(p.position.x, p.position.y, p.position.z, p.orientation,
                      self._phase() in ('straight', 'stop1', 'turn', 'stop2'))

    def _on_truth(self, m: TFMessage):
        for tf in m.transforms:
            parts = [p for p in tf.child_frame_id.replace('/', '::').split('::') if p]
            if parts == [self.model]:                # the model itself, not one of its links
                t = tf.transform.translation
                self.truth.add(t.x, t.y, t.z, tf.transform.rotation,
                               self._phase() in ('straight', 'stop1', 'turn', 'stop2'))

    # ------------------------------------------------------------------ test sequence
    def _tick(self):
        if self.finished:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now <= 0.0:
            return
        if self.phase_i < 0:
            if self.truth.n == 0:
                return                       # wait for the first true pose
            self._next(now)
        name, dur, lin, ang = self.PHASES[self.phase_i]
        if now - self.phase_t0 >= dur:
            if self.phase_i + 1 >= len(self.PHASES):
                self.cmd_pub.publish(Twist())
                self.finished = True
                self._report()
                return
            self._next(now)
            name, dur, lin, ang = self.PHASES[self.phase_i]
        t = Twist()
        t.linear.x = self.speed if lin is None else lin
        t.angular.z = self.rate if ang is None else ang
        self.cmd_pub.publish(t)

    def _next(self, now):
        self.phase_i += 1
        self.phase_t0 = now
        name = self.PHASES[self.phase_i][0]
        self.marks[name] = (self.odom.snapshot(), self.truth.snapshot())
        self.get_logger().info(f'phase: {name}')

    # ------------------------------------------------------------------ report
    def _delta(self, a, b, track):
        """Change in (dist, turn) between the starts of phases a and b."""
        ia = 0 if track == 'odom' else 1
        da, ta, _ = self.marks[a][ia]
        db, tb, _ = self.marks[b][ia]
        return db - da, tb - ta

    def _report(self):
        L = []
        o, tr = self.odom, self.truth
        cmd_d = self.speed * self.PHASES[1][1]
        cmd_t = self.rate * self.PHASES[3][1]
        L.append(f'==== drive_check: {self.model} ====')
        L.append(f'messages received: odometry {o.n}, true pose {tr.n}')
        if tr.n == 0:
            L.append('VERDICT: no true pose -> PosePublisher plugin or the ros_gz_bridge pose entry is missing.')
            return self._print(L)
        z0 = self.marks['settle'][1][2]
        z1 = self.marks['straight'][1][2]
        L.append(f'settle 3 s: true z {z0:.3f} -> {z1:.3f} m (sank {z0 - z1:+.3f} m), '
                 f'lowest z {tr.z_min:.3f}, max tilt {math.degrees(tr.max_tilt):.1f} deg')
        if o.n == 0:
            L.append('VERDICT: no odometry -> the DiffDrive plugin is not running for this model '
                     '(check <left_joint>/<right_joint> names match the model, and the odometry bridge).')
            return self._print(L)
        od, _ = self._delta('straight', 'stop1', 'odom')
        td, _ = self._delta('straight', 'stop1', 'truth')
        _, ot = self._delta('turn', 'stop2', 'odom')
        _, tt = self._delta('turn', 'stop2', 'truth')
        L.append(f'straight 15 s: commanded {cmd_d:.2f} m | wheels(odom) {od:.2f} m | body(truth) {td:.2f} m')
        L.append(f'turn 10 s:     commanded {math.degrees(cmd_t):.0f} deg | wheels(odom) '
                 f'{math.degrees(ot):.0f} deg | body(truth) {math.degrees(tt):.0f} deg')
        slip = 1.0 - td / od if od > 1e-6 else float('nan')
        L.append(f'linear slip = 1 - truth/odom = {100 * slip:.0f} %')

        if od < 0.5 * cmd_d:
            L.append('VERDICT: the wheels are NOT reaching the commanded speed. Something holds them back: '
                     'wrong joint names, a second drive plugin from the Fuel model fighting this one, or '
                     'wheel joints with limits/damping. Check `gz topic -e -t /model/<model>/cmd_vel`.')
        elif td < 0.2 * od:
            if (z0 - z1) > 0.05 or tr.max_tilt > math.radians(20):
                L.append('VERDICT: wheels turn, body does not move, and it sank/tilted -> the rover is '
                         'STUCK IN THE TERRAIN. Spawn it higher (e.g. <pose> z +0.5 m above the surface '
                         'height at the spawn point) and check the terrain collision mesh.')
            else:
                L.append('VERDICT: wheels turn but the body does not move -> NO TRACTION. Either the '
                         'chassis is resting on the ground (wheels not carrying the weight: check the '
                         'chassis collision box vs wheel radius) or friction is ~0 (add <surface><friction>'
                         '<ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface> to the terrain '
                         'collision; low lunar gravity makes this worse).')
        else:
            ratio = td / od
            L.append(f'Body moves. truth/odom = {ratio:.2f}. If there is little slip this means the real '
                     f'wheel radius is ~{self.r_cfg * ratio:.3f} m (moon.sdf says {self.r_cfg}).')
            if abs(tt) > 1e-3 and abs(ot) > 1e-3:
                L.append(f'Turning: truth/odom = {tt / ot:.2f} -> effective wheel_separation '
                         f'~{self.sep_cfg * ot / tt:.3f} m (moon.sdf says {self.sep_cfg}). Skid-steer '
                         f'rovers always need a larger "effective" value than the real track width.')
            L.append('VERDICT: drive works. If missions still stall, the goal controller is the issue '
                     '(check /<vehicle>_node/goal_pose arrives and max_linear > 0).')
        self._print(L)

    def _print(self, lines):
        for ln in lines:
            self.get_logger().info(ln)
        self.finished = True


def main(args=None):
    rclpy.init(args=args)
    node = DriveCheck()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
