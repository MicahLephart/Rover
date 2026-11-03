# moon_rover_bridge

Two ROS 2 vehicle nodes (`collector_rover_node`, `hauler_rover_node`) that mirror and drive the
rovers in `gazebo_worlds/worlds/moon.sdf`.

## What changed in the workspace
- `moon.sdf`: the original rover is now `collector_rover`; a second `hauler_rover` (PLACEHOLDER, same
  Explorer model) was added. Each has its own DiffDrive topic (`/model/<name>/cmd_vel`) and a
  PosePublisher (`/model/<name>/pose`, model + all links).
- `moon.yaml`: bridges `/clock`, and per rover `cmd_vel` (ROS->GZ), `odometry` and `pose` (GZ->ROS).
- `moon_launch.xml`: also starts the two nodes and an optional `set_pose` service bridge.
- `gazebo_worlds/CMakeLists.txt`: installs launch/worlds/config (it had no install rule).

## Run
```bash
cd ~/ros2_ws && colcon build --packages-select moon_rover_bridge gazebo_worlds
source install/setup.bash
ros2 launch gazebo_worlds moon_launch.xml
```

## Try it
```bash
# drive the collector (clamped to 0.5 m/s, 1.0 rad/s like the plugin)
ros2 topic pub -r 5 /collector_rover_node/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}"
# drive the hauler to a point (same law as demo.py)
ros2 topic pub --once /hauler_rover_node/goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: 5.0, y: 2.0}}}"
# rover location and parts in ROS
ros2 topic echo /collector_rover_node/pose
ros2 run tf2_ros tf2_echo map collector_rover
ros2 topic list | grep parts
```

## Mapping specific rover parts
By default every link of the Gazebo model is tracked (`~/parts/<link>` plus TF `<model>/<link>`).
To track only certain parts, set the `parts` parameter, e.g. in the launch XML:
`<param name="parts" value="[scoop_link, chassis]" />`. Find real link names with
`gz topic -e -n 1 -t /model/collector_rover/pose`.

## Notes
- Untested against a live Gazebo session; verify with the commands above.
- If part TFs look misplaced, set `parts_relative_to_model` to false.
- If the world is renamed, change `world_name` in the launch file and the set_pose bridge arg.
