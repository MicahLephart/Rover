# Collector / hauler mission — how it works and how to review it

```bash
cd ~/ros2_ws && colcon build --symlink-install && source install/setup.bash
```

## Step 1 — fix "wheels spin but the rover doesn't move" (drive_check)

The DiffDrive plugin already gives you odometry (`/model/<model>/odometry`, bridged), but that
odometry is computed **from wheel rotation**, so it says "moving" even when the wheels just slip.
`drive_check` compares it against the true body pose from the PosePublisher and tells you which
problem you have:

```bash
ros2 launch gazebo_worlds moon_launch.xml          # world + vehicle nodes only; press play
ros2 run moon_rover_bridge drive_check --ros-args -p use_sim_time:=true
ros2 run moon_rover_bridge drive_check --ros-args -p use_sim_time:=true -p model:=hauler_rover
```

| Verdict | Meaning | Fix |
|---|---|---|
| no odometry | DiffDrive not running | joint names in moon.sdf must match the model's wheel joints |
| wheels not reaching speed | something holds the wheels | duplicate drive plugin from the Fuel model, joint limits/damping |
| wheels turn, body still, **sank/tilted** | rover spawned into the terrain | raise the `<pose>` z of the rover |
| wheels turn, body still | no traction | chassis touching ground, or terrain friction ~0 → add `<surface><friction>` |
| body moves | drive is fine | it also prints the real wheel radius / effective wheel separation to put in moon.sdf |

## Step 2 — hazard map from terrain height

Build the z grid once from the terrain model (reads its `model.sdf` collision: mesh **or** heightmap):

```bash
ros2 run moon_rover_bridge build_hazard_map <path>/lunar_surface0 \
    -o ~/ros2_ws/src/moon_rover_bridge/config/maps/lunar_surface0.npz
# or a mesh you exported from Blender (Y-up is detected automatically):
ros2 run moon_rover_bridge build_hazard_map terrain.obj -o .../config/maps/lunar_surface0.npz
```

It prints z percentiles and saves `lunar_surface0.png` (height / slope / blocked cells). Re-run with
`--z-min --z-max --max-slope` until the black areas in the PNG are the places you want avoided, then
put those numbers in the scenario's `/**` block:

```yaml
/**:
  ros__parameters:
    hazard_map_file: "lunar_surface0.npz"
    hazard_z_min: -0.6           # below = pit / crater floor
    hazard_z_max: 0.8            # above = mound / rock
    hazard_max_slope_deg: 15.0   # steeper = crater wall
    hazard_inflation_m: 1.0      # keep this far away from any hazard
```

`colcon build` again so the .npz is installed. `.obj`/`.stl` work out of the box; `.dae` needs
`pip install trimesh pycollada` (or export OBJ from Blender).

With a map:
* a site is only picked if it is on safe terrain **and** reachable by a safe path no longer than
  `max_detour_factor` (2×) the straight distance (the 100 m / 50 m rule still holds exactly);
* both rovers drive an A* path around hazards as a list of waypoints (`/collector_mission_node/planned_path`
  shows it in RViz); the hauler also keeps 1.2 m away from the parked collector and picks the other
  side / behind / in front if the left side is hazardous.

Without a map (`hazard_map_file: ""`) everything drives straight lines, as before.

## Step 3 — run missions and scenarios

```bash
ros2 launch gazebo_worlds mission_launch.xml                         # baseline (Gazebo, press play)
ros2 launch gazebo_worlds mission_launch.xml scenario:=long_buffer
ros2 run moon_rover_bridge scenario_runner baseline long_buffer --plot   # 24 h each in seconds
cd src/moon_rover_bridge && python3 -m pytest test/test_mission_core.py -v
```

Timers: `ros2 topic echo /collector_mission_node/status`, or rqt_plot
`/collector_mission_node/active_time_s/data` and `/idle_time_s/data`. Results (event CSV + summary JSON)
go to `~/.ros/mission_results/` (Gazebo) or `mission_results_offline/` (runner, plus `*_routes.png`).

New scenario: copy `config/scenarios/baseline.yaml`, change `scenario_name` and e.g. `exchange_buffer_s`,
`colcon build`, run with `scenario:=<name>`. Keep decimal points on floats (`600.0`, not `600`).

## What the collector does

1. First site: random point with `sqrt(x² + y²) = 100 m`, inside ±75 m (±80 field − 5 m margin).
   **Inside a ±75 m square a 100 m circle only exists near the 4 corners** (|x|, |y| ≥ ~66 m), so the
   first site is always in a corner.
2. Drives there at 0.27 m/s → **idle (travel)**.
3. Collects 900 s → **active**. Dumps into in-bay storage (logged only).
4. Next site exactly 50 m from where it is, in bounds (and safe/reachable with a map). Repeat for 24 h sim time.

Exchange: every site the hauler comes over, parks beside it, and the collector pauses for
`exchange_buffer_s` → **idle (exchange)**, then finishes its 15 min. If it finishes first it waits →
**idle (wait for hauler)**. The hauler returns to base before taking the next request.
Active + idle always = 24 h.

## Review order

| File | What |
|---|---|
| `config/scenarios/baseline.yaml` | all numbers |
| `moon_rover_bridge/mission_core.py` | all mission logic, no ROS (`CollectorMission._transition`, `HaulerMission.update`) |
| `moon_rover_bridge/terrain_map.py` | hazard rules (`configure`) and A* planner (`plan`), no ROS |
| `test/test_mission_core.py` | each requirement as a test, incl. "never drives on a hazard" |
| `moon_rover_bridge/build_hazard_map.py` | terrain model → z grid |
| `moon_rover_bridge/drive_check.py` | the slip / stuck diagnosis |
| `moon_rover_bridge/*_mission_node.py` | thin ROS wrappers |
| `moon_rover_bridge/scenario_runner.py` | offline 24 h runs |

The existing vehicle nodes (`rover_base.py` etc.) are unchanged; mission nodes send them `goal_pose`.

## Known limits
* The map is static: it covers terrain in the mesh, not rocks spawned as separate models. Add those as
  part of the mesh, or tell me and I'll add a list of keep-out circles.
* Heightmap images: I assume the top image row = +y. Check the preview PNG against Gazebo once.
* Hauler speed assumed 0.27 m/s; `exchange_buffer_s: 120` is a placeholder.
* 24 h in Gazebo at real-time factor 1 = 24 h of wall time → use the offline runner for comparisons.
