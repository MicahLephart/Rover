"""Offline scenario runner: the full 24 h mission in a few seconds, no Gazebo needed.

Uses the SAME mission_core state machines as the ROS nodes, with simple kinematic rovers
(turn in place at max_angular, then drive straight at max_linear). Use it to compare
scenarios (e.g. different exchange buffers) quickly; use Gazebo for the visual run.

    ros2 run moon_rover_bridge scenario_runner baseline long_buffer
    ros2 run moon_rover_bridge scenario_runner path/to/my_scenario.yaml --seeds 10
    ros2 run moon_rover_bridge scenario_runner baseline --plot      # PNG of the routes over the map
    python3 -m moon_rover_bridge.scenario_runner config/scenarios/*.yaml     (without ROS)
"""

import argparse
import math
import os
import statistics
import sys
from typing import Dict, List

import yaml

from moon_rover_bridge.mission_core import (
    CollectorMission, CollectorState, HaulerMission, collector_config_from_params,
    hauler_config_from_params, wrap_pi, write_results)
from moon_rover_bridge.terrain_map import hazard_map_from_params


class KinematicRover:
    """Point rover: rotate toward goal at w_max, then drive straight at v_max."""

    def __init__(self, x, y, yaw, v_max, w_max):
        self.x, self.y, self.yaw = x, y, yaw
        self.v, self.w = v_max, w_max
        self.distance_m = 0.0

    def drive_to(self, gx, gy, dt):
        dx, dy = gx - self.x, gy - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return
        err = wrap_pi(math.atan2(dy, dx) - self.yaw)
        if abs(err) > 0.05:                          # turn in place first
            self.rotate(err, dt)
            return
        self.yaw += err
        step = min(self.v * dt, dist)
        self.x += step * dx / dist
        self.y += step * dy / dist
        self.distance_m += step

    def rotate(self, err, dt):
        self.yaw = wrap_pi(self.yaw + math.copysign(min(abs(err), self.w * dt), err))


def _params(doc: Dict, node: str) -> Dict:
    """Params a node would get from this file: the '/**' block, then its own block on top."""
    out = dict(((doc.get('/**') or {}).get('ros__parameters') or {}))
    out.update((doc.get(node) or {}).get('ros__parameters', {}) or {})
    return out


def resolve_scenario(arg: str) -> str:
    if os.path.isfile(arg):
        return arg
    try:  # look in the installed package share dir
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory('moon_rover_bridge'),
                         'config', 'scenarios', f'{arg}.yaml')
        if os.path.isfile(p):
            return p
    except Exception:  # noqa: BLE001  (no ROS environment)
        pass
    here = os.path.join(os.path.dirname(__file__), '..', 'config', 'scenarios', f'{arg}.yaml')
    if os.path.isfile(here):
        return os.path.abspath(here)
    raise FileNotFoundError(f'Scenario "{arg}" not found (give a name or a .yaml path)')


def run_scenario(path: str, seed_override=None, dt: float = 0.1):
    with open(path) as f:
        doc = yaml.safe_load(f)
    cp = dict(_params(doc, 'collector_mission_node'))
    if seed_override is not None:
        cp['random_seed'] = seed_override
    ccfg = collector_config_from_params(cp)
    hcfg = hauler_config_from_params(_params(doc, 'hauler_mission_node'))
    cveh = _params(doc, 'collector_rover_node')
    hveh = _params(doc, 'hauler_rover_node')

    hazard_map = hazard_map_from_params(cp)
    col = CollectorMission(ccfg, hazard_map)
    hau = HaulerMission(hcfg, hazard_map)
    crov = KinematicRover(0.0, 0.0, 0.0, cveh.get('max_linear', 0.27), cveh.get('max_angular', 1.0))
    hrov = KinematicRover(hcfg.base_x, hcfg.base_y, 0.0,
                          hveh.get('max_linear', 0.27), hveh.get('max_angular', 1.0))

    t = 0.0
    hauler_aligned = None
    col_track, haul_track = [], []
    while col.state != CollectorState.DONE:
        if round(t / dt) % max(1, int(2.0 / dt)) == 0:       # sample tracks every 2 s
            col_track.append((crov.x, crov.y))
            haul_track.append((hrov.x, hrov.y))
        col.update(t, crov.x, crov.y, crov.yaw, hauler_aligned)
        if col.goal is not None:
            crov.drive_to(col.goal[0], col.goal[1], dt)

        cmd = hau.update(hrov.x, hrov.y, hrov.yaw, col.status())
        if cmd.kind == 'goal':
            hrov.drive_to(cmd.x, cmd.y, dt)
        elif cmd.kind == 'rotate':
            hrov.rotate(cmd.yaw_error, dt)
        hauler_aligned = hau.aligned_id()
        t += dt

    name = cp.get('scenario_name', os.path.splitext(os.path.basename(path))[0])
    crov.track, hrov.track = col_track, haul_track
    return name, col, hau, crov, hrov


def plot_run(path_png, col, crov, hrov):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('(matplotlib not installed: no plot)')
        return
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    m = col.map
    if m is not None:
        import numpy as np
        shade = np.where(m.hazard, 2, np.where(m.blocked, 1, 0))
        ax.imshow(shade, origin='lower', cmap='Greys', vmin=0, vmax=2,
                  extent=[m.ox, m.ox + m.w * m.res, m.oy, m.oy + m.h * m.res])
    hw = col.cfg.field_half_width_m
    ax.add_patch(plt.Rectangle((-hw, -hw), 2 * hw, 2 * hw, fill=False, ec='k', lw=1))
    if hrov.track:
        ax.plot(*zip(*hrov.track), color='tab:orange', lw=0.6, alpha=0.6, label='hauler')
    if crov.track:
        ax.plot(*zip(*crov.track), color='tab:blue', lw=1.0, label='collector')
    sites = [(e.x, e.y) for e in col.events if e.event == 'ARRIVED_START_COLLECTION']
    if sites:
        ax.scatter(*zip(*sites), s=14, color='tab:red', zorder=3, label=f'{len(sites)} sites')
    ax.set_aspect('equal')
    ax.set_xlim(-hw - 2, hw + 2)
    ax.set_ylim(-hw - 2, hw + 2)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(col.summary_line().split(' | sites')[0], fontsize=9)
    fig.tight_layout()
    fig.savefig(path_png, dpi=110)
    plt.close(fig)
    print(f'plot: {path_png}')


def main(argv: List[str] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('scenarios', nargs='+', help='scenario names or .yaml paths')
    ap.add_argument('--seeds', type=int, default=1,
                    help='run each scenario with N seeds (seed, seed+1, ...) and average')
    ap.add_argument('--dt', type=float, default=0.1, help='time step [s] (ROS nodes tick at 0.1)')
    ap.add_argument('--out', default='mission_results_offline', help='results folder')
    ap.add_argument('--plot', action='store_true', help='save a PNG of each run\'s routes')
    a = ap.parse_args(argv)

    rows = []
    for sc in a.scenarios:
        path = resolve_scenario(sc)
        with open(path) as f:
            base_seed = _params(yaml.safe_load(f), 'collector_mission_node').get('random_seed', 42)
        base_seed = 42 if base_seed is None or int(base_seed) < 0 else int(base_seed)
        runs = []
        for k in range(a.seeds):
            name, col, hau, crov, hrov = run_scenario(path, base_seed + k, a.dt)
            tag = f'{name}_seed{base_seed + k}'
            write_results(col, a.out, tag)
            if a.plot:
                plot_run(os.path.join(a.out, f'{tag}_routes.png'), col, crov, hrov)
            runs.append((col, crov, hrov))
        m = lambda f: statistics.mean(f(r) for r in runs)  # noqa: E731
        rows.append((name, a.seeds,
                     m(lambda r: r[0].timers.active_s) / 3600,
                     m(lambda r: r[0].timers.idle_s) / 3600,
                     m(lambda r: r[0].timers.idle_travel_s) / 3600,
                     m(lambda r: r[0].timers.idle_exchange_s) / 3600,
                     m(lambda r: r[0].timers.idle_wait_hauler_s) / 3600,
                     m(lambda r: r[0].sites_reached),
                     m(lambda r: r[0].exchanges_completed),
                     m(lambda r: r[1].distance_m) / 1000,
                     m(lambda r: r[2].distance_m) / 1000))

    hdr = ('scenario', 'runs', 'active_h', 'idle_h', 'travel_h', 'exch_h', 'wait_h',
           'sites', 'exchanges', 'col_km', 'haul_km')
    print(('{:<16}' + '{:>10}' * (len(hdr) - 1)).format(*hdr))
    for r in rows:
        print(('{:<16}{:>10d}' + '{:>10.2f}' * 5 + '{:>10.1f}' * 2 + '{:>10.2f}' * 2).format(*r))
    print(f'\nEvent logs + summaries written to: {os.path.abspath(a.out)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
