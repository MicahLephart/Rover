"""Mission logic for the collector + hauler rovers.  NO ROS IMPORTS IN THIS FILE.

Everything that decides *what* the rovers do lives here, so it can be unit-tested and run
offline (scenario_runner.py) without Gazebo. The ROS nodes are thin wrappers that feed in
time + pose and send the resulting goals to the vehicle nodes.

Collector cycle (repeats until mission_duration_s of sim time has passed):

    TRAVELING ──arrive──> COLLECTING ──15 min done──> (dump load, visual only) ──> TRAVELING
                            │    ▲                         │
          hauler aligned    │    │ buffer done             │ exchange still due
                            ▼    │                         ▼
                          EXCHANGING <──hauler aligned── WAITING_FOR_HAULER

    * First site:  random point exactly first_site_radius_m (100 m) from the field centre.
    * Next sites:  random point exactly next_site_radius_m (50 m) from where the rover is,
                   resampled if |x| or |y| would exceed field_half_width_m - boundary_margin_m.
    * With a hazard map (terrain_map.py) a site must also be on safe terrain and reachable by
      a hazard-free path no longer than max_detour_factor x the straight line. The rover then
      drives that path as a list of waypoints instead of a straight line.
    * Every exchange_every_n_sites-th site the collector requests a container exchange.
      The hauler drives over, parks beside it facing the same way, and the collector then
      pauses collection for exchange_buffer_s (the swap itself is visual only).
      If the 15 min finish before the hauler gets there, the collector waits at the site.

Timers (sim seconds):
    active  = COLLECTING
    idle    = TRAVELING + EXCHANGING + WAITING_FOR_HAULER   (broken down separately too)
    active + idle == mission time, always.
"""

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# ============================================================================ site generator
class SiteGenerator:
    """Random points at an exact distance from a centre, kept inside a square field."""

    def __init__(self, field_half_width_m: float, rng: random.Random, margin_m: float = 0.0):
        self.half_width = float(field_half_width_m) - float(margin_m)
        self.rng = rng

    def in_bounds(self, x: float, y: float) -> bool:
        eps = 1e-9
        return abs(x) <= self.half_width + eps and abs(y) <= self.half_width + eps

    def point_at_distance(self, cx: float, cy: float, r: float,
                          accept: Optional[Callable[[float, float], object]] = None):
        """Random angle on the circle of radius r around (cx, cy), inside the field.

        accept(x, y) may reject a candidate by returning None (e.g. hazardous / unreachable);
        whatever it returns otherwise (e.g. the planned path) comes back as the 3rd value.
        Returns (x, y, payload).
        """
        def check(th):
            x, y = cx + r * math.cos(th), cy + r * math.sin(th)
            if not self.in_bounds(x, y):
                return None
            payload = accept(x, y) if accept else True
            return None if payload is None else (x, y, payload)

        for _ in range(500):                  # rejection sampling: usually succeeds quickly
            got = check(self.rng.uniform(-math.pi, math.pi))
            if got:
                return got
        sweep = [k * 2 * math.pi / 720 for k in range(720)]   # then try every 0.5 deg
        self.rng.shuffle(sweep)
        for th in sweep:
            got = check(th)
            if got:
                return got
        raise ValueError(f'No safe, reachable point {r} m from ({cx:.1f}, {cy:.1f}) inside '
                         f'+/-{self.half_width:.1f} m')


# ============================================================================ collector
class CollectorState(str, Enum):
    WAITING_FOR_POSE = 'WAITING_FOR_POSE'
    TRAVELING = 'TRAVELING'
    COLLECTING = 'COLLECTING'
    EXCHANGING = 'EXCHANGING'
    WAITING_FOR_HAULER = 'WAITING_FOR_HAULER'
    DONE = 'DONE'


ACTIVE_STATES = {CollectorState.COLLECTING}
IDLE_STATES = {CollectorState.TRAVELING, CollectorState.EXCHANGING,
               CollectorState.WAITING_FOR_HAULER}


@dataclass
class CollectorConfig:
    first_site_radius_m: float = 100.0
    next_site_radius_m: float = 50.0
    field_half_width_m: float = 80.0      # field is [-80, 80] x [-80, 80]
    boundary_margin_m: float = 5.0        # sites stay this far inside the edge
    collection_duration_s: float = 900.0  # 1/4 hour
    mission_duration_s: float = 86400.0   # 24 h
    exchange_buffer_s: float = 120.0      # PLACEHOLDER until you calculate it
    exchange_every_n_sites: int = 1       # 0 = never exchange
    arrival_tolerance_m: float = 0.25
    waypoint_tolerance_m: float = 1.0     # intermediate path points (hazard avoidance only)
    max_detour_factor: float = 2.0        # reject sites whose safe path is > 2x straight line
    seed: Optional[int] = None            # None = different random sites every run


@dataclass
class Timers:
    active_s: float = 0.0
    idle_travel_s: float = 0.0
    idle_exchange_s: float = 0.0
    idle_wait_hauler_s: float = 0.0

    @property
    def idle_s(self) -> float:
        return self.idle_travel_s + self.idle_exchange_s + self.idle_wait_hauler_s

    def add(self, state: CollectorState, dt: float) -> None:
        if state == CollectorState.COLLECTING:
            self.active_s += dt
        elif state == CollectorState.TRAVELING:
            self.idle_travel_s += dt
        elif state == CollectorState.EXCHANGING:
            self.idle_exchange_s += dt
        elif state == CollectorState.WAITING_FOR_HAULER:
            self.idle_wait_hauler_s += dt


@dataclass
class Event:
    mission_time_s: float
    event: str
    state: str
    x: float
    y: float
    detail: str = ''


class CollectorMission:
    """Call update(t, x, y, yaw, hauler_aligned_id) every tick. Read .goal / .state."""

    def __init__(self, cfg: CollectorConfig, hazard_map=None):
        self.cfg = cfg
        self.map = hazard_map                              # terrain_map.HazardMap or None
        self.rng = random.Random(cfg.seed)
        self.sites = SiteGenerator(cfg.field_half_width_m, self.rng, cfg.boundary_margin_m)

        self.state = CollectorState.WAITING_FOR_POSE
        self.timers = Timers()
        self.events: List[Event] = []

        self.t_start: Optional[float] = None
        self.t_last: Optional[float] = None
        self.mission_time_s = 0.0

        self.goal: Optional[Tuple[float, float]] = None   # next point to drive to (None = stop)
        self.waypoints: List[Tuple[float, float]] = []    # remaining path, last one = site
        self.site_goal: Optional[Tuple[float, float]] = None
        self.sites_reached = 0
        self.collect_progress_s = 0.0
        self.loads_dumped = 0

        # container exchange bookkeeping (request id = site number it was requested at)
        self.exchange_request_id: Optional[int] = None
        self.last_completed_request_id: Optional[int] = None
        self.exchanges_completed = 0
        self.exchange_left_s = 0.0
        self.site_pose: Optional[Tuple[float, float, float]] = None  # x, y, yaw at arrival

        self._x = self._y = 0.0

    # ------------------------------------------------------------------ main entry point
    def update(self, t: float, x: float, y: float, yaw: float = 0.0,
               hauler_aligned_id: Optional[int] = None) -> CollectorState:
        self._x, self._y = x, y
        if self.state == CollectorState.DONE:
            return self.state

        if self.t_start is None:                       # first tick with a pose
            self.t_start = self.t_last = t
            if self._new_site(0.0, 0.0, self.cfg.first_site_radius_m, x, y):
                self._set_state(CollectorState.TRAVELING, 'MISSION_START',
                                f'first site ({self.site_goal[0]:.2f}, {self.site_goal[1]:.2f}), '
                                f'{len(self.waypoints)} waypoint(s)')
            return self.state

        dt = t - self.t_last
        self.t_last = t
        if dt <= 0.0:                                  # paused sim / clock reset: ignore
            return self.state

        # never count past the end of the mission, so active + idle == mission duration
        dt = min(dt, self.cfg.mission_duration_s - self.mission_time_s)
        self.mission_time_s += dt

        # 1) charge the elapsed time to the state we were in during it
        self.timers.add(self.state, dt)
        if self.state == CollectorState.COLLECTING:
            self.collect_progress_s += dt
        elif self.state == CollectorState.EXCHANGING:
            self.exchange_left_s -= dt

        # 2) mission over?
        if self.mission_time_s >= self.cfg.mission_duration_s - 1e-9:
            self.goal = None
            self._set_state(CollectorState.DONE, 'MISSION_COMPLETE', self.summary_line())
            return self.state

        # 3) state transitions
        self._transition(x, y, yaw, hauler_aligned_id)
        return self.state

    # ------------------------------------------------------------------ transitions
    def _transition(self, x, y, yaw, hauler_aligned_id):
        s = self.state
        aligned = (self.exchange_request_id is not None
                   and hauler_aligned_id == self.exchange_request_id)

        if s == CollectorState.TRAVELING:
            d = math.hypot(self.goal[0] - x, self.goal[1] - y)
            if len(self.waypoints) > 1:
                if d <= self.cfg.waypoint_tolerance_m:      # next leg of the safe path
                    self.waypoints.pop(0)
                    self.goal = self.waypoints[0]
            elif d <= self.cfg.arrival_tolerance_m:
                self._arrive(x, y, yaw)

        elif s == CollectorState.COLLECTING:
            if aligned:
                self._start_exchange()
            elif self.collect_progress_s >= self.cfg.collection_duration_s:
                self._finish_collection()

        elif s == CollectorState.WAITING_FOR_HAULER:
            if aligned:
                self._start_exchange()

        elif s == CollectorState.EXCHANGING:
            if self.exchange_left_s <= 1e-9:
                self._finish_exchange()

    def _arrive(self, x, y, yaw):
        self.sites_reached += 1
        self.goal = None
        self.waypoints = []
        self.collect_progress_s = 0.0
        self.site_pose = (x, y, yaw)
        self._set_state(CollectorState.COLLECTING, 'ARRIVED_START_COLLECTION',
                        f'site #{self.sites_reached}')
        n = self.cfg.exchange_every_n_sites
        if n > 0 and self.sites_reached % n == 0:
            self.exchange_request_id = self.sites_reached
            self._log('EXCHANGE_REQUESTED', f'request #{self.exchange_request_id}')

    def _start_exchange(self):
        self.exchange_left_s = self.cfg.exchange_buffer_s
        self._set_state(CollectorState.EXCHANGING, 'HAULER_ALIGNED_START_EXCHANGE',
                        f'request #{self.exchange_request_id}, buffer {self.cfg.exchange_buffer_s:.0f} s')
        if self.exchange_left_s <= 0.0:                # zero buffer: finish immediately
            self._finish_exchange()

    def _finish_exchange(self):
        self.last_completed_request_id = self.exchange_request_id
        self.exchange_request_id = None
        self.exchanges_completed += 1
        self._log('EXCHANGE_COMPLETE', f'request #{self.last_completed_request_id}')
        if self.collect_progress_s >= self.cfg.collection_duration_s:
            self._depart()                             # we were waiting for the hauler
        else:
            self._set_state(CollectorState.COLLECTING, 'RESUME_COLLECTION',
                            f'{self.cfg.collection_duration_s - self.collect_progress_s:.0f} s left')

    def _finish_collection(self):
        self.loads_dumped += 1
        self._log('COLLECTION_DONE_DUMP_TO_INBAY', f'load #{self.loads_dumped}')
        if self.exchange_request_id is not None:
            self._set_state(CollectorState.WAITING_FOR_HAULER, 'WAITING_FOR_HAULER',
                            f'request #{self.exchange_request_id}')
        else:
            self._depart()

    def _depart(self):
        if self._new_site(self._x, self._y, self.cfg.next_site_radius_m, self._x, self._y):
            self._set_state(CollectorState.TRAVELING, 'DEPART_TO_NEXT_SITE',
                            f'({self.site_goal[0]:.2f}, {self.site_goal[1]:.2f}), '
                            f'{len(self.waypoints)} waypoint(s)')

    def _new_site(self, cx, cy, radius, sx, sy) -> bool:
        """Pick a site `radius` from (cx, cy) and the path to it from (sx, sy)."""
        def accept(x, y):
            if self.map is None:
                return [(x, y)]                            # straight line
            if not self.map.is_free(x, y):
                return None                                # site itself is hazardous
            res = self.map.plan((sx, sy), (x, y))
            if res is None:
                return None                                # no safe path
            path, length = res
            if length > self.cfg.max_detour_factor * max(math.hypot(x - sx, y - sy), 1e-6):
                return None                                # reachable only by a huge detour
            return path
        try:
            x, y, path = self.sites.point_at_distance(cx, cy, radius, accept)
        except ValueError as e:
            self.goal, self.waypoints = None, []
            self._set_state(CollectorState.DONE, 'STOPPED_NO_SAFE_SITE', str(e))
            return False
        self.site_goal = (x, y)
        self.waypoints = list(path)
        self.goal = self.waypoints[0]
        return True

    # ------------------------------------------------------------------ reporting
    def _set_state(self, new: CollectorState, event: str, detail: str = ''):
        self.state = new
        self._log(event, detail)

    def _log(self, event: str, detail: str = ''):
        self.events.append(Event(round(self.mission_time_s, 3), event, self.state.value,
                                 round(self._x, 3), round(self._y, 3), detail))

    def status(self) -> Dict:
        """Snapshot published on /collector_mission_node/status (as JSON) and used by the hauler."""
        return {
            'state': self.state.value,
            'mission_time_s': round(self.mission_time_s, 2),
            'active_s': round(self.timers.active_s, 2),
            'idle_s': round(self.timers.idle_s, 2),
            'idle_travel_s': round(self.timers.idle_travel_s, 2),
            'idle_exchange_s': round(self.timers.idle_exchange_s, 2),
            'idle_wait_hauler_s': round(self.timers.idle_wait_hauler_s, 2),
            'sites_reached': self.sites_reached,
            'loads_dumped': self.loads_dumped,
            'exchanges_completed': self.exchanges_completed,
            'goal': list(self.goal) if self.goal else None,
            'site_goal': list(self.site_goal) if self.site_goal else None,
            'waypoints': [[round(a, 2), round(b, 2)] for a, b in self.waypoints],
            'collect_progress_s': round(self.collect_progress_s, 2),
            'exchange_request_id': self.exchange_request_id,
            'last_completed_request_id': self.last_completed_request_id,
            'site_pose': list(self.site_pose) if self.site_pose else None,
        }

    def summary(self) -> Dict:
        s = self.status()
        s.update({k: v for k, v in self.cfg.__dict__.items()})
        mt = max(self.mission_time_s, 1e-9)
        s['active_fraction'] = round(self.timers.active_s / mt, 4)
        s['idle_fraction'] = round(self.timers.idle_s / mt, 4)
        return s

    def summary_line(self) -> str:
        t = self.timers
        return (f'active {t.active_s / 3600:.2f} h | idle {t.idle_s / 3600:.2f} h '
                f'(travel {t.idle_travel_s / 3600:.2f}, exchange {t.idle_exchange_s / 3600:.2f}, '
                f'wait {t.idle_wait_hauler_s / 3600:.2f}) | sites {self.sites_reached} | '
                f'exchanges {self.exchanges_completed}')


# ============================================================================ hauler
class HaulerState(str, Enum):
    AT_BASE = 'AT_BASE'
    TO_COLLECTOR = 'TO_COLLECTOR'
    ALIGNING = 'ALIGNING'          # at the side-by-side spot, turning to match the collector
    ALIGNED = 'ALIGNED'            # parked; collector runs its exchange buffer
    RETURNING = 'RETURNING'        # carrying the full container back to base


@dataclass
class HaulerConfig:
    base_x: float = 0.0
    base_y: float = 2.0            # hauler spawn point in moon.sdf
    align_offset_m: float = 2.0    # park this far to the collector's LEFT (negative = right)
    position_tolerance_m: float = 0.3
    yaw_tolerance_rad: float = 0.1
    waypoint_tolerance_m: float = 1.0
    collector_keepout_m: float = 1.2   # hauler paths stay this far from the parked collector


@dataclass
class HaulerCommand:
    kind: str                      # 'goal' | 'rotate' | 'stop' | 'none'
    x: float = 0.0
    y: float = 0.0
    yaw_error: float = 0.0


class HaulerMission:
    """Call update(x, y, yaw, collector_status) every tick; act on the returned HaulerCommand."""

    def __init__(self, cfg: HaulerConfig, hazard_map=None):
        self.cfg = cfg
        self.map = hazard_map
        self.state = HaulerState.AT_BASE
        self.request_id: Optional[int] = None
        self.target: Optional[Tuple[float, float, float]] = None   # park x, y, yaw
        self.waypoints: List[Tuple[float, float]] = []
        self.keepout: Optional[Tuple[float, float, float]] = None   # collector x, y, radius
        self.trips = 0
        self.events: List[Tuple[str, str]] = []

    # ------------------------------------------------------------------ geometry / planning
    def park_pose(self, site_pose, side: float = 1.0, along: float = 0.0):
        """Parking spot beside the collector. side=+1 left, -1 right; along=+1 ahead, -1 behind."""
        sx, sy, syaw = site_pose
        d = self.cfg.align_offset_m
        lat, lon = side * d, along * d
        return (sx + lon * math.cos(syaw) - lat * math.sin(syaw),
                sy + lon * math.sin(syaw) + lat * math.cos(syaw), syaw)

    def _plan(self, x, y, goal) -> Optional[List[Tuple[float, float]]]:
        if self.map is None:
            return [goal]
        res = self.map.plan((x, y), goal, [self.keepout] if self.keepout else [])
        return res[0] if res else None

    def _choose_park(self, x, y, site_pose):
        """Left side first; if that is hazardous/unreachable try right, behind, ahead."""
        for side, along in ((1, 0), (-1, 0), (0, -1), (0, 1)):
            pose = self.park_pose(site_pose, side, along)
            if self.map is not None and not self.map.is_free(pose[0], pose[1]):
                continue
            path = self._plan(x, y, pose[:2])
            if path:
                return pose, path
        pose = self.park_pose(site_pose)
        self.events.append((self.state.value, 'NO SAFE PARKING SPOT - driving straight to the left side'))
        return pose, [pose[:2]]

    def _follow(self, x, y, final_tol) -> bool:
        """Advance along waypoints. True when the last one is reached."""
        while len(self.waypoints) > 1 and math.hypot(self.waypoints[0][0] - x,
                                                     self.waypoints[0][1] - y) <= self.cfg.waypoint_tolerance_m:
            self.waypoints.pop(0)
        wx, wy = self.waypoints[0]
        return len(self.waypoints) == 1 and math.hypot(wx - x, wy - y) <= final_tol

    def aligned_id(self) -> Optional[int]:
        return self.request_id if self.state == HaulerState.ALIGNED else None

    # ------------------------------------------------------------------ main entry point
    def update(self, x: float, y: float, yaw: float, collector: Optional[Dict]) -> HaulerCommand:
        req = collector.get('exchange_request_id') if collector else None
        done = collector.get('last_completed_request_id') if collector else None
        s = self.state

        # accept a new request only at base (it must drop the full container and pick up
        # an empty one first). If it is late, the collector waits -> idle_wait_hauler_s.
        if s == HaulerState.AT_BASE and req is not None \
                and req != done and collector.get('site_pose'):
            self.request_id = req
            sp = collector['site_pose']
            self.keepout = (sp[0], sp[1], self.cfg.collector_keepout_m)
            self.target, self.waypoints = self._choose_park(x, y, sp)
            self._set(HaulerState.TO_COLLECTOR, f'request #{req}, {len(self.waypoints)} waypoint(s)')
            s = self.state

        # request vanished / collector mission ended before the swap -> go home
        mission_over = bool(collector) and collector.get('state') == CollectorState.DONE.value
        swapped = done is not None and done == self.request_id   # handled in ALIGNED below
        if s in (HaulerState.TO_COLLECTOR, HaulerState.ALIGNING, HaulerState.ALIGNED) and \
                ((req != self.request_id and not swapped) or mission_over):
            self._go_home(x, y, 'mission over' if mission_over else 'request withdrawn')
            s = self.state

        if s == HaulerState.TO_COLLECTOR:
            if self._follow(x, y, self.cfg.position_tolerance_m):
                self._set(HaulerState.ALIGNING)
                s = self.state
            else:
                return HaulerCommand('goal', *self.waypoints[0])

        if s == HaulerState.ALIGNING:
            err = wrap_pi(self.target[2] - yaw)
            if abs(err) <= self.cfg.yaw_tolerance_rad:
                self._set(HaulerState.ALIGNED, f'request #{self.request_id}')
                return HaulerCommand('stop')
            return HaulerCommand('rotate', yaw_error=err)

        if s == HaulerState.ALIGNED:
            if done == self.request_id:
                self.trips += 1
                self._go_home(x, y, f'container swapped (request #{self.request_id})')
                return HaulerCommand('goal', *self.waypoints[0])
            return HaulerCommand('none')

        if s == HaulerState.RETURNING:
            if self._follow(x, y, self.cfg.position_tolerance_m):
                self.keepout = None
                self._set(HaulerState.AT_BASE, 'unloaded full container (visual only)')
                return HaulerCommand('stop')
            return HaulerCommand('goal', *self.waypoints[0])

        return HaulerCommand('none')

    def _go_home(self, x: float, y: float, why: str):
        self.request_id = None
        self.target = None
        base = (self.cfg.base_x, self.cfg.base_y)
        path = self._plan(x, y, base)
        if path is None:
            self.events.append((self.state.value, 'NO SAFE PATH HOME - driving straight'))
            path = [base]
        self.waypoints = path
        self._set(HaulerState.RETURNING, why)

    def _set(self, new: HaulerState, detail: str = ''):
        self.state = new
        self.events.append((new.value, detail))

    def status(self) -> Dict:
        return {
            'state': self.state.value,
            'request_id': self.request_id,
            'aligned_request_id': self.aligned_id(),
            'trips_completed': self.trips,
            'target': list(self.target) if self.target else None,
            'waypoints': [[round(a, 2), round(b, 2)] for a, b in self.waypoints],
        }


# ============================================================================ config + results
def collector_config_from_params(p: Dict) -> CollectorConfig:
    """Build a CollectorConfig from the collector_mission_node ros__parameters dict."""
    seed = p.get('random_seed', -1)
    return CollectorConfig(
        first_site_radius_m=float(p.get('first_site_radius_m', 100.0)),
        next_site_radius_m=float(p.get('next_site_radius_m', 50.0)),
        field_half_width_m=float(p.get('field_half_width_m', 80.0)),
        boundary_margin_m=float(p.get('boundary_margin_m', 5.0)),
        collection_duration_s=float(p.get('collection_duration_s', 900.0)),
        mission_duration_s=float(p.get('mission_duration_s', 86400.0)),
        exchange_buffer_s=float(p.get('exchange_buffer_s', 120.0)),
        exchange_every_n_sites=int(p.get('exchange_every_n_sites', 1)),
        arrival_tolerance_m=float(p.get('arrival_tolerance_m', 0.25)),
        waypoint_tolerance_m=float(p.get('waypoint_tolerance_m', 1.0)),
        max_detour_factor=float(p.get('max_detour_factor', 2.0)),
        seed=None if int(seed) < 0 else int(seed),
    )


def hauler_config_from_params(p: Dict) -> HaulerConfig:
    """Build a HaulerConfig from the hauler_mission_node ros__parameters dict."""
    return HaulerConfig(
        base_x=float(p.get('base_x', 0.0)),
        base_y=float(p.get('base_y', 2.0)),
        align_offset_m=float(p.get('align_offset_m', 2.0)),
        position_tolerance_m=float(p.get('position_tolerance_m', 0.3)),
        yaw_tolerance_rad=float(p.get('yaw_tolerance_rad', 0.1)),
        waypoint_tolerance_m=float(p.get('waypoint_tolerance_m', 1.0)),
        collector_keepout_m=float(p.get('collector_keepout_m', 1.2)),
    )


def write_results(mission: CollectorMission, out_dir: str, tag: str) -> Tuple[str, str]:
    """Write <tag>_events.csv and <tag>_summary.json into out_dir. Returns both paths."""
    import csv
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'{tag}_events.csv')
    json_path = os.path.join(out_dir, f'{tag}_summary.json')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mission_time_s', 'event', 'state', 'x', 'y', 'detail'])
        for e in mission.events:
            w.writerow([e.mission_time_s, e.event, e.state, e.x, e.y, e.detail])
    with open(json_path, 'w') as f:
        json.dump(mission.summary(), f, indent=2)
    return csv_path, json_path
