# %% [markdown]
# # In-bay storage vs dump-and-pour: excavator active / idle time (ROS 2 kinematic study)
#
# **What this file does.** One excavator and N haulers work a 16 h operating block (the 8 h
# charging block is outside the simulation). Two scenarios with identical rules otherwise:
#
# * **dump_and_pour** (SRC2 style): after each 15 min collection the excavator holds the load in
#   its bucket and is **idle** until a hauler has been allocated (auction open time) **and** has
#   driven over to take the load.
# * **in_bay**: the excavator collects into a 100 kg in-bay container and keeps working. It calls
#   a hauler once the container can no longer guarantee room for another site's load; the hauler
#   swaps containers whenever it gets there. The excavator is only idle if the container fills
#   before the hauler arrives.
#
# **Idle = travel between sites + waiting for the auction + waiting for the hauler to drive over.**
# **Active = collecting.** Active + idle = 16 h exactly.
#
# **How ROS 2 is used.** Each rover is a ROS 2 node. A `sim_clock` node steps simulated time:
# every tick it publishes `/sim/tick` (time + a snapshot of every rover's last status), each rover
# advances by dt and answers on `/sim/status`. Nothing advances until every rover has answered,
# so a run is fast (no wall-clock waiting) and **the same seed always gives the same result**.
# The auction is *not* implemented as a negotiation: it is modelled as its measured duration
# (Kashid et al. 2024, Policy 1, excavator-to-hauler: 9.3 min open time), after which the nearest
# free hauler (shortest safe path; their cost = distance) gets the task.
#
# **How to run.** Open in VS Code (Python + Jupyter extensions), started from a terminal where
# you ran `source /opt/ros/lyrical/setup.bash` (and your workspace `install/setup.bash`).
# in "direct" mode (same logic, no middleware) so you can check results anywhere.
#
# **Reading order for review:** cell 2 (parameters) -> cell 4 (rover logic: `Excavator`,
# `Hauler`) -> cell 8 (checks). Cells 5-6 are plumbing.

# %% 1. Imports
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Tuple
from typing import Iterable, Sequence, Callable
import heapq

import numpy as np

_here = os.path.dirname(os.path.abspath(globals().get('__file__', os.path.join(os.getcwd(), '_'))))
_pkg = os.path.dirname(_here)                       # .../src/moon_rover_bridge
if os.path.isdir(os.path.join(_pkg, 'moon_rover_bridge')) and _pkg not in sys.path:
    sys.path.insert(0, _pkg)

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

# %% 2. Parameters  <-- the only cell you normally edit
@dataclass(frozen=True)
class Params:
    scenario: str = 'in_bay'              # 'in_bay' or 'dump_and_pour'
    seed: int = 1                         # same seed -> same sites, loads, terrain choices
    # time
    sim_hours: float = 16.0               # operating block; stops wherever it is at 16 h
    dt: float = 1.0                       # s, simulation step
    # field + sites
    base: Tuple[float, float] = (0.0, 0.0)  # home base / processing plant
    field_half_width: float = 80.0        # field is [-80, 80] x [-80, 80]
    boundary_margin: float = 5.0          # sites and paths stay inside +/-75 m
    first_site_radius: float = 100.0      # first site exactly 100 m from base
    next_site_radius: float = 50.0        # every next site exactly 50 m from the current one
    max_detour_factor: float = 2.0        # skip sites whose safe path is > 2x straight distance
    # rovers
    speed: float = 0.27                   # m/s, excavator and hauler, constant (payload ignored)
    n_haulers: int = 1
    collect_s: float = 900.0              # 15 min per site
    load_kg_min: float = 40.0             # regolith per site ~ U(40, 60) kg (RASSOR)
    load_kg_max: float = 60.0
    container_kg: float = 100.0           # in-bay container capacity
    auction_s: float = 9.3 * 60           # auction open time (Kashid 2024, Policy 1, best case)
    handoff_s: float = 0.0                # transfer / swap itself: common irreducible loss -> 0
    park_offset: float = 2.0              # hauler parks this far beside the excavator
    keepout: float = 1.2                  # hauler paths stay this far from the excavator
    # terrain (see cell 3)
    terrain_obj: str = ''                 # path to your Blender OBJ; '' = synthetic crater field
    terrain_up: str = 'auto'              # OBJ up axis: 'auto', 'y' (Blender default), 'z'
    terrain_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    terrain_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    terrain_res: float = 0.5              # m per grid cell
    use_terrain: bool = True              # False = flat, straight lines (for the arithmetic check)
    z_min: float = -0.5                   # hazard if terrain z below this   (SET for your terrain)
    z_max: float = 0.5                    # hazard if terrain z above this   (SET for your terrain)
    max_slope_deg: float = 15.0           # hazard if steeper
    inflation: float = 1.0                # keep rover centres this far from hazards


BASE = Params()
SCENARIOS = ['dump_and_pour', 'in_bay']
print(BASE)

# %% 3. Terrain: your Blender OBJ, or a synthetic crater field for testing
def dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Binary dilation with a disc (no scipy needed)."""
    if radius_cells <= 0:
        return mask.copy()
    out = mask.copy()
    h, w = mask.shape
    r = radius_cells
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r or (dx == 0 and dy == 0):
                continue
            ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
            xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
            out[yd, xd] |= mask[ys, xs]
    return out

class HazardMap:

    def __init__(self, z: np.ndarray, origin_x: float, origin_y: float, resolution: float):
        self.z = np.asarray(z, dtype=float)
        self.ox, self.oy, self.res = float(origin_x), float(origin_y), float(resolution)
        self.h, self.w = self.z.shape
        self.slope_deg = np.zeros_like(self.z)
        self.hazard = ~np.isfinite(self.z)
        self.blocked = self.hazard.copy()
        self.inflation_cells = 0

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str) -> 'HazardMap':
        d = np.load(path)
        return cls(d['z'], float(d['origin'][0]), float(d['origin'][1]), float(d['resolution']))

    def save(self, path: str) -> None:
        np.savez_compressed(path, z=self.z, origin=np.array([self.ox, self.oy]),
                            resolution=np.array(self.res))

    # ------------------------------------------------------------------ hazards
    def configure(self, z_min: float, z_max: float, max_slope_deg: float, inflation_m: float,
                  field_limit_m: Optional[float] = None) -> 'HazardMap':
        z = self.z
        missing = ~np.isfinite(z)
        with np.errstate(invalid='ignore'):
            gy, gx = np.gradient(z, self.res)
            self.slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
            haz = missing | (z < z_min) | (z > z_max) | (self.slope_deg > max_slope_deg)
        if field_limit_m is not None:
            xc, yc = self.cell_centres()
            haz |= (np.abs(xc)[None, :] > field_limit_m) | (np.abs(yc)[:, None] > field_limit_m)
        self.hazard = haz
        self.inflation_cells = int(math.ceil(inflation_m / self.res))
        self.blocked = dilate(haz, self.inflation_cells)
        return self

    def cell_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.ox + (np.arange(self.w) + 0.5) * self.res,
                self.oy + (np.arange(self.h) + 0.5) * self.res)

    # ------------------------------------------------------------------ queries
    def cell(self, x: float, y: float) -> Tuple[int, int]:
        return int(math.floor((y - self.oy) / self.res)), int(math.floor((x - self.ox) / self.res))

    def centre(self, r: int, c: int) -> XY:
        return self.ox + (c + 0.5) * self.res, self.oy + (r + 0.5) * self.res

    def inside(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w

    def is_free(self, x: float, y: float, grid: Optional[np.ndarray] = None) -> bool:
        r, c = self.cell(x, y)
        g = self.blocked if grid is None else grid
        return self.inside(r, c) and not g[r, c]

    def height_at(self, x: float, y: float) -> float:
        r, c = self.cell(x, y)
        return float(self.z[r, c]) if self.inside(r, c) else float('nan')

    def free_fraction(self) -> float:
        return float(1.0 - self.blocked.mean())

    # ------------------------------------------------------------------ planning
    def _with_extras(self, start: XY, keepouts: Iterable[Tuple[float, float, float]]) -> np.ndarray:
        g = self.blocked.copy()
        rr, cc = np.mgrid[0:self.h, 0:self.w]
        xs = self.ox + (cc + 0.5) * self.res
        ys = self.oy + (rr + 0.5) * self.res
        for kx, ky, kr in keepouts:                      # e.g. the parked collector
            g |= (xs - kx) ** 2 + (ys - ky) ** 2 <= kr * kr
        # let the rover escape if it is standing in the inflated band (not on raw hazard)
        er = (self.inflation_cells + 1) * self.res
        near = (xs - start[0]) ** 2 + (ys - start[1]) ** 2 <= er * er
        g[near & ~self.hazard] = False
        return g

    def line_free(self, a: XY, b: XY, grid: np.ndarray) -> bool:
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(d / (self.res / 3.0)) + 1)
        for k in range(n + 1):
            t = k / n
            if not self.is_free(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), grid):
                return False
        return True

    def plan(self, start: XY, goal: XY,
             keepouts: Sequence[Tuple[float, float, float]] = ()) -> Optional[Tuple[List[XY], float]]:
        """A* (8-connected, no corner cutting) + line-of-sight smoothing.

        Returns (waypoints, length_m) with waypoints[-1] == goal exactly, or None if unreachable.
        """
        grid = self._with_extras(start, keepouts)
        if not self.is_free(goal[0], goal[1], grid):
            return None
        if self.line_free(start, goal, grid):
            return [goal], math.hypot(goal[0] - start[0], goal[1] - start[1])

        sr, sc = self.cell(*start)
        gr, gc = self.cell(*goal)
        if not self.inside(sr, sc):
            return None
        w = self.w
        free = ~grid.ravel()
        s, g = sr * w + sc, gr * w + gc
        free[s] = True
        INF = float('inf')
        cost = np.full(self.h * w, INF)
        parent = np.full(self.h * w, -1, dtype=np.int64)
        cost[s] = 0.0
        SQ2 = math.sqrt(2.0)
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2)]

        def hfun(i):
            dr, dc = abs(i // w - gr), abs(i % w - gc)
            return (dr + dc) + (SQ2 - 2.0) * min(dr, dc)

        heap = [(hfun(s), 0.0, s)]
        found = False
        while heap:
            _, gcost, i = heapq.heappop(heap)
            if i == g:
                found = True
                break
            if gcost > cost[i]:
                continue
            r, c = divmod(i, w)
            for dr, dc, step in nbrs:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.h and 0 <= nc < w):
                    continue
                j = nr * w + nc
                if not free[j]:
                    continue
                if dr and dc and not (free[r * w + nc] and free[nr * w + c]):
                    continue                              # no squeezing past a corner
                ng = gcost + step
                if ng < cost[j]:
                    cost[j] = ng
                    parent[j] = i
                    heapq.heappush(heap, (ng + hfun(j), ng, j))
        if not found:
            return None

        cells = []
        i = g
        while i != -1:
            cells.append(i)
            i = int(parent[i])
        cells.reverse()
        pts = [start] + [self.centre(*divmod(k, w)) for k in cells[1:-1]] + [goal]

        # string-pulling: keep only the points needed for straight, hazard-free legs
        out, anchor = [], 0
        for k in range(2, len(pts)):
            if not self.line_free(pts[anchor], pts[k], grid):
                out.append(pts[k - 1])
                anchor = k - 1
        out.append(goal)
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip([start] + out[:-1], out))
        return out, length


def synthetic_crater_field(seed=7, res=0.5, half=80.0, n_craters=14, clear_radius=12.0):
    """Placeholder lunar-like terrain (bowl craters with raised rims). NOT real data:
    set Params.terrain_obj to your Blender export for the paper."""
    rng = np.random.default_rng(seed)
    xc = -half + (np.arange(int(2 * half / res)) + 0.5) * res
    X, Y = np.meshgrid(xc, xc)
    z = 0.05 * np.sin(X / 7.0) * np.cos(Y / 9.0)
    placed = 0
    while placed < n_craters:
        cx, cy = rng.uniform(-half + 8, half - 8, 2)
        r = rng.uniform(3.0, 10.0)
        if math.hypot(cx, cy) < clear_radius + r:
            continue                                   # keep the base area clear
        rho = np.hypot(X - cx, Y - cy) / r
        depth = 0.25 * r
        z += np.where(rho < 1, -depth * (1 - rho ** 2), 0.0) + 0.15 * depth * np.exp(-((rho - 1) / 0.25) ** 2)
        placed += 1
    return HazardMap(z, -half, -half, res)


def load_terrain(P: Params) -> Optional[HazardMap]:
    if not P.use_terrain:
        return None
    if P.terrain_obj:
        from moon_rover_bridge.build_hazard_map import grid_from_mesh
        z, x0, y0 = grid_from_mesh(os.path.expanduser(P.terrain_obj), P.terrain_res,
                                   P.terrain_scale, P.terrain_offset, P.terrain_up)
        m = HazardMap(z, x0, y0, P.terrain_res)
    else:
        m = synthetic_crater_field(res=P.terrain_res, half=P.field_half_width)
    m.configure(P.z_min, P.z_max, P.max_slope_deg, P.inflation,
                field_limit_m=P.field_half_width - P.boundary_margin)
    if not m.is_free(*P.base):
        raise ValueError('The base is on a hazard / safety margin: move Params.base or relax thresholds.')
    return m


TERRAIN = load_terrain(BASE)
if TERRAIN is not None:
    zf = TERRAIN.z[np.isfinite(TERRAIN.z)]
    print(f"terrain: {'Blender OBJ ' + BASE.terrain_obj if BASE.terrain_obj else 'SYNTHETIC crater field'}"
          f" | z p1/p50/p99 = {np.percentile(zf, 1):.2f} / {np.percentile(zf, 50):.2f} / "
          f"{np.percentile(zf, 99):.2f} m | drivable {100 * TERRAIN.free_fraction():.1f} %")

# %% 4. Rover logic (plain Python, no ROS). This is what to review.

XY = Tuple[float, float]


def advance(pos: XY, path: List[XY], dist: float) -> Tuple[XY, float]:
    """Move up to `dist` metres along the waypoint list (consumed in place)."""
    moved = 0.0
    while path and dist > 1e-12:
        tx, ty = path[0]
        dx, dy = tx - pos[0], ty - pos[1]
        d = math.hypot(dx, dy)
        if d <= dist:
            pos, dist, moved = (tx, ty), dist - d, moved + d
            path.pop(0)
        else:
            pos = (pos[0] + dx / d * dist, pos[1] + dy / d * dist)
            moved, dist = moved + dist, 0.0
    return pos, moved


def plan(terrain: Optional[HazardMap], a: XY, b: XY, keepouts=()) -> Optional[Tuple[List[XY], float]]:
    """Safe path a -> b (waypoints, length). Straight line when there is no terrain."""
    if terrain is None:
        return [tuple(b)], math.hypot(b[0] - a[0], b[1] - a[1])
    return terrain.plan(tuple(a), tuple(b), keepouts)


IDLE_KEYS = ('idle_travel', 'idle_auction', 'idle_hauler')

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


class Excavator:
    """States: TRAVEL (idle) | COLLECT (active) | WAIT_HANDOFF (dump_and_pour, idle) |
    WAIT_FULL (in_bay container full, idle)."""
    name = 'excavator'

    def __init__(self, P: Params, terrain: Optional[HazardMap]):
        self.P, self.map = P, terrain
        self.rng_sites = random.Random(P.seed)          # site sequence (same in both scenarios)
        self.rng_loads = random.Random(P.seed + 10_000)  # load per site (same k-th load in both)
        self.sites = SiteGenerator(P.field_half_width, self.rng_sites, P.boundary_margin)
        self.pos: XY = tuple(P.base)
        self.path: List[XY] = []
        self.site: Optional[XY] = None                   # where it is collecting (None = driving)
        self.site_goal: Optional[XY] = None
        self.heading = 0.0                               # direction of the leg into the site
        self.state = 'TRAVEL'
        self.collect_done_s = 0.0
        self.site_load = 0.0
        self.held_kg = 0.0                               # bucket (dump) or container (in_bay)
        self.request: Optional[Dict] = None              # open hauler request
        self.n_requests = 0
        self.last_handoff = None                         # (request id, kg) for the hauler
        self.T = {'active': 0.0, 'idle_travel': 0.0, 'idle_auction': 0.0, 'idle_hauler': 0.0}
        self.kg_collected = 0.0
        self.sites_started = 0
        self.sites_completed = 0
        self.distance = 0.0
        self.handoffs: List[Dict] = []
        self.timeline: List[Tuple[float, str]] = []      # (start time, category) on change
        self.track: List[XY] = []
        self._choose_site(P.base, P.first_site_radius)

    # ---------------------------------------------------------------- one step [t, t + dt)
    def step(self, t: float, dt: float, snap: Dict[str, Dict]):
        haulers = {n: s for n, s in snap.items() if n != self.name}

        # (a) the assigned hauler is parked beside us and ready -> hand the load over
        if self.request and self.request['hauler'] and self.site is not None:
            hs = haulers.get(self.request['hauler'])
            if hs and hs['ready_for'] == self.request['id']:
                self._handoff(t)

        # (b) auction: closes after auction_s, nearest FREE hauler wins (stays open if none free)
        if self.request and self.request['hauler'] is None and t >= self.request['t_open'] + self.P.auction_s:
            free = [(n, s) for n, s in haulers.items() if s['free']]
            if free:
                meet = self.meet_site()
                bids = sorted((self._path_len(s['pos'], meet), n) for n, s in free)
                self.request['hauler'], self.request['t_assigned'] = bids[0][1], t

        # (c) charge this interval to the current activity
        cat = self._category()
        self.T[cat] += dt
        if not self.timeline or self.timeline[-1][1] != cat:
            self.timeline.append((t, cat))
        if int(t) % 10 == 0:
            self.track.append(self.pos)

        # (d) do the activity
        if self.state == 'TRAVEL':
            self.pos, moved = advance(self.pos, self.path, self.P.speed * dt)
            self.distance += moved
            if not self.path:
                self._arrive()
        elif self.state == 'COLLECT':
            self._collect(t, dt)
        # WAIT_HANDOFF / WAIT_FULL: nothing to do but wait

    def _category(self) -> str:
        if self.state == 'COLLECT':
            return 'active'
        if self.state == 'TRAVEL':
            return 'idle_travel'
        return 'idle_auction' if (self.request and self.request['hauler'] is None) else 'idle_hauler'

    # ---------------------------------------------------------------- collection
    def _collect(self, t, dt):
        P = self.P
        rate = self.site_load / P.collect_s                       # kg/s at this site
        need_s = min(dt, P.collect_s - self.collect_done_s)       # seconds left at this site
        want = rate * need_s
        if P.scenario == 'in_bay':
            take = min(want, P.container_kg - self.held_kg)        # container may fill mid-site
        else:
            take = want                                           # bucket holds one site's load
        used_s = take / rate                                      # seconds actually collecting
        self.held_kg += take
        self.kg_collected += take
        self.collect_done_s += used_s

        if P.scenario == 'in_bay' and self.request is None and \
                self.held_kg > P.container_kg - P.load_kg_max + 1e-9:
            self._open_request(t + dt, 'container cannot fit another full site load')

        if self.collect_done_s >= P.collect_s - 1e-9:             # site finished
            self.sites_completed += 1
            if P.scenario == 'dump_and_pour':
                self.state = 'WAIT_HANDOFF'
                self._open_request(t + dt, 'site load in bucket')
            else:
                self._depart()
        elif take < want - 1e-12:                                 # in_bay container full
            self.state = 'WAIT_FULL'
            if self.request is None:
                self._open_request(t + dt, 'container full')
        spare = dt - used_s               # part of this second not spent collecting: charge it
        if spare > 1e-12:                 # to what the excavator does next (travel / waiting)
            self.T['active'] -= spare
            self.T[self._category()] += spare

    def _open_request(self, t, why):
        self.n_requests += 1
        self.request = {'id': self.n_requests, 't_open': t, 'hauler': None, 't_assigned': None, 'why': why}

    def _handoff(self, t):
        r = self.request
        self.handoffs.append({'id': r['id'], 'why': r['why'], 'hauler': r['hauler'],
                              't_open': r['t_open'], 't_assigned': r['t_assigned'], 't_done': t,
                              'kg': self.held_kg, 'excavator_state': self.state})
        self.last_handoff = (r['id'], self.held_kg)
        self.held_kg = 0.0
        self.request = None
        if self.state == 'WAIT_HANDOFF':
            self._depart()
        elif self.state == 'WAIT_FULL':
            self.state = 'COLLECT'

    # ---------------------------------------------------------------- sites
    def _arrive(self):
        self.site = self.site_goal
        self.state = 'COLLECT'
        self.collect_done_s = 0.0
        self.site_load = self.rng_loads.uniform(self.P.load_kg_min, self.P.load_kg_max)
        self.sites_started += 1

    def _depart(self):
        self.site = None
        self._choose_site(self.pos, self.P.next_site_radius)
        self.state = 'TRAVEL'

    def _choose_site(self, centre: XY, radius: float):
        start = self.pos

        def accept(x, y):                      # safe, reachable, not a huge detour
            if self.map is not None and not self.map.is_free(x, y):
                return None
            res = plan(self.map, start, (x, y))
            if res is None or res[1] > self.P.max_detour_factor * max(math.hypot(x - start[0], y - start[1]), 1e-6):
                return None
            return res[0]
        x, y, path = self.sites.point_at_distance(centre[0], centre[1], radius, accept)
        self.site_goal, self.path = (x, y), list(path)
        prev = ([start] + self.path)[-2]
        self.heading = math.atan2(y - prev[1], x - prev[0])

    def meet_site(self) -> XY:
        return self.site if self.site is not None else self.site_goal

    def _path_len(self, a, b) -> float:
        res = plan(self.map, a, b)
        return res[1] if res else float('inf')

    def status(self) -> Dict:
        r = self.request
        return {'state': self.state, 'pos': list(self.pos),
                'at_site': list(self.site) if self.site is not None else None,
                'meet_site': list(self.meet_site()), 'heading': self.heading,
                'request_id': r['id'] if r else None, 'assigned': r['hauler'] if r else None,
                'last_handoff': list(self.last_handoff) if self.last_handoff else None}


class Hauler:
    """States: AT_BASE (free) | TO_SITE | AT_SITE (parked, waiting) | TO_BASE (carrying load)."""

    def __init__(self, P: Params, terrain: Optional[HazardMap], name: str):
        self.P, self.map, self.name = P, terrain, name
        self.pos: XY = tuple(P.base)
        self.path: List[XY] = []
        self.state = 'AT_BASE'
        self.request_id = None
        self.target_site = None
        self.ready_for = None
        self.load_kg = 0.0
        self.delivered_kg = 0.0
        self.trips = 0
        self.distance = 0.0
        self.no_safe_path = 0
        self.track: List[XY] = []

    def step(self, t: float, dt: float, snap: Dict[str, Dict]):
        ex = snap['excavator']
        self.ready_for = None

        # assigned to a new request while at base -> drive to the excavator
        if self.state == 'AT_BASE' and ex['assigned'] == self.name and ex['request_id'] != self.request_id:
            self.request_id = ex['request_id']
            self._drive_to_excavator(ex)
        elif self.state in ('TO_SITE', 'AT_SITE'):
            done = ex['last_handoff']
            if done and done[0] == self.request_id:        # load handed over -> take it home
                self.load_kg = done[1]
                self.trips += 1
                self._drive(tuple(self.P.base), keepout=ex['pos'])
                self.state = 'TO_BASE'
            elif ex['meet_site'] != self.target_site:       # excavator moved on -> follow it
                self._drive_to_excavator(ex)

        if int(t) % 10 == 0:
            self.track.append(self.pos)
        if self.state in ('TO_SITE', 'TO_BASE'):
            self.pos, moved = advance(self.pos, self.path, self.P.speed * dt)
            self.distance += moved
            if not self.path:
                if self.state == 'TO_SITE':
                    self.state = 'AT_SITE'
                else:
                    self.delivered_kg += self.load_kg       # unloaded at base (visual only)
                    self.load_kg = 0.0
                    self.state = 'AT_BASE'
        if self.state == 'AT_SITE' and ex['at_site'] == self.target_site:
            self.ready_for = self.request_id                # excavator is here: ready to swap

    def _drive_to_excavator(self, ex):
        self.target_site = ex['meet_site']
        sx, sy = self.target_site
        h, d = ex['heading'], self.P.park_offset
        for lat, lon in ((1, 0), (-1, 0), (0, -1), (0, 1)):     # left, right, behind, ahead
            px = sx + d * (lon * math.cos(h) - lat * math.sin(h))
            py = sy + d * (lon * math.sin(h) + lat * math.cos(h))
            if self.map is not None and not self.map.is_free(px, py):
                continue
            res = plan(self.map, self.pos, (px, py), [(sx, sy, self.P.keepout)])
            if res:
                self.path, self.state = list(res[0]), 'TO_SITE'
                return
        self.no_safe_path += 1                                   # fallback: straight line
        self.path, self.state = [(sx - d * math.sin(h), sy + d * math.cos(h))], 'TO_SITE'

    def _drive(self, goal: XY, keepout=None):
        ko = [(keepout[0], keepout[1], self.P.keepout)] if keepout else []
        res = plan(self.map, self.pos, goal, ko)
        if res is None:
            self.no_safe_path += 1
            res = ([goal], 0.0)
        self.path = list(res[0])

    def status(self) -> Dict:
        return {'state': self.state, 'pos': list(self.pos), 'free': self.state == 'AT_BASE',
                'ready_for': self.ready_for}


def make_rovers(P: Params, terrain):
    return [Excavator(P, terrain)] + [Hauler(P, terrain, f'hauler_{i + 1}') for i in range(P.n_haulers)]


def roundtrip(d):  # what a status looks like after going through a ROS message
    return json.loads(json.dumps(d))

# %% 5. ROS 2 nodes: one per rover + the stepping clock


QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=100)

class RoverNode(Node):
            """Wraps one rover: on /sim/tick k it steps once and answers on /sim/status."""

            def __init__(self, logic, context):
                super().__init__(logic.name, context=context)
                self.logic, self.last_k, self.cached = logic, -1, None
                self.pub = self.create_publisher(String, '/sim/status', QOS)
                self.create_subscription(String, '/sim/tick', self._on_tick, QOS)

            def _on_tick(self, msg):
                d = json.loads(msg.data)
                k = d['k']
                if k == self.last_k and self.cached is not None:     # clock re-sent: answer again
                    self.pub.publish(self.cached)
                    return
                if k != self.last_k + 1:
                    return
                if k > 0:                                            # tick 0 = roll call only
                    self.logic.step(d['t'], d['dt'], d['snapshot'])
                self.last_k = k
                self.cached = String(data=json.dumps({'k': k, 'name': self.logic.name,
                                                    'status': self.logic.status()}))
                self.pub.publish(self.cached)

class SimClock(Node):
    """Publishes tick k with everyone's status from tick k-1; waits for all answers."""

    def __init__(self, names, context):
        super().__init__('sim_clock', context=context)
        self.names = set(names)
        self.inbox: Dict[int, Dict] = {}
        self.pub = self.create_publisher(String, '/sim/tick', QOS)
        self.create_subscription(String, '/sim/status', self._on_status, QOS)

    def _on_status(self, msg):
        d = json.loads(msg.data)
        self.inbox.setdefault(d['k'], {})[d['name']] = d['status']

    def send(self, k, t, dt, snapshot):
        self.pub.publish(String(data=json.dumps({'k': k, 't': t, 'dt': dt, 'snapshot': snapshot})))

    def complete(self, k):
        return set(self.inbox.get(k, {})) == self.names

# %% 6. Direct runner (same logic, no middleware) + results


def run_direct(P: Params, terrain=None):
    terrain = load_terrain(P) if terrain is None else terrain
    rovers = make_rovers(P, terrain)
    snapshot = {r.name: roundtrip(r.status()) for r in rovers}       # = tick 0 roll call
    for k in range(int(round(P.sim_hours * 3600 / P.dt))):
        t = k * P.dt
        new = {}
        for r in rovers:                     # every rover sees the SAME previous snapshot
            r.step(t, P.dt, snapshot)
            new[r.name] = roundtrip(r.status())
        snapshot = new
    return Result(P, rovers, terrain)


class Result:
    def __init__(self, P, rovers, terrain):
        self.P, self.terrain = P, terrain
        self.ex: Excavator = rovers[0]
        self.haulers: List[Hauler] = rovers[1:]

    def summary(self) -> Dict:
        e, P = self.ex, self.P
        h = {k: v / 3600 for k, v in e.T.items()}
        idle = sum(h[k] for k in IDLE_KEYS)
        return {'scenario': P.scenario, 'seed': P.seed, 'n_haulers': P.n_haulers,
                'auction_min': round(P.auction_s / 60, 3), 'active_h': round(h['active'], 4),
                'idle_h': round(idle, 4), 'idle_travel_h': round(h['idle_travel'], 4),
                'idle_auction_h': round(h['idle_auction'], 4), 'idle_hauler_h': round(h['idle_hauler'], 4),
                'active_pct': round(100 * h['active'] / P.sim_hours, 3),
                'sites_started': e.sites_started, 'sites_completed': e.sites_completed,
                'kg_collected': round(e.kg_collected, 2),
                'kg_delivered': round(sum(x.delivered_kg for x in self.haulers), 2),
                'handoffs': len(e.handoffs),
                'excavator_km': round(e.distance / 1000, 3),
                'hauler_km': round(sum(x.distance for x in self.haulers) / 1000, 3)}


def table(rows: List[Dict], cols=None):
    cols = cols or list(rows[0])
    w = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    print('  '.join(c.ljust(n) for c, n in zip(cols, w)))
    for r in rows:
        print('  '.join(str(r[c]).ljust(n) for c, n in zip(cols, w)))

# %% 7. Headline runs: both scenarios through ROS 2


MAIN_COLS = ['scenario', 'active_h', 'idle_h', 'idle_travel_h', 'idle_auction_h', 'idle_hauler_h',
             'active_pct', 'sites_completed', 'kg_collected', 'handoffs']
RUNS = {s: run_direct(replace(BASE, scenario=s), TERRAIN) for s in SCENARIOS}
table([RUNS[s].summary() for s in SCENARIOS], MAIN_COLS)
a = {s: RUNS[s].summary()['active_pct'] for s in SCENARIOS}
print(f"\nin-bay gain: {a['in_bay'] - a['dump_and_pour']:+.2f} percentage points of active time "
      f"({100 * (a['in_bay'] / a['dump_and_pour'] - 1):+.1f} % relative)")

# %% 8. Checks: each one prints PASS or raises


def check(name, ok, detail=''):
    if not ok:
        raise AssertionError(f'FAIL {name} {detail}')
    print(f'PASS  {name} {detail}')


for s, R in RUNS.items():
    e, P = R.ex, R.P
    check(f'[{s}] active + idle = {P.sim_hours} h', abs(sum(e.T.values()) - P.sim_hours * 3600) < 1e-6)
    held = e.held_kg + sum(h.load_kg for h in R.haulers)
    if len(e.handoffs) > sum(h.trips for h in R.haulers):   # handed over in the very last second,
        held += e.handoffs[-1]['kg']                          # hauler hadn't picked it up yet
    check(f'[{s}] mass balance', abs(e.kg_collected - (sum(h.delivered_kg for h in R.haulers) + held)) < 1e-6,
          f'collected {e.kg_collected:.1f} = delivered + still carried')
    in_progress = e.collect_done_s if (e.site is not None and e.collect_done_s < P.collect_s - 1e-9) else 0.0
    check(f'[{s}] active time = 15 min x completed sites + current site progress',
          abs(e.T['active'] - (P.collect_s * e.sites_completed + in_progress)) < 1e-6,
          f"({e.T['active']:.1f} s, {e.sites_completed} sites)")
    lat = [hd['t_assigned'] - hd['t_open'] for hd in e.handoffs]
    check(f'[{s}] every auction lasted >= {P.auction_s / 60:.1f} min', all(x >= P.auction_s - 1e-6 for x in lat),
          f'(min {min(lat) / 60:.2f}, max {max(lat) / 60:.2f} min)' if lat else '')
    if s == 'dump_and_pour':  # idle while waiting = auction part + hauler-drive part, per hand-off
        auction_sum = sum(lat)
        if e.request:                                   # a hand-off still in progress at 16 h
            r_ = e.request
            auction_sum += (r_['t_assigned'] if r_['hauler'] else P.sim_hours * 3600) - r_['t_open']
        check(f'[{s}] idle_auction = sum of auction times', abs(e.T['idle_auction'] - auction_sum) <= len(lat) * P.dt + P.dt,
              f"({e.T['idle_auction'] / 60:.1f} vs {auction_sum / 60:.1f} min)")
    if s == 'in_bay':
        check(f'[{s}] containers never over capacity', all(hd['kg'] <= P.container_kg + 1e-6 for hd in e.handoffs))
    if R.terrain is not None:
        m = R.terrain
        bad = sum(bool(m.hazard[m.cell(*p)]) for rv in [e] + R.haulers for p in rv.track)
        check(f'[{s}] no rover ever on hazard terrain', bad == 0, f'({sum(len(r.track) for r in [e] + R.haulers)} samples)')


# Arithmetic cross-check (flat ground, straight lines): dump-and-pour auction idle should equal
# hand-offs x 9.3 min, the same quantity as the abstract's 35 x 9 min estimate.
flat = run_direct(replace(BASE, scenario='dump_and_pour', use_terrain=False))
fs = flat.summary()
check('[flat dump_and_pour] idle_auction = hand-offs x auction time',
      abs(fs['idle_auction_h'] * 60 - fs['handoffs'] * BASE.auction_s / 60) <= BASE.auction_s / 60 + 1e-6,
      f"({fs['idle_auction_h'] * 60:.1f} min vs {fs['handoffs']} x {BASE.auction_s / 60:.1f} min; "
      f"the last one may still be open at 16 h)")
print(f"   -> auction idle share {100 * fs['idle_auction_h'] / BASE.sim_hours:.1f} % of 16 h "
      f"(abstract arithmetic: 35 x 9 min = 32.8 %)")

# %% 9. Figures


def figures(runs: Dict[str, Result], out_dir='inbay_results'):
    import matplotlib
    if 'ipykernel' not in sys.modules:          # plain `python3 inbay_study.py`: just save PNGs
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    colors = {'active': '#2a9d8f', 'idle_travel': '#e9c46a', 'idle_auction': '#e76f51', 'idle_hauler': '#9c6644'}
    labels = {'active': 'active (collecting)', 'idle_travel': 'idle: travel between sites',
              'idle_auction': 'idle: auction open', 'idle_hauler': 'idle: hauler driving over'}

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={'width_ratios': [1, 2.2]})
    for i, s in enumerate(SCENARIOS):                      # (a) stacked hours
        bottom = 0.0
        for k in ['active', *IDLE_KEYS]:
            v = runs[s].ex.T[k] / 3600
            ax[0].bar(i, v, bottom=bottom, color=colors[k], label=labels[k] if i == 0 else None)
            if v > 0.25:
                ax[0].text(i, bottom + v / 2, f'{v:.2f} h', ha='center', va='center', fontsize=8)
            bottom += v
    ax[0].set_xticks(range(len(SCENARIOS)), [s.replace('_', ' ') for s in SCENARIOS])
    ax[0].set_ylabel('hours of the 16 h block')
    ax[0].set_title('Excavator time budget')
    ax[0].legend(fontsize=7, loc='lower right')
    for i, s in enumerate(SCENARIOS):                      # (b) timeline strips
        tl = runs[s].ex.timeline + [(runs[s].P.sim_hours * 3600, None)]
        for (t0, c), (t1, _) in zip(tl, tl[1:]):
            ax[1].barh(i, (t1 - t0) / 3600, left=t0 / 3600, color=colors[c], height=0.6)
    ax[1].set_yticks(range(len(SCENARIOS)), [s.replace('_', ' ') for s in SCENARIOS])
    ax[1].set_xlabel('mission time [h]')
    ax[1].set_xlim(0, BASE.sim_hours)
    ax[1].set_title('Excavator activity over the 16 h block')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'time_budget.png'), dpi=150)

    fig2, axs = plt.subplots(1, len(SCENARIOS), figsize=(12, 6))  # (c) routes on the hazard map
    for a_, s in zip(axs, SCENARIOS):
        R = runs[s]
        m = R.terrain
        if m is not None:
            a_.imshow(np.where(m.hazard, 2, np.where(m.blocked, 1, 0)), origin='lower', cmap='Greys',
                      vmin=0, vmax=2, extent=[m.ox, m.ox + m.w * m.res, m.oy, m.oy + m.h * m.res])
        for h in R.haulers:
            a_.plot(*zip(*h.track), color='tab:orange', lw=0.6, alpha=0.7)
        a_.plot(*zip(*R.ex.track), color='tab:blue', lw=1.0)
        hw = R.P.field_half_width
        a_.set_xlim(-hw, hw)
        a_.set_ylim(-hw, hw)
        a_.set_aspect('equal')
        a_.set_title(f"{s.replace('_', ' ')}: {R.ex.sites_completed} sites, "
                     f"{R.ex.kg_collected:.0f} kg (blue excavator, orange hauler)", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'routes.png'), dpi=130)
    print('saved', os.path.join(out_dir, 'time_budget.png'), 'and routes.png')
    return fig, fig2


figures(RUNS)

# %% 10. Sweep: auction open time 0-20 min x seeds (direct runner; identical to ROS 2, see cell 8)

SWEEP_AUCTION_MIN = [0, 2.5, 5, 7.5, 9.3, 12.5, 15, 17.5, 20]
SWEEP_SEEDS = [1, 2, 3, 4, 5]
t0 = time.time()
SWEEP = [run_direct(replace(BASE, scenario=s, auction_s=a_min * 60, seed=sd), TERRAIN).summary()
         for a_min in SWEEP_AUCTION_MIN for s in SCENARIOS for sd in SWEEP_SEEDS]
print(f'{len(SWEEP)} runs in {time.time() - t0:.0f} s')


def sweep_figure(rows, out_dir='inbay_results'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for s, c in zip(SCENARIOS, ['#e76f51', '#2a9d8f']):
        m_ = [np.mean([r['active_pct'] for r in rows if r['scenario'] == s and r['auction_min'] == a_m])
              for a_m in SWEEP_AUCTION_MIN]
        vals = [[r['active_pct'] for r in rows if r['scenario'] == s and r['auction_min'] == a_m]
                for a_m in SWEEP_AUCTION_MIN]
        lo, hi = [min(v) for v in vals], [max(v) for v in vals]
        ax[0].plot(SWEEP_AUCTION_MIN, m_, 'o-', color=c, label=s.replace('_', ' '))
        ax[0].fill_between(SWEEP_AUCTION_MIN, lo, hi, color=c, alpha=0.2)
    gain = []
    for a_m in SWEEP_AUCTION_MIN:
        g = [r_i['active_pct'] - r_d['active_pct']
             for r_i in rows if r_i['scenario'] == 'in_bay' and r_i['auction_min'] == a_m
             for r_d in rows if r_d['scenario'] == 'dump_and_pour' and r_d['auction_min'] == a_m
             and r_d['seed'] == r_i['seed']]
        gain.append((np.mean(g), min(g), max(g)))
    ax[1].plot(SWEEP_AUCTION_MIN, [g[0] for g in gain], 'o-', color='k')
    ax[1].fill_between(SWEEP_AUCTION_MIN, [g[1] for g in gain], [g[2] for g in gain], color='k', alpha=0.15)
    for a_ in ax:
        a_.axvline(9.3, color='grey', ls='--', lw=1)
        a_.set_xlabel('auction open time [min]  (dashed: 9.3 min, Kashid 2024 Policy 1)')
    ax[0].set_ylabel('excavator active time [% of 16 h]')
    ax[0].legend()
    ax[1].set_ylabel('in-bay gain [percentage points]')
    ax[0].set_title(f'Active time (mean, band = min-max over {len(SWEEP_SEEDS)} seeds)')
    ax[1].set_title('In-bay minus dump-and-pour (paired by seed)')
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'auction_sweep.png'), dpi=150)
    print('saved', os.path.join(out_dir, 'auction_sweep.png'))
    return fig


sweep_figure(SWEEP)
table([r for r in SWEEP if r['auction_min'] == 9.3], ['scenario', 'seed'] + MAIN_COLS[1:])

# %% 11. Export everything to CSV (for the paper's tables)


def export(runs, sweep, out_dir='inbay_results'):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'params.json'), 'w') as f:
        json.dump(asdict(BASE), f, indent=2)
    for name, rows in [('headline_runs', [runs[s].summary() for s in SCENARIOS]), ('auction_sweep', sweep)]:
        with open(os.path.join(out_dir, f'{name}.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for s in SCENARIOS:
        rows = runs[s].ex.handoffs
        if rows:
            with open(os.path.join(out_dir, f'handoffs_{s}.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print('exported to', os.path.abspath(out_dir))


export(RUNS, SWEEP)
