"""Terrain height grid -> hazard grid -> A* paths.  NO ROS IMPORTS IN THIS FILE.

The z grid is built once from your terrain model with `build_hazard_map` and saved as .npz.
Each scenario then decides what counts as hazardous (in its YAML), so you can change the
thresholds without rebuilding the grid:

    hazard  =  z < hazard_z_min                (pits / crater floors)
            |  z > hazard_z_max                (mounds / rocks / ridges)
            |  slope > hazard_max_slope_deg    (crater walls, steep sides)
            |  no terrain under the cell       (off the edge of the mesh)
            |  outside the field (+/- field_half_width_m - boundary_margin_m)
    blocked =  hazard grown by hazard_inflation_m   (rover half-width + safety margin)

Grid convention: z[row, col], row 0 = lowest y, col 0 = lowest x. origin = world (x, y) of the
lower-left CORNER of cell [0, 0]; cell centres are origin + (i + 0.5) * resolution.
"""

import heapq
import math
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

XY = Tuple[float, float]


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


# ============================================================================ helpers
HAZARD_DEFAULTS = {
    'hazard_map_file': '',          # '' = no hazard avoidance, straight-line driving
    'hazard_z_min': -1000.0,        # m   below this = hazard (pits)
    'hazard_z_max': 1000.0,         # m   above this = hazard (mounds/rocks)
    'hazard_max_slope_deg': 90.0,   # deg steeper than this = hazard
    'hazard_inflation_m': 1.0,      # m   keep this far from any hazard
    'field_half_width_m': 80.0,     # field is [-80, 80] x [-80, 80]
    'boundary_margin_m': 5.0,       # stay this far inside the field edge
}


def resolve_map_file(name: str) -> str:
    """Absolute path, or a file name inside <moon_rover_bridge share>/config/maps/."""
    if not name or os.path.isabs(name) or os.path.isfile(name):
        return name
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory('moon_rover_bridge'), 'config', 'maps', name)
        if os.path.isfile(p):
            return p
    except Exception:  # noqa: BLE001  (no ROS environment)
        pass
    here = os.path.join(os.path.dirname(__file__), '..', 'config', 'maps', name)
    return os.path.abspath(here) if os.path.isfile(here) else name


def hazard_map_from_params(p: dict) -> Optional[HazardMap]:
    """None when hazard_map_file is empty. Raises FileNotFoundError if it is set but missing."""
    name = p.get('hazard_map_file', '')
    if not name:
        return None
    path = resolve_map_file(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'hazard_map_file "{name}" not found (looked for {path})')
    limit = float(p.get('field_half_width_m', 80.0)) - float(p.get('boundary_margin_m', 5.0))
    return HazardMap.load(path).configure(
        float(p.get('hazard_z_min', -1000.0)), float(p.get('hazard_z_max', 1000.0)),
        float(p.get('hazard_max_slope_deg', 90.0)), float(p.get('hazard_inflation_m', 1.0)),
        field_limit_m=limit)
