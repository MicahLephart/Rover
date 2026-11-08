"""Build the terrain z-height grid that hazard avoidance uses.

Reads your terrain the same way Gazebo does (the <collision> geometry in its model.sdf) and
samples the TOP surface height on a regular grid. Supports:
    * <mesh>       .obj and .stl built in;  .dae/.glb/.ply/... if `pip install trimesh pycollada`
    * <heightmap>  grayscale image (PNG 8/16-bit) with <size> and <pos>
You can also point it straight at a mesh exported from Blender.

    ros2 run moon_rover_bridge build_hazard_map ~/gazebo_models/lunar_surface0 \
        -o ~/ros2_ws/src/moon_rover_bridge/config/maps/lunar_surface0.npz

    # try thresholds and look at the preview PNG before putting them in your scenario yaml
    ros2 run moon_rover_bridge build_hazard_map terrain.obj -o map.npz --z-min -0.6 --z-max 0.8 --max-slope 15

It prints z statistics so you can choose hazard_z_min / hazard_z_max, and writes
<output>.png (needs matplotlib) showing height, slope and the blocked cells.
"""

import argparse
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

import numpy as np

from moon_rover_bridge.terrain_map import HazardMap


# ============================================================================ mesh loading
def load_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    with open(path, errors='replace') as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(v) for v in line.split()[1:4]])
            elif line.startswith('f '):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split('/')[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):        # fan-triangulate polygons
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.array(verts, float), np.array(faces, np.int64)


def load_stl(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, 'rb') as f:
        data = f.read()
    tris = []
    n = struct.unpack('<I', data[80:84])[0] if len(data) >= 84 else 0
    if len(data) == 84 + 50 * n:                         # binary STL
        arr = np.frombuffer(data, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)),
                                                  ('a', '<u2')]), count=n, offset=84)
        tris = arr['v'].astype(float)
    else:                                                # ascii STL
        vs = [list(map(float, ln.split()[1:4])) for ln in data.decode(errors='replace').splitlines()
              if ln.strip().startswith('vertex')]
        tris = np.array(vs, float).reshape(-1, 3, 3)
    tris = np.asarray(tris, float)
    return tris.reshape(-1, 3), np.arange(len(tris) * 3).reshape(-1, 3)


def load_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.obj':
        return load_obj(path)
    if ext == '.stl':
        return load_stl(path)
    try:
        import trimesh
    except ImportError:
        sys.exit(f'{ext} needs trimesh:  pip install trimesh pycollada   '
                 f'(or export the terrain from Blender as .obj)')
    m = trimesh.load(path, force='mesh')
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


# ============================================================================ rasterising
def rasterize_top_surface(V: np.ndarray, F: np.ndarray, res: float,
                          bounds: Optional[Tuple[float, float, float, float]] = None):
    """Highest surface z at each cell centre (NaN where no triangle covers the cell)."""
    x0, y0, x1, y1 = bounds or (V[:, 0].min(), V[:, 1].min(), V[:, 0].max(), V[:, 1].max())
    nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
    z = np.full(ny * nx, -np.inf)
    T = V[F]                                              # (T, 3, 3)

    def splat(cols, rows, zs):
        ok = (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)
        np.maximum.at(z, rows[ok] * nx + cols[ok], zs[ok])

    # vertices first (catches triangles smaller than a cell)
    splat(np.floor((V[:, 0] - x0) / res).astype(int), np.floor((V[:, 1] - y0) / res).astype(int), V[:, 2])

    tx, ty = T[:, :, 0], T[:, :, 1]
    c_lo = np.ceil((tx.min(1) - x0) / res - 0.5).astype(int)
    c_hi = np.floor((tx.max(1) - x0) / res - 0.5).astype(int)
    r_lo = np.ceil((ty.min(1) - y0) / res - 0.5).astype(int)
    r_hi = np.floor((ty.max(1) - y0) / res - 0.5).astype(int)
    span = np.maximum(c_hi - c_lo, r_hi - r_lo) + 1
    S = 8
    small = np.nonzero((span > 0) & (span <= S))[0]
    big = np.nonzero(span > S)[0]

    def bary(idx, cols, rows):
        px = x0 + (cols + 0.5) * res
        py = y0 + (rows + 0.5) * res
        # idx: (B,), cols/rows: (B, K) -> vertex coords as (B, 1) columns
        a, b, c = T[idx, 0], T[idx, 1], T[idx, 2]
        ax, ay, az = a[:, 0:1], a[:, 1:2], a[:, 2:3]
        bx, by, bz = b[:, 0:1], b[:, 1:2], b[:, 2:3]
        cx, cy, cz = c[:, 0:1], c[:, 1:2], c[:, 2:3]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        with np.errstate(divide='ignore', invalid='ignore'):
            l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
            l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
        l3 = 1 - l1 - l2
        e = -1e-9
        inside = (l1 >= e) & (l2 >= e) & (l3 >= e) & (np.abs(det) > 1e-15)
        return inside, l1 * az + l2 * bz + l3 * cz

    off = np.arange(S)
    for k in range(0, len(small), 20000):                # batches of small triangles
        idx = small[k:k + 20000]
        cols = (c_lo[idx, None, None] + off[None, None, :]) + 0 * off[None, :, None]
        rows = (r_lo[idx, None, None] + off[None, :, None]) + 0 * off[None, None, :]
        cols, rows = cols.reshape(len(idx), -1), rows.reshape(len(idx), -1)
        inside, zz = bary(idx, cols, rows)
        inside &= (cols <= c_hi[idx, None]) & (rows <= r_hi[idx, None])
        splat(cols[inside], rows[inside], zz[inside])

    for i in big:                                        # few large triangles, one at a time
        cc, rr = np.meshgrid(np.arange(c_lo[i], c_hi[i] + 1), np.arange(r_lo[i], r_hi[i] + 1))
        cc, rr = cc.ravel(), rr.ravel()
        inside, zz = bary(np.full(1, i), cc[None, :], rr[None, :])
        splat(cc[inside[0]], rr[inside[0]], zz[0][inside[0]])

    z[~np.isfinite(z)] = np.nan
    return z.reshape(ny, nx), x0, y0


def heightmap_grid(img_path: str, size, pos, res: float):
    """Gazebo <heightmap>: image covers size_x * size_y centred on pos; white = pos_z + size_z."""
    from PIL import Image
    im = np.array(Image.open(img_path))
    if im.ndim == 3:
        im = im[..., :3].mean(axis=2)
    im = im.astype(float)
    maxv = 65535.0 if im.max() > 255 else 255.0
    hgt = im / maxv * size[2] + pos[2]
    hgt = hgt[::-1, :]                                   # image top row = +y  (CHECK the preview!)
    x0, y0 = pos[0] - size[0] / 2, pos[1] - size[1] / 2
    nx, ny = int(math.ceil(size[0] / res)), int(math.ceil(size[1] / res))
    xs = (np.arange(nx) + 0.5) * res / size[0] * (im.shape[1] - 1)
    ys = (np.arange(ny) + 0.5) * res / size[1] * (im.shape[0] - 1)
    # bilinear sample
    xi, yi = np.clip(xs.astype(int), 0, im.shape[1] - 2), np.clip(ys.astype(int), 0, im.shape[0] - 2)
    fx, fy = (xs - xi)[None, :], (ys - yi)[:, None]
    h00, h01 = hgt[yi][:, xi], hgt[yi][:, xi + 1]
    h10, h11 = hgt[yi + 1][:, xi], hgt[yi + 1][:, xi + 1]
    z = (h00 * (1 - fx) + h01 * fx) * (1 - fy) + (h10 * (1 - fx) + h11 * fx) * fy
    return z, x0, y0


# ============================================================================ SDF
def _vec(el, n, default):
    return [float(v) for v in el.text.split()][:n] if el is not None and el.text else default


def resolve_uri(uri: str, model_dir: str) -> str:
    uri = uri.strip()
    if uri.startswith('file://'):
        return uri[7:]
    if uri.startswith('model://'):
        rest = uri[8:]
        name, _, sub = rest.partition('/')
        roots = [os.path.dirname(os.path.abspath(model_dir))]
        roots += [p for p in os.environ.get('GZ_SIM_RESOURCE_PATH', '').split(':') if p]
        for root in roots:
            p = os.path.join(root, name, sub)
            if os.path.isfile(p):
                return p
        if os.path.basename(os.path.abspath(model_dir)) == name:
            return os.path.join(model_dir, sub)
        raise FileNotFoundError(f'Cannot resolve {uri} (searched {roots})')
    return uri if os.path.isabs(uri) else os.path.join(model_dir, uri)


def grid_from_model(model: str, res: float):
    model_sdf = os.path.join(model, 'model.sdf') if os.path.isdir(model) else model
    model_dir = os.path.dirname(os.path.abspath(model_sdf))
    root = ET.parse(model_sdf).getroot()
    col = root.find('.//collision')
    if col is None:
        sys.exit(f'No <collision> in {model_sdf}')
    pose = _vec(col.find('pose'), 6, [0.0] * 6)
    link_pose = _vec(root.find('.//link/pose'), 6, [0.0] * 6)
    off = [pose[i] + link_pose[i] for i in range(3)]
    if any(abs(v) > 1e-9 for v in pose[3:] + link_pose[3:]):
        print('WARNING: collision/link has a rotation; only the xyz offset is applied.')
    hm = col.find('.//heightmap')
    if hm is not None:
        img = resolve_uri(hm.find('uri').text, model_dir)
        size = _vec(hm.find('size'), 3, [1.0, 1.0, 1.0])
        p = _vec(hm.find('pos'), 3, [0.0, 0.0, 0.0])
        print(f'heightmap {img}  size {size}  pos {p}')
        z, x0, y0 = heightmap_grid(img, size, [p[i] + off[i] for i in range(3)], res)
        return z, x0, y0
    mesh = col.find('.//mesh')
    if mesh is None:
        sys.exit('Collision geometry is neither <mesh> nor <heightmap>.')
    path = resolve_uri(mesh.find('uri').text, model_dir)
    scale = _vec(mesh.find('scale'), 3, [1.0, 1.0, 1.0])
    print(f'mesh {path}  scale {scale}  offset {off}')
    return grid_from_mesh(path, res, scale, off, up='z')


def grid_from_mesh(path: str, res: float, scale=(1, 1, 1), offset=(0, 0, 0), up='auto'):
    V, F = load_mesh(path)
    if up == 'auto':                                     # terrain: up axis has the smallest extent
        ext = V.max(0) - V.min(0)
        up = 'xyz'[int(np.argmin(ext))]
        print(f'auto up-axis: {up}  (mesh extents {np.round(ext, 2).tolist()})')
    if up == 'y':                                        # Blender's default OBJ export is Y-up
        V = np.c_[V[:, 0], -V[:, 2], V[:, 1]]
    elif up == 'x':
        V = np.c_[V[:, 1], V[:, 2], V[:, 0]]
    V = V * np.asarray(scale, float) + np.asarray(offset, float)
    print(f'{len(V)} vertices, {len(F)} triangles, '
          f'x [{V[:, 0].min():.1f}, {V[:, 0].max():.1f}]  y [{V[:, 1].min():.1f}, {V[:, 1].max():.1f}]  '
          f'z [{V[:, 2].min():.2f}, {V[:, 2].max():.2f}]')
    return rasterize_top_surface(V, F, res)


# ============================================================================ report
def report(m: HazardMap, out_png: Optional[str], field_limit: float):
    z = m.z[np.isfinite(m.z)]
    pct = np.percentile(z, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    print('\nz statistics [m]:  ' + '  '.join(f'p{q}={v:.3f}' for q, v in zip(qs, pct)))
    s = m.slope_deg[np.isfinite(m.slope_deg)]
    sp = np.percentile(s, [50, 90, 95, 99])
    print('slope [deg]:       ' + '  '.join(f'p{q}={v:.1f}' for q, v in zip([50, 90, 95, 99], sp)))
    print(f'grid {m.w} x {m.h} cells @ {m.res} m, origin ({m.ox:.2f}, {m.oy:.2f}); '
          f'free (after inflation) {100 * m.free_fraction():.1f} %')
    if not out_png:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('(matplotlib not installed: no preview PNG)')
        return
    ext = [m.ox, m.ox + m.w * m.res, m.oy, m.oy + m.h * m.res]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    for a, data, title, cmap in [(ax[0], m.z, 'height z [m]', 'terrain'),
                                 (ax[1], m.slope_deg, 'slope [deg]', 'magma')]:
        im = a.imshow(data, origin='lower', extent=ext, cmap=cmap)
        fig.colorbar(im, ax=a, shrink=0.85)
        a.set_title(title)
    shade = np.where(m.hazard, 2, np.where(m.blocked, 1, 0))
    ax[2].imshow(shade, origin='lower', extent=ext, cmap='Greys', vmin=0, vmax=2)
    ax[2].set_title('blocked: black = hazard, grey = safety margin')
    for a in ax:
        a.add_patch(plt.Rectangle((-field_limit, -field_limit), 2 * field_limit, 2 * field_limit,
                                  fill=False, ec='tab:blue', lw=1, ls='--'))
        a.set_xlabel('x [m]')
        a.set_ylabel('y [m]')
        a.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print(f'preview: {out_png}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('source', help='model dir / model.sdf, or a mesh file (.obj .stl .dae ...)')
    ap.add_argument('-o', '--output', required=True, help='output .npz')
    ap.add_argument('--res', type=float, default=0.5, help='grid cell size [m] (default 0.5)')
    ap.add_argument('--up', choices=['auto', 'x', 'y', 'z'], default='auto',
                    help='up axis for a bare mesh file (Blender OBJ export is usually y)')
    ap.add_argument('--scale', type=float, nargs=3, default=[1, 1, 1])
    ap.add_argument('--offset', type=float, nargs=3, default=[0, 0, 0], help='terrain pose in the world')
    # preview thresholds (the real ones live in your scenario yaml)
    ap.add_argument('--z-min', type=float, default=-1000.0)
    ap.add_argument('--z-max', type=float, default=1000.0)
    ap.add_argument('--max-slope', type=float, default=90.0)
    ap.add_argument('--inflation', type=float, default=1.0)
    ap.add_argument('--field-half-width', type=float, default=80.0)
    ap.add_argument('--margin', type=float, default=5.0)
    a = ap.parse_args(argv)

    src = os.path.expanduser(a.source)
    if os.path.isdir(src) or src.endswith('.sdf'):
        z, x0, y0 = grid_from_model(src, a.res)
    else:
        z, x0, y0 = grid_from_mesh(src, a.res, a.scale, a.offset, a.up)

    m = HazardMap(z, x0, y0, a.res)
    out = os.path.expanduser(a.output)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    m.save(out)
    print(f'saved {out}')
    limit = a.field_half_width - a.margin
    m.configure(a.z_min, a.z_max, a.max_slope, a.inflation, field_limit_m=limit)
    report(m, os.path.splitext(out)[0] + '.png', limit)
    return 0


if __name__ == '__main__':
    sys.exit(main())
