"""Unit tests for mission_core.py (no ROS / Gazebo needed).

    cd ~/ros2_ws/src/moon_rover_bridge && python3 -m pytest test/test_mission_core.py -v
"""

import math
import os
import random

import numpy as np
import pytest

from moon_rover_bridge.mission_core import (
    CollectorConfig, CollectorMission, CollectorState, HaulerConfig, HaulerMission,
    HaulerState, SiteGenerator)
from moon_rover_bridge.scenario_runner import run_scenario
from moon_rover_bridge.terrain_map import HazardMap

SCEN = os.path.join(os.path.dirname(__file__), '..', 'config', 'scenarios')
DT = 0.1


def drive(m: CollectorMission, aligned_after_s=None, v=0.27, until=None, track=None):
    """Teleport-free point-mass drive of the collector; optionally a 'hauler' that is
    aligned aligned_after_s seconds after each exchange request. Returns final (x, y)."""
    x = y = t = 0.0
    req_t = None
    until = until or m.cfg.mission_duration_s + 1
    while m.state != CollectorState.DONE and t < until:
        aligned = None
        if m.exchange_request_id is not None and aligned_after_s is not None:
            req_t = t if req_t is None else req_t
            if t - req_t >= aligned_after_s:
                aligned = m.exchange_request_id
        else:
            req_t = None
        m.update(t, x, y, 0.0, aligned)
        if m.goal:
            dx, dy = m.goal[0] - x, m.goal[1] - y
            d = math.hypot(dx, dy)
            step = min(v * DT, d)
            if d > 0:
                x, y = x + step * dx / d, y + step * dy / d
        if track is not None:
            track.append((x, y))
        t += DT
    return x, y


def times_of(m, event):
    return [e.mission_time_s for e in m.events if e.event == event]


# ------------------------------------------------------------------ site generation
@pytest.mark.parametrize('half_width,margin', [(80.0, 5.0), (80.0, 0.0), (160.0, 0.0)])
def test_sites_exact_radius_and_in_bounds(half_width, margin):
    lim = half_width - margin
    for seed in range(50):
        g = SiteGenerator(half_width, random.Random(seed), margin)
        x, y, _ = g.point_at_distance(0, 0, 100.0)
        assert math.hypot(x, y) == pytest.approx(100.0)
        assert abs(x) <= lim + 1e-6 and abs(y) <= lim + 1e-6
        for _ in range(200):
            nx, ny, _ = g.point_at_distance(x, y, 50.0)
            assert math.hypot(nx - x, ny - y) == pytest.approx(50.0)
            assert abs(nx) <= lim + 1e-6 and abs(ny) <= lim + 1e-6
            x, y = nx, ny


def test_corner_start_still_finds_site():
    g = SiteGenerator(80.0, random.Random(0), 5.0)
    for _ in range(100):
        x, y, _ = g.point_at_distance(75.0, 75.0, 50.0)
        assert x <= 75.0 + 1e-6 and y <= 75.0 + 1e-6


def test_same_seed_same_sites():
    a = CollectorMission(CollectorConfig(seed=7, mission_duration_s=6000))
    b = CollectorMission(CollectorConfig(seed=7, mission_duration_s=6000))
    drive(a, aligned_after_s=30)
    drive(b, aligned_after_s=30)
    assert [e.detail for e in a.events] == [e.detail for e in b.events]


# ------------------------------------------------------------------ collector timing
def test_first_leg_and_timer_sum():
    m = CollectorMission(CollectorConfig(seed=1))
    drive(m, aligned_after_s=60)
    arrive = times_of(m, 'ARRIVED_START_COLLECTION')
    # 100 m minus the 0.25 m arrival tolerance at 0.27 m/s (this test rover has no turning)
    assert arrive[0] == pytest.approx((100.0 - 0.25) / 0.27, abs=0.3)   # ~369.4 s
    t = m.timers
    assert t.active_s + t.idle_s == pytest.approx(86400.0, abs=1e-6)
    assert m.mission_time_s == pytest.approx(86400.0)
    assert m.state == CollectorState.DONE


def test_each_site_collects_exactly_15_min_active_and_buffer_is_idle():
    cfg = CollectorConfig(seed=3, exchange_buffer_s=120.0, mission_duration_s=20000)
    m = CollectorMission(cfg)
    drive(m, aligned_after_s=200)          # hauler shows up mid-collection
    arr = times_of(m, 'ARRIVED_START_COLLECTION')
    done = times_of(m, 'COLLECTION_DONE_DUMP_TO_INBAY')
    ex_s = times_of(m, 'HAULER_ALIGNED_START_EXCHANGE')
    ex_e = times_of(m, 'EXCHANGE_COMPLETE')
    assert len(done) >= 3
    for i in range(len(done)):
        assert ex_s[i] - arr[i] == pytest.approx(200.0, abs=0.3)
        assert ex_e[i] - ex_s[i] == pytest.approx(120.0, abs=0.3)
        # wall span at site = 15 min collecting + 2 min exchange pause
        assert done[i] - arr[i] == pytest.approx(900.0 + 120.0, abs=0.3)
    assert m.timers.idle_wait_hauler_s == 0.0


def test_collector_waits_if_hauler_is_late():
    cfg = CollectorConfig(seed=4, exchange_buffer_s=60.0, mission_duration_s=2000)  # 1 wait
    m = CollectorMission(cfg)
    drive(m, aligned_after_s=1000)         # hauler arrives 100 s after collection ends
    assert 'WAITING_FOR_HAULER' in [e.event for e in m.events]
    assert m.timers.idle_wait_hauler_s == pytest.approx(100.0, abs=0.3)
    # after the exchange it must leave for the next site
    i = [e.event for e in m.events].index('EXCHANGE_COMPLETE')
    assert m.events[i + 1].event == 'DEPART_TO_NEXT_SITE'


def test_no_hauler_ever_means_stuck_waiting_and_counted_idle():
    m = CollectorMission(CollectorConfig(seed=5, mission_duration_s=5000))
    drive(m, aligned_after_s=None)
    assert m.sites_reached == 1
    assert m.timers.idle_wait_hauler_s == pytest.approx(5000 - 370.4 - 900, abs=1.0)


def test_exchanges_disabled():
    m = CollectorMission(CollectorConfig(seed=6, exchange_every_n_sites=0, mission_duration_s=10000))
    drive(m)
    assert m.exchanges_completed == 0 and m.timers.idle_exchange_s == 0.0
    assert m.timers.idle_s == pytest.approx(m.timers.idle_travel_s)


def test_every_third_site():
    m = CollectorMission(CollectorConfig(seed=8, exchange_every_n_sites=3, mission_duration_s=20000))
    drive(m, aligned_after_s=10)
    req = [e.detail for e in m.events if e.event == 'EXCHANGE_REQUESTED']
    assert req[:2] == ['request #3', 'request #6']


def test_zero_buffer():
    m = CollectorMission(CollectorConfig(seed=9, exchange_buffer_s=0.0, mission_duration_s=5000))
    drive(m, aligned_after_s=10)
    assert m.exchanges_completed >= 2
    assert m.timers.idle_exchange_s == 0.0


def test_paused_clock_counts_nothing():
    m = CollectorMission(CollectorConfig(seed=1))
    m.update(10.0, 0, 0)
    for _ in range(5):
        m.update(10.0, 0, 0)                # sim paused: same time repeatedly
    m.update(9.0, 0, 0)                     # clock went backwards (reset)
    assert m.mission_time_s == 0.0


# ------------------------------------------------------------------ hauler
def test_hauler_parks_left_of_collector_same_heading():
    h = HaulerMission(HaulerConfig(align_offset_m=2.0))
    assert h.park_pose((10.0, 5.0, 0.0)) == pytest.approx((10.0, 7.0, 0.0))
    assert h.park_pose((10.0, 5.0, math.pi / 2)) == pytest.approx((8.0, 5.0, math.pi / 2))


def test_hauler_cycle():
    h = HaulerMission(HaulerConfig())
    st = {'state': 'COLLECTING', 'exchange_request_id': 1, 'last_completed_request_id': None,
          'site_pose': [10.0, 0.0, 0.0]}
    c = h.update(0.0, 2.0, 0.0, st)
    assert h.state == HaulerState.TO_COLLECTOR and c.kind == 'goal' and (c.x, c.y) == (10.0, 2.0)
    c = h.update(10.0, 2.0, 1.0, st)        # arrived, wrong heading
    assert h.state == HaulerState.ALIGNING and c.kind == 'rotate' and c.yaw_error == pytest.approx(-1.0)
    c = h.update(10.0, 2.0, 0.05, st)
    assert h.state == HaulerState.ALIGNED and h.aligned_id() == 1
    st.update(exchange_request_id=None, last_completed_request_id=1)
    c = h.update(10.0, 2.0, 0.0, st)
    assert h.state == HaulerState.RETURNING and h.trips == 1
    st.update(exchange_request_id=2, site_pose=[50.0, 0.0, 0.0])
    h.update(5.0, 2.0, 0.0, st)             # new request while returning: must go home first
    assert h.state == HaulerState.RETURNING
    h.update(0.0, 2.0, 0.0, st)
    h.update(0.0, 2.0, 0.0, st)
    assert h.state == HaulerState.TO_COLLECTOR and h.request_id == 2


def test_hauler_goes_home_when_mission_ends():
    h = HaulerMission(HaulerConfig())
    st = {'state': 'COLLECTING', 'exchange_request_id': 4, 'last_completed_request_id': 3,
          'site_pose': [10.0, 0.0, 0.0]}
    h.update(0.0, 2.0, 0.0, st)
    st['state'] = 'DONE'
    h.update(5.0, 2.0, 0.0, st)
    assert h.state == HaulerState.RETURNING


# ------------------------------------------------------------------ end-to-end offline
def test_offline_baseline_scenario():
    name, col, hau, crov, _ = run_scenario(os.path.join(SCEN, 'baseline.yaml'), dt=0.2)
    t = col.timers
    assert name == 'baseline'
    assert t.active_s + t.idle_s == pytest.approx(86400.0, abs=1e-6)
    # every completed site gave 900 s of active time (last site may be partial)
    assert t.active_s == pytest.approx(900.0 * col.loads_dumped + col.collect_progress_s, abs=30)
    # hauler did a round trip per completed exchange
    assert hau.trips == col.exchanges_completed or hau.trips == col.exchanges_completed - 1


# ------------------------------------------------------------------ hazard avoidance
def crater_map(res=0.5):
    """Flat +/-80 m field with a 3 m deep crater at (40, 40) and a 2 m mound at (-45, -45)."""
    xc = -80 + (np.arange(int(160 / res)) + 0.5) * res
    X, Y = np.meshgrid(xc, xc)
    z = (-3 * np.exp(-((X - 40) ** 2 + (Y - 40) ** 2) / (2 * 8 ** 2))
         + 2 * np.exp(-((X + 45) ** 2 + (Y + 45) ** 2) / (2 * 5 ** 2)))
    return HazardMap(z, -80, -80, res).configure(-0.5, 0.5, 15.0, 1.0, field_limit_m=75.0)


def test_hazard_grid_marks_crater_and_mound():
    m = crater_map()
    assert not m.is_free(40, 40) and not m.is_free(-45, -45)
    assert m.is_free(0, 0) and m.is_free(-40, 40)
    assert not m.is_free(78, 0)                       # outside field margin


def test_planner_goes_around_crater():
    m = crater_map()
    res = m.plan((20.0, 20.0), (60.0, 60.0 - 0.1))   # straight line crosses the crater
    assert res is not None
    path, length = res
    pts = [(20.0, 20.0)] + path
    for a, b in zip(pts, pts[1:]):
        assert m.line_free(a, b, m.blocked)
    assert length > math.hypot(40, 40 - 0.1) + 1.0     # it had to detour
    assert m.plan((0, 0), (40, 40)) is None           # goal in the crater


def test_mission_never_enters_hazard():
    m = crater_map()
    cfg = CollectorConfig(seed=11, mission_duration_s=40000.0)
    col = CollectorMission(cfg, m)
    track = []
    drive(col, aligned_after_s=30, track=track)
    sites = [(e.x, e.y) for e in col.events if e.event == 'ARRIVED_START_COLLECTION']
    assert len(sites) >= 20
    assert all(not m.hazard[m.cell(x, y)] for x, y in track)          # never on hazard terrain
    for x, y in sites:
        assert m.is_free(x, y) and abs(x) <= 75 + 1e-6 and abs(y) <= 75 + 1e-6
    arr = [e for e in col.events if e.event in ('MISSION_START', 'DEPART_TO_NEXT_SITE')]
    assert col.timers.active_s + col.timers.idle_s == pytest.approx(40000.0)
    assert arr


def test_hauler_uses_safe_path_and_avoids_collector():
    m = crater_map()
    h = HaulerMission(HaulerConfig(base_x=0.0, base_y=2.0), m)
    st = {'state': 'COLLECTING', 'exchange_request_id': 1, 'last_completed_request_id': None,
          'site_pose': [60.0, 60.0, math.pi]}
    h.update(0.0, 2.0, 0.0, st)
    assert h.state == HaulerState.TO_COLLECTOR
    pts = [(0.0, 2.0)] + h.waypoints
    for a, b in zip(pts, pts[1:]):
        assert m.line_free(a, b, m.blocked)
    px, py = h.waypoints[-1]
    assert m.is_free(px, py) and math.hypot(px - 60, py - 60) == pytest.approx(2.0)


def test_offline_with_hazard_map(tmp_path):
    m = crater_map()
    mp = tmp_path / 'crater.npz'
    m.save(str(mp))
    import yaml
    doc = yaml.safe_load(open(os.path.join(SCEN, 'baseline.yaml')))
    doc['/**']['ros__parameters'].update(hazard_map_file=str(mp), hazard_z_min=-0.5, hazard_z_max=0.5)
    doc['collector_mission_node']['ros__parameters']['mission_duration_s'] = 20000.0
    sc = tmp_path / 'haz.yaml'
    yaml.safe_dump(doc, open(sc, 'w'))
    name, col, hau, crov, hrov = run_scenario(str(sc), dt=0.2)
    assert col.map is not None and col.sites_reached >= 10
    assert col.timers.active_s + col.timers.idle_s == pytest.approx(20000.0)
