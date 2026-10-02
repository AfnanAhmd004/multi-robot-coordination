import numpy as np

from mrc import MovingObstacle, Rect, Robot, Task, World, allocate, drive, path_length, robots_needed, rrt_star, warehouse


def empty_world(**kw):
    return World(20, 20, **kw)


def test_rrt_star_avoids_wall():
    w = empty_world(shelves=[Rect(9, 0, 11, 16)])
    path = rrt_star(w, (2, 2), (18, 2), seed=0)
    assert path is not None
    assert all(w.segment_free(a, b) for a, b in zip(path[:-1], path[1:]))
    assert max(p[1] for p in path) > 16  # had to go around the top of the wall


def test_rrt_star_is_near_straight_in_free_space():
    path = rrt_star(empty_world(), (1, 1), (19, 19), seed=1)
    assert path_length(path) < 1.05 * np.hypot(18, 18)


def test_team_size_from_load():
    assert robots_needed(Task(0, np.zeros(2), np.ones(2), 35), 20) == 2
    assert robots_needed(Task(0, np.zeros(2), np.ones(2), 5), 20) == 1


def test_allocation_picks_nearest_and_defers_when_short():
    robots = [Robot(0, np.array([0.0, 0.0])), Robot(1, np.array([10.0, 10.0]))]
    tasks = [Task(0, np.array([9.0, 9.0]), np.zeros(2), 10), Task(1, np.array([1.0, 1.0]), np.zeros(2), 50)]
    teams, deferred = allocate(tasks, robots)
    assert [t.id for t in deferred] == [1]  # 50 kg needs 3 robots, only 2 exist
    assert teams[0].leader.id == 1


def test_drive_reaches_goal_with_moving_obstacle():
    w = empty_world(movers=[MovingObstacle(np.array([10.0, 10.0]), np.array([0.0, 0.5]))])
    stats = drive(w, (2, 10), (18, 10), seed=0)
    assert stats.reached
    assert stats.min_clearance > 0  # never inside an obstacle


def test_warehouse_has_free_aisles():
    w = warehouse()
    assert w.free(np.array([8.0, 10.0]), include_movers=False)
    assert not w.free(np.array([5.5, 10.0]), include_movers=False)
