"""Shared geometric initialization/repair, not a full cost-optimal tower solver.

8-neighbour raster A* avoids forbidden cells and corner cutting. Its output is
compressed to visible tower spans. Both evolutionary methods use this same
initializer, including its runtime. It never calls the objective evaluator.
"""

import heapq
import math
import random
import numpy as np
from routing_geometry import segment_is_clear


def find_grid_path(grid, start, goal, cell_cost=None):
    if any(len(p) != 2 or any(not math.isfinite(v) or int(v) != v for v in p)
           for p in (start, goal)):
        raise ValueError("A* endpoints must be integer grid coordinates")
    start, goal = tuple(map(int, start)), tuple(map(int, goal))
    height, width = grid.shape
    if any(not (0 <= p[0] < width and 0 <= p[1] < height) or grid[p[1], p[0]] == -1
           for p in (start, goal)):
        return None
    if start == goal:
        return [start]
    costs = np.ones(grid.shape) if cell_cost is None else cell_cost
    if np.any(costs < 1) or not np.all(np.isfinite(costs)):
        raise ValueError("Raster A* costs must be finite and at least 1")
    distances = {start: 0.0}
    previous = {}
    queue = [(math.dist(start, goal), 0.0, start)]
    steps = [(a, b, math.hypot(a, b)) for a, b in
             [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]]
    while queue:
        _, distance, current = heapq.heappop(queue)
        if distance > distances.get(current, math.inf):
            continue
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(previous[path[-1]])
            return path[::-1]
        x, y = current
        for dx, dy, length in steps:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height) or grid[ny, nx] == -1:
                continue
            if dx and dy and (grid[y, nx] == -1 or grid[ny, x] == -1):
                continue
            new_distance = distance + length * (costs[y, x] + costs[ny, nx]) * 0.5
            point = (nx, ny)
            if new_distance + 1e-12 < distances.get(point, math.inf):
                distances[point] = new_distance
                previous[point] = current
                heapq.heappush(queue, (new_distance + math.dist(point, goal), new_distance, point))
    return None


def compress_route(path, grid, max_step_grid, randomized=False):
    """Greedy visible-span compression; optional shorter spans vary tower placement."""
    if not path:
        return None
    result = [tuple(path[0])]
    i = 0
    while i < len(path) - 1:
        limit = max_step_grid * (random.uniform(0.82, 1.0) if randomized else 1.0)
        candidates = []
        # Raster route arcs can wind; inspect a bounded forward window.
        for j in range(i + 1, min(len(path), i + int(max_step_grid * 3) + 4)):
            if math.dist(path[i], path[j]) <= limit + 1e-12:
                candidates.append(j)
        selected = None
        for j in reversed(candidates):
            if segment_is_clear(grid, path[i], path[j]):
                selected = j
                break
        if selected is None:
            if math.dist(path[i], path[i + 1]) > max_step_grid + 1e-12:
                return None
            selected = i + 1
            if not segment_is_clear(grid, path[i], path[selected]):
                return None
        result.append(tuple(path[selected]))
        i = selected
    return result


def remove_loops(path):
    result, positions = [], {}
    for point in map(tuple, path):
        if point in positions:
            result = result[:positions[point] + 1]
            positions = {p: i for i, p in enumerate(result)}
        else:
            positions[point] = len(result)
            result.append(point)
    return result


def repair_route(path, grid_map, max_span_km=2.0):
    """Repair invalid bridges geometrically, returning None if disconnected.

    Forbidden tower coordinates are not silently moved. Repairs use only raster
    connectivity; all complete candidates still require PathEvaluator.evaluate.
    """
    grid = grid_map.grid
    height, width = grid.shape
    points = remove_loops(path)
    if len(points) < 2:
        return None
    for x, y in points:
        if not (0 <= x < width and 0 <= y < height) or grid[y, x] == -1:
            return None
    max_step = max_span_km / (grid_map.config.resolution_m / 1000.0)
    repaired = [points[0]]
    for a, b in zip(points, points[1:]):
        if math.dist(a, b) <= max_step + 1e-12 and segment_is_clear(grid, a, b):
            bridge = [a, b]
        else:
            raster = find_grid_path(grid, a, b)
            bridge = compress_route(raster, grid, max_step)
            if bridge is None:
                return None
        repaired.extend(bridge[1:])
    return remove_loops(repaired)


def generate_feasible_path(grid_map, max_span_km=2.0):
    """Sample one of eight randomized cost-surface corridors, then vary towers.

    Corridor pool is created lazily per map/run. Clear _initial_route_pool between
    runs; the experiment runner supplies a fresh map for every algorithm/seed.
    The unweighted corridor is always one of the pool entries. Random fields are
    heuristic initialization only, not extra objective evaluations.
    """
    grid = grid_map.grid
    pool = getattr(grid_map, '_initial_route_pool', None)
    if pool is None:
        pool = {}
        grid_map._initial_route_pool = pool
    slot = 0 if not pool else random.randrange(8)
    if slot not in pool:
        if slot == 0:
            costs = np.ones(grid.shape)
        else:
            h, w = grid.shape
            blocks = np.random.uniform(0, 4, (math.ceil(h / 16), math.ceil(w / 16)))
            noise = np.repeat(np.repeat(blocks, 16, axis=0), 16, axis=1)[:h, :w]
            costs = 1.0 + noise + np.maximum(grid, 0) * random.uniform(0, 3)
        raster = find_grid_path(grid, grid_map.substation_a, grid_map.substation_b, costs)
        if raster is None:
            raise ValueError("No feasible route connects the substations")
        pool[slot] = raster
    max_step = max_span_km / (grid_map.config.resolution_m / 1000.0)
    route = compress_route(pool[slot], grid, max_step, randomized=True)
    if route is None:
        raise ValueError("Maximum span is too short for this grid's connectivity")
    return route
