"""Subroute replanning with soft penalties around the parent's old corridor.

This is an application-specific mutation adapted from established replanning and
corridor-penalizing ideas (e.g. Wiedemann & Adjiashvili, Section 5,
https://arxiv.org/abs/2101.03388). Soft corridor penalties and A* are not claimed
as original principles. The experiment must test whether inserting this operator
into a spatial QD archive adds value over the existing multi-route baseline.

One call makes one raster A* attempt. A large arc-length subroute is selected,
with a fixed 20% probability of replanning the entire route. Cell traversal cost
is 1 + terrain + 8 * indicator(old-subroute disk-buffer), with radius 3 cells.
All costs remain nonnegative. A* avoids forbidden cells and diagonal corner
cutting; its Euclidean heuristic is admissible for this heuristic raster cost.
The raster result is compressed to feasible visible tower spans, and the shared
geometry validator checks the combined route. This heuristic cost is not the
paper's exact tower/turning objective; only the caller evaluates that objective.

"corridor_changed" is a diagnostic: at least 20% of the new subroute's centreline
cells leave the old subroute's buffer. It is not a homotopy-class guarantee.
No objective calls are hidden in the operator. Initialization, raster search,
geometry checks and cancellation overhead belong to the caller's wall-clock
budget. ``deadline`` is an absolute ``time.perf_counter()`` timestamp.
"""

from __future__ import annotations

import heapq
import math
import random
import time

import numpy as np
from scipy.ndimage import binary_dilation

from path_evaluator import CostWeights, PathEvaluator
from qd_individual import Individual
from route_search import compress_route, remove_loops
from routing_geometry import segment_cells


class CorridorMutation:
    """Large feasible subroute mutation with bounded, cancellable work."""

    radius_cells = 3
    penalty_strength = 8.0
    whole_route_probability = 0.20
    departure_fraction_threshold = 0.20

    def __init__(self, grid_map, max_span_km=2.0, deadline=None):
        if not math.isfinite(max_span_km) or max_span_km <= 0:
            raise ValueError("max_span_km must be positive and finite")
        self.grid_map = grid_map
        self.grid = grid_map.grid
        self.height, self.width = self.grid.shape
        self.max_span_km = float(max_span_km)
        self.max_step_grid = self.max_span_km / (grid_map.config.resolution_m / 1000.0)
        self.deadline = deadline
        self.validator = PathEvaluator(grid_map, CostWeights(max_span_km=max_span_km))
        radius = self.radius_cells
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        self._disk = xx * xx + yy * yy <= radius * radius
        self._terrain = np.maximum(self.grid, 0.0).astype(float)
        self.stats = dict(attempted=0, successful=0, changed=0, corridor_changed=0,
                          raster_expansions=0, deadline_count=0, no_path_count=0,
                          span_compression_failed_count=0, invalid_bridge_count=0,
                          invalid_parent_count=0, unchanged_count=0)
        self.parameters = dict(radius_cells=self.radius_cells,
                               penalty_strength=self.penalty_strength,
                               whole_route_probability=self.whole_route_probability,
                               subroute_arc_fraction_range=[.30, .80],
                               departure_fraction_threshold=self.departure_fraction_threshold,
                               raster_base_cost=1.0, raster_terrain_weight=1.0,
                               diagonal_length=math.sqrt(2.0), deadline_check_expansions=64,
                               tie_breaking='seeded_random_neighbor_order',
                               max_span_km=self.max_span_km)

    def _expired(self):
        return self.deadline is not None and time.perf_counter() >= self.deadline

    @staticmethod
    def _fresh_copy(parent, reason):
        result = parent.copy()
        result.fitness = None
        result.behavior = None
        result.metadata = dict(result.metadata or {})
        result.metadata.update(mutation_operator='corridor_replan', replan_reason=reason,
                               replan_changed=False, replan_corridor_changed=False,
                               replan_departure_fraction=0.0)
        return result

    def _select_interval(self, path):
        if len(path) < 4 or random.random() < self.whole_route_probability:
            return 0, len(path) - 1
        lengths = [math.dist(a, b) for a, b in zip(path, path[1:])]
        cumulative = np.r_[0.0, np.cumsum(lengths)]
        total = float(cumulative[-1])
        fraction = random.uniform(0.30, 0.80)
        start_arc = random.uniform(0.0, (1.0 - fraction) * total)
        end_arc = start_arc + fraction * total
        first = max(0, int(np.searchsorted(cumulative, start_arc, side='right')) - 1)
        last = min(len(path) - 1, int(np.searchsorted(cumulative, end_arc, side='left')))
        if last <= first:
            last = min(len(path) - 1, first + 1)
        return first, last

    def _corridor_buffer(self, path):
        mask = np.zeros(self.grid.shape, dtype=bool)
        for first, last in zip(path, path[1:]):
            for x, y in segment_cells(first, last):
                mask[y, x] = True
        return binary_dilation(mask, structure=self._disk)

    def _raster_astar(self, start, goal, costs):
        """A* with periodic deadline checks; returns (path, termination reason)."""
        start, goal = tuple(start), tuple(goal)
        distance = {start: 0.0}
        previous = {}
        queue = [(math.dist(start, goal), 0, 0.0, start)]
        sequence = 0
        expanded = 0
        moves = [(dx, dy, math.hypot(dx, dy))
                 for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                                (1, 1), (1, -1), (-1, 1), (-1, -1)]]
        # Randomize equal-priority traversal without perturbing the search cost.
        # This removes the previous lexicographic coordinate preference.
        random.shuffle(moves)
        while queue:
            if expanded % 64 == 0 and self._expired():
                self.stats['raster_expansions'] += expanded
                return None, 'deadline'
            _, _, cost, point = heapq.heappop(queue)
            if cost > distance.get(point, math.inf):
                continue
            expanded += 1
            if point == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                self.stats['raster_expansions'] += expanded
                return path[::-1], 'found'
            x, y = point
            for dx, dy, length in moves:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if self.grid[ny, nx] == -1:
                    continue
                if dx and dy and (self.grid[y, nx] == -1 or self.grid[ny, x] == -1):
                    continue
                candidate_cost = cost + length * (costs[y, x] + costs[ny, nx]) * .5
                candidate = (nx, ny)
                if candidate_cost + 1e-12 < distance.get(candidate, math.inf):
                    distance[candidate] = candidate_cost
                    previous[candidate] = point
                    sequence += 1
                    heapq.heappush(queue, (candidate_cost + math.dist(candidate, goal),
                                           sequence, candidate_cost, candidate))
        self.stats['raster_expansions'] += expanded
        return None, 'no_path'

    @staticmethod
    def _departure_fraction(bridge, corridor_buffer, retained_path=None):
        cells = {cell for a, b in zip(bridge, bridge[1:]) for cell in segment_cells(a, b)}
        if retained_path is not None:
            # Loop removal can discard part of a bridge. Diagnose only cells
            # that actually survive in the returned route.
            retained = {cell for a, b in zip(retained_path, retained_path[1:])
                        for cell in segment_cells(a, b)}
            cells.intersection_update(retained)
        if not cells:
            return 0.0
        return sum(not corridor_buffer[y, x] for x, y in cells) / len(cells)

    def mutate(self, parent: Individual) -> Individual:
        self.stats['attempted'] += 1
        fallback = self._fresh_copy(parent, 'unchanged')
        if self._expired():
            self.stats['deadline_count'] += 1
            fallback.metadata['replan_reason'] = 'deadline'
            return fallback
        if not self.validator.is_valid_path(parent.path)[0]:
            self.stats['invalid_parent_count'] += 1
            raise ValueError("CorridorMutation requires a feasible parent route")
        path = [tuple(map(int, point)) for point in parent.path]
        first, last = self._select_interval(path)
        old_subroute = path[first:last + 1]
        buffer = self._corridor_buffer(old_subroute)
        costs = 1.0 + self._terrain + self.penalty_strength * buffer
        if self._expired():
            self.stats['deadline_count'] += 1
            fallback.metadata['replan_reason'] = 'deadline'
            return fallback
        raster, reason = self._raster_astar(path[first], path[last], costs)
        if raster is None:
            self.stats['deadline_count' if reason == 'deadline' else 'no_path_count'] += 1
            fallback.metadata['replan_reason'] = reason
            return fallback
        if self._expired():
            self.stats['deadline_count'] += 1
            fallback.metadata['replan_reason'] = 'deadline'
            return fallback
        bridge = compress_route(raster, self.grid, self.max_step_grid)
        if bridge is None:
            self.stats['span_compression_failed_count'] += 1
            fallback.metadata['replan_reason'] = 'span_compression_failed'
            return fallback
        candidate_path = remove_loops(path[:first] + bridge + path[last + 1:])
        if len(candidate_path) < 2 or not self.validator.is_valid_path(candidate_path)[0]:
            self.stats['invalid_bridge_count'] += 1
            fallback.metadata['replan_reason'] = 'invalid_bridge'
            return fallback
        if self._expired():
            self.stats['deadline_count'] += 1
            fallback.metadata['replan_reason'] = 'deadline'
            return fallback
        self.stats['successful'] += 1
        changed = candidate_path != path
        departure = self._departure_fraction(bridge, buffer, candidate_path)
        corridor_changed = changed and departure >= self.departure_fraction_threshold
        self.stats['changed'] += int(changed)
        self.stats['unchanged_count'] += int(not changed)
        self.stats['corridor_changed'] += int(corridor_changed)
        result = self._fresh_copy(parent, 'found')
        result.path = candidate_path
        result.metadata.update(replan_changed=changed,
                               replan_corridor_changed=bool(corridor_changed),
                               replan_departure_fraction=float(departure),
                               replan_interval=[first, last])
        return result
