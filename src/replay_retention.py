"""Capacity-controlled policies for replaying an identical evaluated route stream.

No path generation, objective evaluation, or historical candidate pool is hidden
in this module.  Every policy deduplicates exact geometric curves and retains
the cheapest tower representation of a curve it currently holds.  The
``cost_ordered`` policy borrows nearest-fitter competition from DNS, but uses
a declared total cost/tie order, one-candidate updates, and uniform-parent
semantics elsewhere; it is not a full implementation of the published DNS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Iterable

import numpy as np


METHODS = ('fixed_anchor', 'cost_ordered', 'strict_dns', 'unique_topk', 'spatial_greedy')


def _number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('path coordinates must be finite')
    return int(value) if value.is_integer() else value


def geometric_path_key(path):
    """Remove redundant collinear samples, retaining backtracking and order."""
    result = []
    for raw in path:
        if len(raw) != 2:
            raise ValueError('path points must have two coordinates')
        point = tuple(_number(v) for v in raw)
        if result and point == result[-1]:
            continue
        while len(result) >= 2:
            a, b = result[-2:]
            u = (b[0] - a[0], b[1] - a[1])
            v = (point[0] - b[0], point[1] - b[1])
            if u[0] * v[1] - u[1] * v[0] != 0 or u[0] * v[0] + u[1] * v[1] < 0:
                break
            result.pop()
        result.append(point)
    return tuple(result)


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


@dataclass(frozen=True, eq=False)
class RouteRecord:
    route_id: str
    path: tuple
    cost: float
    feature: np.ndarray = field(repr=False)
    geometry_key: tuple = field(init=False)
    path_key: tuple = field(init=False)

    def __post_init__(self):
        route_id = str(self.route_id)
        if not route_id:
            raise ValueError('route_id must not be empty')
        points = tuple(tuple(_number(v) for v in point) for point in self.path)
        if len(points) < 2 or any(len(point) != 2 for point in points):
            raise ValueError('a path must contain at least two two-dimensional points')
        geometry = geometric_path_key(points)
        if len(geometry) < 2:
            raise ValueError('route geometry must have nonzero extent')
        cost = float(self.cost)
        if not math.isfinite(cost):
            raise ValueError('cost must be finite')
        raw = np.asarray(self.feature, dtype=np.float64)
        if raw.ndim != 1 or raw.size == 0 or not np.isfinite(raw).all():
            raise ValueError('feature must be a nonempty finite vector')
        # A bytes-backed array cannot have its write flag re-enabled by callers.
        feature = np.frombuffer(np.ascontiguousarray(raw).tobytes(), dtype=np.float64)
        object.__setattr__(self, 'route_id', route_id)
        object.__setattr__(self, 'path', points)
        object.__setattr__(self, 'path_key', points)
        object.__setattr__(self, 'geometry_key', geometry)
        object.__setattr__(self, 'cost', cost)
        object.__setattr__(self, 'feature', feature)

    def to_dict(self, include_feature=False):
        value = dict(route_id=self.route_id, path=[list(p) for p in self.path], cost=self.cost,
                     geometry_key=[list(p) for p in self.geometry_key])
        if include_feature:
            value['feature'] = self.feature.tolist()
        return value


class RetentionPolicy:
    """A bounded current population plus fixed prototypes or a distance cache.

    ``records`` exposes immutable records in stable population order.  The
    returned list itself is a copy.  Each ``offer`` event is JSON-safe, including
    competition scores whose infinities are represented by null plus a flag.
    """

    def __init__(self, initial_records: Iterable[RouteRecord], method='fixed_anchor',
                 capacity=None, tie_salt='20261031', greedy_tolerance=.1,
                 separation=1e-10):
        records = list(initial_records)
        if method not in METHODS:
            raise ValueError(f'method must be one of {METHODS}')
        capacity = len(records) if capacity is None else capacity
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise ValueError('capacity must be a positive integer')
        if len(records) != capacity:
            raise ValueError('initial_records must contain exactly capacity routes')
        if not all(isinstance(r, RouteRecord) for r in records):
            raise TypeError('initial_records must contain RouteRecord objects')
        if len({r.route_id for r in records}) != capacity:
            raise ValueError('initial route IDs must be distinct')
        if len({r.geometry_key for r in records}) != capacity:
            raise ValueError('initial route geometries must be distinct')
        dimension = records[0].feature.size
        if any(r.feature.size != dimension for r in records):
            raise ValueError('all records must have the same feature dimension')
        if not math.isfinite(separation) or separation <= 0:
            raise ValueError('separation must be finite and positive')
        if not math.isfinite(greedy_tolerance) or greedy_tolerance < 0:
            raise ValueError('greedy_tolerance must be finite and nonnegative')
        self.method, self.capacity, self.dimension = method, int(capacity), dimension
        self.tie_salt, self.greedy_tolerance = str(tie_salt), float(greedy_tolerance)
        self.separation = float(separation)
        self._records = records.copy()
        self._distance_cache = {}
        self._tie_cache = {}
        self.anchors = None
        if method == 'fixed_anchor':
            for i, route in enumerate(records):
                if any(np.linalg.norm(route.feature - prev.feature) <= separation for prev in records[:i]):
                    raise ValueError('initial anchor features collide')
            self.anchors = np.stack([r.feature for r in records])
            self.anchors.setflags(write=False)
        self.stats = dict(offers=0, accepted=0, rejected=0, geometry_duplicate_offers=0,
                          geometry_replacements=0, exact_duplicate_offers=0,
                          distance_computations=0, distance_cache_hits=0,
                          selection_changes=0, max_distance_cache_entries=0,
                          greedy_quality_fill_slots=0, equal_cost_offers=0)

    @property
    def records(self):
        return self._records.copy()

    @property
    def population(self):
        return self.records

    def _tie(self, route):
        if route.route_id not in self._tie_cache:
            self._tie_cache[route.route_id] = hashlib.sha256(
                self.tie_salt.encode('utf-8') + b'\0' + _encoded(route.geometry_key)).hexdigest()
        return self._tie_cache[route.route_id]

    def _cost_order(self, route):
        return route.cost, self._tie(route), route.path_key, route.route_id

    def _raw_cost_order(self, route):
        return route.cost, route.path_key, route.route_id

    def _distance(self, first, second):
        if first.route_id == second.route_id:
            return 0.0
        pair = tuple(sorted((first.route_id, second.route_id)))
        if pair not in self._distance_cache:
            self._distance_cache[pair] = float(np.linalg.norm(first.feature - second.feature))
            self.stats['distance_computations'] += 1
            self.stats['max_distance_cache_entries'] = max(
                self.stats['max_distance_cache_entries'], len(self._distance_cache))
        else:
            self.stats['distance_cache_hits'] += 1
        return self._distance_cache[pair]

    def _prune_caches(self):
        living = {r.route_id for r in self._records}
        self._distance_cache = {pair: value for pair, value in self._distance_cache.items()
                                if pair[0] in living and pair[1] in living}
        self._tie_cache = {key: value for key, value in self._tie_cache.items() if key in living}

    def _competition(self, routes):
        ordered = sorted(routes, key=self._cost_order)
        scores, references = {}, {}
        for index, route in enumerate(ordered):
            if self.method == 'strict_dns':
                predecessors = [r for r in ordered[:index] if r.cost < route.cost]
            else:
                predecessors = ordered[:index]
            if not predecessors:
                scores[route.route_id] = math.inf
                references[route.route_id] = None
            else:
                reference = min(predecessors, key=lambda r: (self._distance(route, r),
                                                             self._raw_cost_order(r)))
                scores[route.route_id] = self._distance(route, reference)
                references[route.route_id] = reference.route_id
        chosen = sorted(routes, key=lambda r: (-scores[r.route_id], r.cost, r.path_key,
                                               r.route_id))[:self.capacity]
        audit = {r.route_id: dict(score=None if math.isinf(scores[r.route_id]) else scores[r.route_id],
                                  score_infinite=math.isinf(scores[r.route_id]),
                                  reference_id=references[r.route_id]) for r in routes}
        return chosen, dict(competition=audit,
                            strict_minimum_tie_count=sum(r.cost == ordered[0].cost for r in ordered))

    def _greedy(self, routes):
        ordered = sorted(routes, key=self._cost_order)
        best = ordered[0]
        cutoff = best.cost * (1.0 + self.greedy_tolerance)
        # Positive routing costs are expected; this expression retains the best
        # even for finite negative toy costs used in independent diagnostics.
        eligible = [r for r in ordered if r.cost <= cutoff or r.route_id == best.route_id]
        selected = [best]
        remaining = [r for r in eligible if r.route_id != best.route_id]
        nearest = {r.route_id: self._distance(r, best) for r in remaining}
        while remaining and len(selected) < self.capacity:
            chosen = min(remaining, key=lambda r: (-nearest[r.route_id], self._raw_cost_order(r)))
            selected.append(chosen)
            remaining = [r for r in remaining if r.route_id != chosen.route_id]
            for route in remaining:
                nearest[route.route_id] = min(nearest[route.route_id], self._distance(route, chosen))
        fill_count = 0
        selected_ids = {r.route_id for r in selected}
        for route in ordered:
            if len(selected) >= self.capacity:
                break
            if route.route_id not in selected_ids:
                selected.append(route)
                selected_ids.add(route.route_id)
                fill_count += 1
        self.stats['greedy_quality_fill_slots'] += fill_count
        return selected, dict(online_quality_cutoff=cutoff, quality_eligible_count=len(eligible),
                              quality_fill_slots=fill_count)

    def offer(self, candidate: RouteRecord):
        if not isinstance(candidate, RouteRecord):
            raise TypeError('candidate must be a RouteRecord')
        if candidate.feature.size != self.dimension:
            raise ValueError('candidate feature dimension differs')
        current_id = next((r for r in self._records if r.route_id == candidate.route_id), None)
        if current_id is not None and (current_id.path_key != candidate.path_key
                                      or current_id.cost != candidate.cost
                                      or not np.array_equal(current_id.feature, candidate.feature)):
            raise ValueError('a currently retained route_id cannot identify different data')
        self.stats['offers'] += 1
        self.stats['equal_cost_offers'] += int(any(r.cost == candidate.cost for r in self._records))
        before = self._records.copy()
        event = dict(candidate_id=candidate.route_id, candidate_cost=candidate.cost,
                     method=self.method, accepted=False, reason=None,
                     deleted_ids=[], reference_id=None, score=None, score_infinite=False)
        duplicate_index = next((i for i, r in enumerate(before)
                                if r.geometry_key == candidate.geometry_key), None)
        if duplicate_index is not None:
            previous = before[duplicate_index]
            self.stats['geometry_duplicate_offers'] += 1
            self.stats['exact_duplicate_offers'] += int(previous.path_key == candidate.path_key)
            event['geometry_duplicate_of'] = previous.route_id
            if self._raw_cost_order(candidate) < self._raw_cost_order(previous):
                if np.linalg.norm(candidate.feature - previous.feature) > self.separation:
                    raise ValueError('same geometry has inconsistent features')
                self._records[duplicate_index] = candidate
                self.stats['geometry_replacements'] += 1
                event.update(accepted=True, reason='geometry_cheaper_or_stable_tie',
                             deleted_ids=[previous.route_id])
            else:
                event['reason'] = 'geometry_not_better'
        elif self.method == 'fixed_anchor':
            difference = self.anchors - candidate.feature
            # Contract directly instead of materializing a second K-by-D
            # squared array. This is important for the occupancy descriptor.
            squared = np.einsum('ij,ij->i', difference, difference, optimize=False)
            index = int(np.argmin(squared))
            previous = before[index]
            event.update(anchor_index=index, anchor_distance=math.sqrt(float(squared[index])),
                         reference_id=previous.route_id)
            if candidate.cost < previous.cost:
                self._records[index] = candidate
                event.update(accepted=True, reason='cell_cost_improvement', deleted_ids=[previous.route_id])
            else:
                event['reason'] = 'cell_cost_not_better'
        else:
            routes = before + [candidate]
            if self.method in ('cost_ordered', 'strict_dns'):
                self._records, extra = self._competition(routes)
                event.update(extra)
                event.update(event['competition'][candidate.route_id])
            elif self.method == 'unique_topk':
                self._records = sorted(routes, key=self._raw_cost_order)[:self.capacity]
            else:
                self._records, extra = self._greedy(routes)
                event.update(extra)
            selected = {r.route_id for r in self._records}
            event['accepted'] = candidate.route_id in selected
            event['deleted_ids'] = [r.route_id for r in before if r.route_id not in selected]
            event['reason'] = 'policy_selected' if event['accepted'] else 'policy_not_selected'
        self.stats['accepted' if event['accepted'] else 'rejected'] += 1
        self.stats['selection_changes'] += int(bool(event['deleted_ids']))
        self._prune_caches()
        event.update(retained_count=len(self._records),
                     geometry_unique_count=len({r.geometry_key for r in self._records}),
                     retained_ids=[r.route_id for r in self._records])
        return event

    def snapshot(self, include_records=True):
        value = dict(method=self.method, capacity=self.capacity, retained_count=len(self._records),
                     feature_dimension=self.dimension,
                     geometry_unique_count=len({r.geometry_key for r in self._records}),
                     best_cost=min(r.cost for r in self._records),
                     distance_cache_entries=len(self._distance_cache),
                     tie_cache_entries=len(self._tie_cache), stats=dict(self.stats))
        if include_records:
            value['records'] = [r.to_dict() for r in self._records]
        return value
