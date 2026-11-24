"""Early geographic route alternatives, separate from engineering tower spotting.

Control points describe a simple planar polyline.  Cost is normal-area-equivalent
kilometres: length + weighted line-resistance exposure + weighted total turning.
Neither the number of samples nor a fictitious maximum tower span enters cost.
The default weights are declared simulation preferences, not investment prices.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np


MODEL_VERSION = 'early_geographic_corridor_v1'
COST_UNIT = 'equivalent_normal_km'
_EPS = np.finfo(float).eps


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real number')
    result = float(value)
    if not math.isfinite(result) or result < 0 or positive and result <= 0:
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return result


def _point(point):
    try:
        if len(point) != 2:
            raise ValueError('A point must have two finite real coordinates')
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in point):
            raise ValueError('A point must have two finite real coordinates')
        result = tuple(float(x) for x in point)
    except (TypeError, OverflowError) as error:
        raise ValueError('A point must have two finite real coordinates') from error
    if not all(math.isfinite(x) for x in result):
        raise ValueError('A point must have two finite real coordinates')
    return result


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _difference(a, b):
    return a[0] - b[0], a[1] - b[1]


def _collinear(a, b):
    tolerance = 64 * _EPS * math.hypot(*a) * math.hypot(*b)
    return abs(_cross(a, b)) <= tolerance


def canonical_path(path):
    """Remove forward collinear samples to roundoff, never backtracking.

    This is geometric normalization only; it does not certify map feasibility.
    Consecutive duplicate points are rejected rather than silently repaired.
    """
    try:
        points = tuple(_point(point) for point in path)
    except TypeError as error:
        raise ValueError('A path must be a sequence of points') from error
    if len(points) < 2:
        raise ValueError('A path needs at least two points')
    if any(a == b for a, b in zip(points, points[1:])):
        raise ValueError('Zero-length path segments are not allowed')
    reduced = []
    for point in points:
        while len(reduced) >= 2:
            left = _difference(reduced[-1], reduced[-2])
            right = _difference(point, reduced[-1])
            if not _collinear(left, right) or left[0]*right[0] + left[1]*right[1] <= 0:
                break
            reduced.pop()
        reduced.append(point)
    return tuple(reduced)


def _orientation(a, b, c):
    left, right = _difference(b, a), _difference(c, a)
    if _collinear(left, right):
        return 0
    return 1 if _cross(left, right) > 0 else -1


def _on_segment(a, b, p):
    tolerance = 64 * _EPS * max(1., *(abs(x) for q in (a, b, p) for x in q))
    return (_orientation(a, b, p) == 0
            and min(a[0], b[0])-tolerance <= p[0] <= max(a[0], b[0])+tolerance
            and min(a[1], b[1])-tolerance <= p[1] <= max(a[1], b[1])+tolerance)


def _intersect(a, b, c, d):
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    return (o1 * o2 < 0 and o3 * o4 < 0
            or o1 == 0 and _on_segment(a, b, c)
            or o2 == 0 and _on_segment(a, b, d)
            or o3 == 0 and _on_segment(c, d, a)
            or o4 == 0 and _on_segment(c, d, b))


def _simple_reason(points):
    for a, b, c in zip(points, points[1:], points[2:]):
        left, right = _difference(b, a), _difference(c, b)
        if _collinear(left, right) and left[0]*right[0] + left[1]*right[1] <= 0:
            return 'Backtracking along an adjacent segment is not a simple route'
    for i, (a, b) in enumerate(zip(points, points[1:])):
        for j in range(i + 2, len(points) - 1):
            if _intersect(a, b, points[j], points[j + 1]):
                return 'A route may not self-intersect or revisit a nonadjacent segment'
    return None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class CorridorWeights:
    resistance_weight: float = 1.
    turn_equivalent_km: float = .5

    def __post_init__(self):
        for name in ('resistance_weight', 'turn_equivalent_km'):
            object.__setattr__(self, name, _number(getattr(self, name), name))


class _FrozenConfiguration:
    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('Model configuration is immutable; construct a new model')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, '_sealed', False):
            raise AttributeError('Model configuration is immutable; construct a new model')
        object.__delattr__(self, name)


class CorridorMap(_FrozenConfiguration):
    """Separate forbidden-crossing cells and normalized line resistance.

    Grid-coordinate integers denote cell centres. ``origin_km`` is the physical
    coordinate of cell (0, 0).  The default domain is the cell-centre hull; an
    explicit physical domain within the raster footprint enables fixed-domain
    resolution comparisons. Inputs are copied and arrays are made read-only.
    """
    def __init__(self, forbidden, resistance, start, goal, *, resolution_m=200.,
                 origin_km=(0., 0.), name='corridor_map', domain_km=None):
        raw_forbidden, raw_resistance = np.asarray(forbidden), np.asarray(resistance)
        if (raw_forbidden.ndim != 2 or not all(raw_forbidden.shape)
                or raw_forbidden.dtype.kind not in 'biuf'
                or not np.isfinite(raw_forbidden).all()
                or not np.isin(raw_forbidden, [0, 1]).all()):
            raise ValueError('forbidden must be a nonempty two-dimensional Boolean/0-1 array')
        if (raw_resistance.shape != raw_forbidden.shape
                or raw_resistance.dtype.kind not in 'biuf'
                or not np.isfinite(raw_resistance).all()
                or np.any(raw_resistance < 0) or np.any(raw_resistance > 1)):
            raise ValueError('resistance must have the same shape and finite values in [0, 1]')
        # Immutable bytes backing prevents re-enabling NumPy's WRITEABLE flag.
        self.forbidden = np.frombuffer(np.asarray(raw_forbidden, dtype=bool).tobytes(),
                                       dtype=bool).reshape(raw_forbidden.shape)
        self.resistance = np.frombuffer(np.asarray(raw_resistance, dtype=np.float64).tobytes(),
                                        dtype=np.float64).reshape(raw_resistance.shape)
        self.resolution_km = _number(resolution_m, 'resolution_m', positive=True) / 1000
        if not math.isfinite(self.resolution_km) or self.resolution_km <= 0:
            raise ValueError('Resolution must remain positive and finite in kilometres')
        self.resolution_m = self.resolution_km * 1000
        self.origin_km = _point(origin_km)
        self.start, self.goal = _point(start), _point(goal)
        self.name = str(name)
        self.height, self.width = self.forbidden.shape
        self.shape = self.forbidden.shape
        h, (ox, oy) = self.resolution_km, self.origin_km
        if any(origin + h == origin for origin in (ox, oy)):
            raise ValueError('Physical origin/resolution cannot numerically distinguish grid cells')
        footprint = (ox-h/2, oy-h/2, ox+(self.width-.5)*h, oy+(self.height-.5)*h)
        if domain_km is None:
            self.domain_km = (ox, oy, ox+(self.width-1)*h, oy+(self.height-1)*h)
            self.domain_grid = (0., 0., float(self.width-1), float(self.height-1))
        else:
            try:
                low, high = _point(domain_km[:2]), _point(domain_km[2:])
            except (TypeError, IndexError) as error:
                raise ValueError('domain_km must be xmin,ymin,xmax,ymax') from error
            self.domain_km = (*low, *high)
            if (low[0] > high[0] or low[1] > high[1]
                    or low[0] < footprint[0] or low[1] < footprint[1]
                    or high[0] > footprint[2] or high[1] > footprint[3]):
                raise ValueError('domain_km must be ordered and inside the raster footprint')
            self.domain_grid = ((self.domain_km[0]-ox)/h, (self.domain_km[1]-oy)/h,
                                (self.domain_km[2]-ox)/h, (self.domain_km[3]-oy)/h)
        if not all(math.isfinite(v) for v in self.domain_km):
            raise ValueError('The physical domain must remain finite')
        if self.start == self.goal:
            raise ValueError('Distinct endpoints are required')
        if not self.contains(self.start) or not self.contains(self.goal):
            raise ValueError('Both endpoints must lie inside the declared domain')
        if self.to_physical(self.start) == self.to_physical(self.goal):
            raise ValueError('Physical endpoints cannot be distinguished at this coordinate precision')
        self.map_sha256 = _digest(dict(
            shape=self.shape, resolution_km=h, origin_km=self.origin_km,
            domain_km=self.domain_km, start=self.start, goal=self.goal,
            forbidden_sha256=hashlib.sha256(self.forbidden.tobytes()).hexdigest(),
            resistance_sha256=hashlib.sha256(self.resistance.astype('<f8').tobytes()).hexdigest()))
        self._sealed = True

    def contains(self, point):
        x, y = point
        # Domain roundoff must never admit a point outside the raster itself:
        # there is no density beyond this closed footprint to integrate.
        if not (-.5 <= x <= self.width-.5 and -.5 <= y <= self.height-.5):
            return False
        xmin, ymin, xmax, ymax = self.domain_grid
        tolerance = 64 * _EPS * max(1., abs(xmin), abs(ymin), abs(xmax), abs(ymax))
        return xmin-tolerance <= x <= xmax+tolerance and ymin-tolerance <= y <= ymax+tolerance

    def to_physical(self, point):
        x, y = _point(point)
        return self.origin_km[0]+x*self.resolution_km, self.origin_km[1]+y*self.resolution_km

    @classmethod
    def from_legacy(cls, grid_map, *, soft_semantics):
        """Explicitly reinterpret a legacy soft layer for this new model only."""
        if soft_semantics != 'line_resistance':
            raise ValueError('Legacy soft values must explicitly be interpreted as line_resistance')
        grid = np.asarray(grid_map.grid)
        if (grid.ndim != 2 or grid.dtype.kind not in 'biuf' or not np.isfinite(grid).all()
                or np.any((grid < 0) & (grid != -1)) or np.any(grid > 1)):
            raise ValueError('Legacy grid must contain -1 or normalized values in [0, 1]')
        return cls(grid == -1, np.maximum(grid, 0), grid_map.substation_a,
                   grid_map.substation_b, resolution_m=grid_map.config.resolution_m,
                   name='legacy_explicit_line_resistance')


def _axis_cells(value, limit):
    shifted = value + .5
    nearest = round(shifted)
    tolerance = 32 * _EPS * max(1., abs(shifted))
    if abs(shifted-nearest) <= tolerance:
        candidates = (nearest-1, nearest)
    else:
        candidates = (math.floor(shifted),)
    return tuple(i for i in candidates if 0 <= i < limit)


def _point_cells(point, shape):
    height, width = shape
    return tuple((x, y) for y in _axis_cells(point[1], height)
                 for x in _axis_cells(point[0], width))


def _segment_parts(start, end, shape):
    cuts = [0., 1.]
    delta = _difference(end, start)
    for origin, change in zip(start, delta):
        if change:
            low, high = sorted((origin, origin+change))
            first, last = math.floor(low-.5)+1, math.ceil(high-.5)-1
            cuts.extend((i+.5-origin)/change for i in range(first, last+1))
    cuts = sorted(set(max(0., min(1., t)) for t in cuts))
    touched = set()
    for t in cuts:
        touched.update(_point_cells((start[0]+t*delta[0], start[1]+t*delta[1]), shape))
    intervals = []
    for begin, finish in zip(cuts, cuts[1:]):
        if finish <= begin:
            continue
        mid = (begin+finish)/2
        cells = _point_cells((start[0]+mid*delta[0], start[1]+mid*delta[1]), shape)
        touched.update(cells)
        intervals.append((finish-begin, cells))
    return touched, intervals


class CorridorEvaluator(_FrozenConfiguration):
    """Common deterministic objective/constraint interface for all algorithms."""
    cost_unit = COST_UNIT
    version = MODEL_VERSION

    def __init__(self, corridor_map, weights=None):
        if not isinstance(corridor_map, CorridorMap):
            raise TypeError('corridor_map must be a CorridorMap')
        self.corridor_map = corridor_map
        self.weights = weights if weights is not None else CorridorWeights()
        if not isinstance(self.weights, CorridorWeights):
            raise TypeError('weights must be CorridorWeights')
        self.implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.model_sha256 = _digest(dict(version=self.version, unit=self.cost_unit,
            map_sha256=corridor_map.map_sha256, weights=asdict(self.weights),
            implementation_sha256=self.implementation_sha256))
        self._sealed = True

    def _prepare(self, path):
        try:
            points = canonical_path(path)
        except (TypeError, ValueError, OverflowError) as error:
            return None, str(error)
        grid = self.corridor_map
        # Endpoints identify one fixed task, not a proximity tolerance region.
        if points[0] != grid.start or points[-1] != grid.goal:
            return None, 'Path endpoints must equal the configured start and goal'
        if not all(grid.contains(point) for point in points):
            return None, 'Every control point must be inside the declared domain'
        physical = [grid.to_physical(point) for point in points]
        if (not all(math.isfinite(v) for p in physical for v in p)
                or any(a == b for a, b in zip(physical, physical[1:]))):
            return None, 'Physical segments are not numerically distinguishable'
        reason = _simple_reason(points)
        if reason:
            return None, reason
        return points, None

    def is_valid_path(self, path):
        points, reason = self._prepare(path)
        if reason:
            return False, reason
        grid = self.corridor_map
        for start, end in zip(points, points[1:]):
            touched, _ = _segment_parts(start, end, grid.shape)
            if any(grid.forbidden[y, x] for x, y in touched):
                return False, 'A segment touches a forbidden crossing cell'
        return True, ''

    def evaluate(self, path):
        points, reason = self._prepare(path)
        base = dict(valid=False, cost_unit=self.cost_unit, model_version=self.version,
                    model_sha256=self.model_sha256)
        if reason:
            return math.inf, dict(base, failure_reason=reason)
        grid = self.corridor_map
        lengths, exposures = [], []
        for start, end in zip(points, points[1:]):
            touched, intervals = _segment_parts(start, end, grid.shape)
            if any(grid.forbidden[y, x] for x, y in touched):
                return math.inf, dict(base, failure_reason='A segment touches a forbidden crossing cell')
            length = math.dist(start, end) * grid.resolution_km
            lengths.append(length)
            # At a grid boundary, density is the mean of incident cells. An
            # isolated corner has zero exposure length but still blocks hard.
            exposures.append(length * math.fsum(
                fraction * math.fsum(float(grid.resistance[y, x]) for x, y in cells)/len(cells)
                for fraction, cells in intervals))
        turns = []
        for before, current, after in zip(points, points[1:], points[2:]):
            left, right = _difference(current, before), _difference(after, current)
            turns.append(math.atan2(abs(_cross(left, right)),
                                   left[0]*right[0]+left[1]*right[1]))
        length, exposure, turning = math.fsum(lengths), math.fsum(exposures), math.fsum(turns)
        components = (length, self.weights.resistance_weight*exposure,
                      self.weights.turn_equivalent_km*turning/(math.pi/2))
        total = math.fsum(components)
        if not math.isfinite(total) or total <= 0:
            return math.inf, dict(base, failure_reason='Route cost must remain finite and positive')
        return total, dict(base, valid=True, failure_reason=None, length_km=length,
            resistance_exposure_km=exposure, turning_radians=turning,
            turning_90_equivalents=turning/(math.pi/2), length_component=components[0],
            resistance_component=components[1], turn_component=components[2],
            canonical_control_point_count=len(points))

    def intersects_disk(self, path, center_km, radius_km):
        """Closed contact with an exact continuous physical disk.

        This geometric query does not certify original-map route feasibility.
        It does not rasterize the new disk or infer connectedness after closure.
        """
        points = canonical_path(path)
        center = _point(center_km)
        radius = _number(radius_km, 'radius_km')
        physical = [self.corridor_map.to_physical(point) for point in points]
        for start, end in zip(physical, physical[1:]):
            vector, relative = _difference(end, start), _difference(center, start)
            length = math.hypot(*vector)
            if not math.isfinite(length) or length <= 0:
                raise ValueError('Physical segment is outside the numerical coordinate range')
            # Unit-direction projection avoids squaring a tiny segment length.
            unit = vector[0]/length, vector[1]/length
            projection = relative[0]*unit[0]+relative[1]*unit[1]
            if math.isnan(projection):
                raise ValueError('Disk coordinates exceed the numerical coordinate range')
            along = max(0., min(length, projection))
            distance = math.hypot(relative[0]-along*unit[0], relative[1]-along*unit[1])
            if math.isinf(distance):
                continue
            tolerance = 32*_EPS*max(radius, distance, *(abs(v) for p in (start,end,center) for v in p))
            if distance <= radius + tolerance:
                return True
        return False
