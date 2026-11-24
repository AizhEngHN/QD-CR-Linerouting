"""Arc-length-integrated, soft spatial occupancy descriptors for route curves.

Each arc-length section contributes a probability distribution over a physical
bilinear lattice at each specified scale.  Concatenated square-root masses,
divided by sqrt(2 * sections * scales), embed the mean squared Hellinger
distance as ordinary Euclidean distance.  This is a finite geometric summary,
not a complete corridor/homotopy classifier or a route feasibility validator.

The integration is exact up to floating-point arithmetic: a straight segment
is split at section and lattice boundaries, and Simpson's rule integrates its
quadratic bilinear weights exactly.  Tower counts never act as sample weights.
The input tower sequence and any separately computed engineering cost remain
untouched.  Map values do not enter the representation.
"""

from __future__ import annotations

import math

import numpy as np


class SoftOccupancyDescriptor:
    """Fixed, map-only descriptor geometry, with coordinates supplied in cells.

    ``probabilities(path)`` returns one ``(segments, ny * nx)`` array per
    scale, ordered as ``spacings_km``; nodes use row-major ``y * nx + x``.
    ``compute(path)`` flattens those arrays in the same order.  Floating tower
    coordinates are permitted for interpolation diagnostics, while endpoints
    must equal the configured substations and all points must be in the map.
    Forbidden regions, tower integrality and maximum spans are deliberately
    checked by the common route evaluator, not here.
    """

    version = 'arc_sections_bilinear_occupancy_v1'

    def __init__(self, grid_map, segments=4, spacings_km=(0.4, 0.8)):
        try:
            integer_segments = int(segments)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('segments must be a positive integer') from error
        if isinstance(segments, bool) or integer_segments != segments or segments < 1:
            raise ValueError('segments must be a positive integer')
        self.segments = integer_segments
        try:
            self.width = int(grid_map.grid_width)
            self.height = int(grid_map.grid_height)
            self.resolution_km = float(grid_map.config.resolution_m) / 1000.0
            spacings = tuple(float(x) for x in spacings_km)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise ValueError('map dimensions, resolution and spacings must be valid') from error
        if self.width < 1 or self.height < 1:
            raise ValueError('map dimensions must be positive')
        if not math.isfinite(self.resolution_km) or self.resolution_km <= 0:
            raise ValueError('resolution_m must be positive and finite')
        if not spacings or any(not math.isfinite(x) or x <= 0 for x in spacings):
            raise ValueError('spacings_km must contain positive finite values')
        if len(set(spacings)) != len(spacings):
            raise ValueError('spacings_km must not contain duplicate scales')
        self.spacings_km = spacings
        self.start = self._point(grid_map.substation_a)
        self.goal = self._point(grid_map.substation_b)
        if not self._inside(self.start) or not self._inside(self.goal):
            raise ValueError('substations must lie inside the map')
        if np.array_equal(self.start, self.goal):
            raise ValueError('distinct substations are required')
        extent_x = (self.width - 1) * self.resolution_km
        extent_y = (self.height - 1) * self.resolution_km
        self.node_shapes = tuple(
            (max(2, math.ceil(extent_y / h) + 1),
             max(2, math.ceil(extent_x / h) + 1)) for h in spacings)
        self.dimension = self.segments * sum(ny * nx for ny, nx in self.node_shapes)
        self._embedding_scale = math.sqrt(2 * self.segments * len(spacings))

    @staticmethod
    def _point(value):
        try:
            result = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('substations must be finite 2-D coordinates') from error
        if result.shape != (2,) or not np.all(np.isfinite(result)):
            raise ValueError('substations must be finite 2-D coordinates')
        return result.copy()

    def _inside(self, point):
        return (0 <= point[0] <= self.width - 1 and
                0 <= point[1] <= self.height - 1)

    def _canonical_points(self, path):
        try:
            points = np.asarray(path, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('path must contain finite 2-D coordinates') from error
        if (points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2
                or not np.all(np.isfinite(points))):
            raise ValueError('path must contain at least two finite 2-D coordinates')
        if not np.array_equal(points[0], self.start) or not np.array_equal(points[-1], self.goal):
            raise ValueError('path endpoints must match configured substations')
        if (np.any(points[:, 0] < 0) or np.any(points[:, 0] > self.width - 1)
                or np.any(points[:, 1] < 0) or np.any(points[:, 1] > self.height - 1)):
            raise ValueError('all path coordinates must lie inside the map')
        differences = np.diff(points, axis=0)
        if np.any(np.hypot(differences[:, 0], differences[:, 1]) <= 0):
            raise ValueError('zero-length path segments are not permitted')
        # Remove only forward collinear samples (to roundoff). Backtracking
        # must preserve its actual traveled length, not be silently simplified.
        reduced = []
        for point in points:
            while len(reduced) >= 2:
                left = reduced[-1] - reduced[-2]
                right = point - reduced[-1]
                cross = left[0] * right[1] - left[1] * right[0]
                tolerance = (64 * np.finfo(float).eps * math.hypot(*left)
                             * math.hypot(*right))
                if abs(cross) > tolerance or np.dot(left, right) <= 0:
                    break
                reduced.pop()
            reduced.append(point.copy())
        return np.asarray(reduced, dtype=float) * self.resolution_km

    @staticmethod
    def _lattice_cuts(origin, change, spacing):
        if change == 0:
            return np.empty(0)
        lower, upper = sorted((origin, origin + change))
        first = math.floor(lower / spacing) + 1
        last = math.ceil(upper / spacing) - 1
        if last < first:
            return np.empty(0)
        return (np.arange(first, last + 1, dtype=float) * spacing - origin) / change

    @staticmethod
    def _weights(fx, fy):
        return np.column_stack(((1 - fx) * (1 - fy), fx * (1 - fy),
                                (1 - fx) * fy, fx * fy))

    def probabilities(self, path):
        """Return fresh arrays of normalized arc-length occupancy masses.

        Row sums are checked before correcting floating-point roundoff. No
        positive mass may be lost outside the map: the fixed node lattice
        extends to/past the final cell centre and contains at least two nodes
        in each axis even for a one-cell-wide map.
        """
        points = self._canonical_points(path)
        deltas = np.diff(points, axis=0)
        lengths = np.hypot(deltas[:, 0], deltas[:, 1])
        total = float(np.sum(lengths))
        if not math.isfinite(total) or total <= 0:
            raise ValueError('path arc length must be positive and finite')
        section_length = total / self.segments
        section_edges = np.arange(1, self.segments, dtype=float) * section_length
        segment_starts = np.r_[0., np.cumsum(lengths)[:-1]]
        results = []
        for spacing, (ny, nx) in zip(self.spacings_km, self.node_shapes):
            piece_start, piece_end, piece_length, piece_section = [], [], [], []
            for origin, change, length, arc_start in zip(points[:-1], deltas, lengths, segment_starts):
                section_cuts = (section_edges - arc_start) / length
                candidates = np.r_[0., 1., section_cuts,
                                   self._lattice_cuts(origin[0], change[0], spacing),
                                   self._lattice_cuts(origin[1], change[1], spacing)]
                cuts = np.unique(candidates[(candidates >= 0) & (candidates <= 1)])
                left, right = cuts[:-1], cuts[1:]
                midpoint = (left + right) * .5
                piece_start.append(origin + left[:, None] * change)
                piece_end.append(origin + right[:, None] * change)
                piece_length.append((right - left) * length)
                sections = np.floor((arc_start + midpoint * length) / section_length).astype(int)
                piece_section.append(np.clip(sections, 0, self.segments - 1))
            starts, ends = np.concatenate(piece_start), np.concatenate(piece_end)
            widths, sections = np.concatenate(piece_length), np.concatenate(piece_section)
            mids = (starts + ends) * .5
            ix = np.clip(np.floor(mids[:, 0] / spacing).astype(int), 0, nx - 2)
            iy = np.clip(np.floor(mids[:, 1] / spacing).astype(int), 0, ny - 2)
            fractions = [starts[:, 0] / spacing - ix, starts[:, 1] / spacing - iy,
                         ends[:, 0] / spacing - ix, ends[:, 1] / spacing - iy]
            if any(np.any(f < -2e-11) or np.any(f > 1 + 2e-11) for f in fractions):
                raise ArithmeticError('a segment crossed an unsplit lattice boundary')
            fx0, fy0, fx1, fy1 = [np.clip(f, 0, 1) for f in fractions]
            # For fixed lattice patch, each basis weight is quadratic along
            # the line. Simpson's 1:4:1 combination integrates it exactly.
            weights = (self._weights(fx0, fy0)
                       + 4 * self._weights((fx0 + fx1) * .5, (fy0 + fy1) * .5)
                       + self._weights(fx1, fy1)) / 6
            nodes = np.column_stack((iy * nx + ix, iy * nx + ix + 1,
                                     (iy + 1) * nx + ix, (iy + 1) * nx + ix + 1))
            masses = np.zeros((self.segments, ny * nx), dtype=float)
            np.add.at(masses, (sections[:, None], nodes),
                      weights * (widths / section_length)[:, None])
            row_sums = masses.sum(axis=1)
            if (np.any(~np.isfinite(masses)) or np.any(masses < -1e-13)
                    or np.max(np.abs(row_sums - 1)) > 2e-10):
                raise ArithmeticError('occupancy mass failed normalization audit')
            masses = np.maximum(masses, 0)
            masses /= masses.sum(axis=1)[:, None]
            results.append(masses)
        return tuple(results)

    def compute(self, path):
        """Return the Euclidean embedding of the averaged Hellinger distance."""
        return np.concatenate([np.sqrt(p).ravel() for p in self.probabilities(path)]) / self._embedding_scale

    def metadata(self):
        return dict(version=self.version, segments=self.segments,
                    spacings_km=list(self.spacings_km), resolution_km=self.resolution_km,
                    map_shape_cells=[self.height, self.width],
                    node_shapes_yx=[list(x) for x in self.node_shapes],
                    dimension=self.dimension, dtype='float64',
                    ordering='scale, arc-length section, row-major geographical node',
                    lattice_origin_km=[0., 0.],
                    integration='section/lattice boundary splitting; exact quadratic Simpson rule',
                    endpoint_policy='configured substations; real coordinates allowed for diagnostics',
                    collinear_tolerance='64 * float_epsilon * adjacent segment norm product; forward only',
                    distance='Euclidean embedding of mean section/scale squared Hellinger',
                    interpretation='finite space-distribution summary; not complete corridor or topology identification')
