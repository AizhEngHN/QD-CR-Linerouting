"""Physical curve geometry, independent of archive features and raster cost.

Capsule/segment intersections are analytic; the mean nearest-curve distance is
a midpoint quadrature with a reported Lipschitz absolute-error bound. Distances
and endpoint clipping are in km. No route feasibility is certified here.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import quad


def canonical_points(path):
    x = np.asarray(path, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Expected at least two finite physical points')
    out = []
    for p in x:
        if out and np.array_equal(out[-1], p):
            continue
        while len(out) >= 2:
            u, v = out[-1]-out[-2], p-out[-1]
            cross = u[0]*v[1]-u[1]*v[0]
            if abs(cross) > 64*np.finfo(float).eps*np.linalg.norm(u)*np.linalg.norm(v) or u@v <= 0:
                break
            out.pop()
        out.append(p.copy())
    if len(out) < 2:
        raise ValueError('Zero-length route')
    return np.array(out)


def _circle_intervals(a, v, centers, radius):
    """Parameter intervals on each a+t*v inside each closed circle."""
    delta = a[:, None, :] - np.asarray(centers)[None, :, :]
    aa = np.sum(v*v, axis=1)[:, None]
    bb = 2*np.sum(delta*v[:, None, :], axis=2)
    cc = np.sum(delta*delta, axis=2)-radius*radius
    disc = bb*bb-4*aa*cc
    tol = 32*np.finfo(float).eps*(bb*bb+np.abs(4*aa*cc)+1)
    root = np.sqrt(np.maximum(disc, 0))
    lo = np.maximum(0, (-bb-root)/(2*aa))
    hi = np.minimum(1, (-bb+root)/(2*aa))
    valid = (disc >= -tol) & (hi >= lo)
    return np.where(valid, lo, 1.), np.where(valid, hi, 0.)


def _union_lengths(lo, hi):
    order = np.argsort(lo, axis=1, kind='stable')
    left, right = np.take_along_axis(lo, order, 1), np.take_along_axis(hi, order, 1)
    previous = np.maximum.accumulate(np.column_stack([np.zeros(len(lo)), right]), axis=1)[:, :-1]
    return np.maximum(0, right-np.maximum(left, previous)).sum(axis=1)


def clip_endpoint_disks(path, endpoints, radius=1.):
    """Return a union of oriented line segments outside both endpoint disks."""
    p = canonical_points(path)
    a, v = p[:-1], np.diff(p, axis=0)
    if radius < 0 or not math.isfinite(radius):
        raise ValueError('Invalid endpoint radius')
    if radius == 0:
        return np.stack([a, a+v], axis=1)
    lo, hi = _circle_intervals(a, v, np.asarray(endpoints, float), radius)
    result = []
    for i in range(len(a)):
        intervals = sorted((l, h) for l, h in zip(lo[i], hi[i]) if h > l)
        cursor = 0.
        for l, h in intervals:
            if l > cursor:
                result.append([a[i]+cursor*v[i], a[i]+l*v[i]])
            cursor = max(cursor, h)
        if cursor < 1:
            result.append([a[i]+cursor*v[i], a[i]+v[i]])
    if not result:
        raise ValueError('No route remains outside endpoint disks')
    return np.asarray(result)


def curve_length(segments):
    return float(np.linalg.norm(np.diff(segments, axis=1)[:, 0], axis=1).sum())


def point_curve_distances(points, segments, chunk=2048):
    a, v = segments[:, 0], segments[:, 1]-segments[:, 0]
    vv = np.sum(v*v, axis=1)
    if len(a) == 0 or np.any(vv <= 0):
        raise ValueError('Nonempty positive-length segments required')
    result = []
    for start in range(0, len(points), chunk):
        delta = np.asarray(points)[start:start+chunk, None, :]-a[None, :, :]
        t = np.clip(np.sum(delta*v[None, :, :], axis=2)/vv[None, :], 0, 1)
        result.append(np.sqrt(np.min(np.sum((delta-t[:, :, None]*v)**2, axis=2), axis=1)))
    return np.concatenate(result) if result else np.empty(0)


def _linear_band(value, slope, lower, upper):
    value, slope, lower, upper = np.broadcast_arrays(value, slope, lower, upper)
    # A rigid rotation of parallel lines creates tiny nonzero slopes and moves
    # a closed-boundary value by a few ulps. Expand only by arithmetic roundoff,
    # and treat sub-tolerance slopes as constant over the unit parameter range.
    tol=64*np.finfo(float).eps*(1+np.abs(value)+np.abs(slope)+np.abs(lower)+np.abs(upper))
    lower,upper=lower-tol,upper+tol
    moving = np.abs(slope)>tol
    t1 = np.divide(lower-value, slope, out=np.zeros_like(value), where=moving)
    t2 = np.divide(upper-value, slope, out=np.ones_like(value), where=moving)
    lo, hi = np.maximum(0, np.minimum(t1, t2)), np.minimum(1, np.maximum(t1, t2))
    constant_ok = (value >= lower) & (value <= upper)
    valid = np.where(moving, hi >= lo, constant_ok)
    return np.where(valid & moving, lo, np.where(valid, 0., 1.)), np.where(valid & moving, hi, np.where(valid, 1., 0.))


def covered_length(source, target, radius):
    """Exact (to floating arithmetic) source length in target capsule union."""
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError('Positive finite buffer radius required')
    a, v = source[:, 0], source[:, 1]-source[:, 0]
    c, w = target[:, 0], target[:, 1]-target[:, 0]
    delta = a[:, None, :]-c[None, :, :]
    ww = np.sum(w*w, axis=1)[None, :]
    dot = np.sum(delta*w[None, :, :], axis=2)
    ds = np.sum(v[:, None, :]*w[None, :, :], axis=2)
    l1, h1 = _linear_band(dot, ds, 0., ww)
    cross = delta[:, :, 0]*w[None, :, 1]-delta[:, :, 1]*w[None, :, 0]
    cs = v[:, None, 0]*w[None, :, 1]-v[:, None, 1]*w[None, :, 0]
    width = radius*np.sqrt(ww)
    l2, h2 = _linear_band(cross, cs, -width, width)
    rl, rh = np.maximum(l1, l2), np.minimum(h1, h2)
    ok = rh >= rl
    starts, ends = [np.where(ok, rl, 1.)], [np.where(ok, rh, 0.)]
    for centers in (target[:, 0], target[:, 1]):
        l, h = _circle_intervals(a, v, centers, radius)
        starts.append(l); ends.append(h)
    fractions = _union_lengths(np.concatenate(starts, axis=1), np.concatenate(ends, axis=1))
    return float(np.dot(np.linalg.norm(v, axis=1), np.clip(fractions, 0, 1)))


def geometric_difference(first, second, radius=.5):
    overlap = .5*(covered_length(first, second, radius)/curve_length(first)
                  + covered_length(second, first, radius)/curve_length(second))
    return float(np.clip(1-overlap, 0, 1))


def mean_nearest_distance(first, second, max_step=.05):
    """Symmetric arc-weighted mean distance and midpoint absolute error bound."""
    if max_step <= 0:
        raise ValueError('Positive sampling step required')
    values, bounds = [], []
    for source, target in ((first, second), (second, first)):
        points, weights = [], []
        for a, b in source:
            length = np.linalg.norm(b-a)
            n = max(1, math.ceil(length/max_step))
            t = (np.arange(n)+.5)/n
            points.extend(a+t[:, None]*(b-a)); weights.extend([length/n]*n)
        weights = np.asarray(weights)
        values.append(float(point_curve_distances(np.asarray(points), target)@weights/weights.sum()))
        bounds.append(float(np.sum(weights*weights)/(4*weights.sum())))
    return .5*sum(values), .5*sum(bounds)


def _disk_box_area(x0, y0, x1, y1, center, radius):
    cx, cy = center
    dx = max(x0-cx, cx-x1, 0); dy = max(y0-cy, cy-y1, 0)
    if dx*dx+dy*dy >= radius*radius:
        return 0.
    if max((x-cx)**2+(y-cy)**2 for x in (x0, x1) for y in (y0, y1)) <= radius*radius:
        return (x1-x0)*(y1-y0)
    lo, hi = max(x0, cx-radius), min(x1, cx+radius)
    def f(x):
        height = math.sqrt(max(0., radius*radius-(x-cx)**2))
        return max(0., min(y1, cy+height)-max(y0, cy-height))
    cuts = [lo, hi]
    for y in (y0, y1):
        if abs(y-cy) < radius:
            dx = math.sqrt(radius*radius-(y-cy)**2)
            cuts.extend([max(lo, min(hi, cx-dx)), max(lo, min(hi, cx+dx))])
    cuts = sorted(set(cuts))
    return sum(quad(f, a, b, epsabs=1e-12, epsrel=1e-10)[0]
               for a, b in zip(cuts, cuts[1:]) if b-a > 1e-12)


class PhysicalDistanceField:
    """Area-weighted fixed-grid quadrature of compact route distance fields."""
    def __init__(self, domain, endpoints, spacing=.4, radius=.5, endpoint_radius=1., shift=(0., 0.)):
        self.domain = tuple(domain); self.endpoints = np.asarray(endpoints, float)
        self.spacing, self.radius, self.endpoint_radius = float(spacing), float(radius), float(endpoint_radius)
        if spacing <= 0 or radius <= 0 or endpoint_radius < 0:
            raise ValueError('Invalid physical scales')
        if np.linalg.norm(self.endpoints[1]-self.endpoints[0]) <= 2*endpoint_radius:
            raise ValueError('This grid quadrature requires disjoint endpoint disks')
        x0, y0, x1, y1 = domain
        def edges(a, b, offset):
            interior = np.arange(a+offset, b+spacing, spacing)
            return np.unique(np.r_[a, interior[(interior>a)&(interior<b)], b])
        xx, yy = edges(x0, x1, shift[0]), edges(y0, y1, shift[1])
        points, weights = [], []
        for ya, yb in zip(yy, yy[1:]):
            for xa, xb in zip(xx, xx[1:]):
                area = (xb-xa)*(yb-ya)
                for ep in self.endpoints:
                    area -= _disk_box_area(xa, ya, xb, yb, ep, endpoint_radius)
                if area <= 1e-12:
                    continue
                center = np.array([(xa+xb)/2, (ya+yb)/2])
                if np.min(np.linalg.norm(self.endpoints-center, axis=1)) <= endpoint_radius:
                    corners = np.array([[xa, ya], [xa, yb], [xb, ya], [xb, yb]])
                    clearance = np.linalg.norm(corners[:, None, :]-self.endpoints, axis=2).min(axis=1)
                    center = corners[np.argmax(clearance)]
                points.append(center); weights.append(area)
        self.points = np.array(points); self.weights = np.array(weights)
        self.metadata = dict(domain_km=list(domain), spacing_km=spacing, support_radius_km=radius,
            endpoint_radius_km=endpoint_radius, grid_shift_km=list(shift), dimension=len(points),
            weight_area_km2=float(self.weights.sum()), area_weights='disk-box integration',
            representative='cell midpoint; endpoint-excluded midpoint replaced by maximum-clearance corner',
            distance='area-weighted soft Jaccard; not Euclidean', zero_union='explicit compression error')

    def compute(self, segments):
        phi = np.maximum(0, 1-point_curve_distances(self.points, segments)/self.radius)
        weighted = phi*self.weights
        if weighted.sum() <= 0:
            raise ValueError('Distance-field compression has zero mass')
        return weighted


def soft_jaccard(first, second):
    union = np.maximum(first, second).sum()
    if union <= 0:
        raise ValueError('Zero distance-field union')
    return float(np.clip(1-np.minimum(first, second).sum()/union, 0, 1))
