"""Integer coordinates denote cell centres; touching a forbidden cell is blocked.

The supercover includes both side cells at a grid-corner crossing. This avoids
the diagonal corner cutting missed by ordinary Bresenham centre sampling.
"""

import math


def segment_cells(start, end):
    """Yield every closed grid cell touched by a centre-to-centre segment."""
    x, y = map(int, start)
    tx, ty = map(int, end)
    yield x, y
    dx, dy = tx - x, ty - y
    sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
    adx, ady = abs(dx), abs(dy)
    ix = iy = 0
    while ix < adx or iy < ady:
        # Exact integer comparison of next half-cell boundary parameters.
        left = (2 * ix + 1) * ady
        right = (2 * iy + 1) * adx
        if ix < adx and (iy == ady or left < right):
            x += sx
            ix += 1
        elif iy < ady and (ix == adx or left > right):
            y += sy
            iy += 1
        else:
            yield x + sx, y
            yield x, y + sy
            x += sx
            y += sy
            ix += 1
            iy += 1
        yield x, y


def segment_is_clear(grid, start, end):
    """Check all touched cells, including endpoints and corner contacts."""
    height, width = grid.shape
    for point in (start, end):
        if len(point) != 2 or any(not math.isfinite(v) or int(v) != v for v in point):
            return False
    for x, y in segment_cells(start, end):
        if not (0 <= x < width and 0 <= y < height) or grid[y, x] == -1:
            return False
    return True
