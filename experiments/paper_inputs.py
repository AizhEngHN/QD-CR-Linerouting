"""Map loading, the frozen descriptor, and result serialization."""
import gzip
import json
from pathlib import Path

import numpy as np

from corridor_model import CorridorEvaluator, CorridorMap, CorridorWeights
from gis_comparison_adapter import legacy_for
from soft_occupancy import SoftOccupancyDescriptor


def core_from_arrays(arrays):
    grid = CorridorMap(
        arrays["forbidden"], arrays["resistance"], arrays["start"], arrays["goal"],
        resolution_m=float(arrays["resolution_m"]),
        origin_km=arrays["origin_km"], domain_km=arrays["domain_km"],
    )
    return CorridorEvaluator(grid, CorridorWeights(float(arrays["alpha"]), float(arrays["kappa"])))


def load_case(inputs, case, profile):
    with np.load(Path(inputs) / case / profile / "input_100m.npz") as arrays:
        core = core_from_arrays(arrays)
        layers = {name: arrays[name].copy() for name in ("q_land", "q_terrain")}
    return core, layers


def descriptor_for(core, label):
    segments = 1 if label == "D-no-seg" else 4
    return SoftOccupancyDescriptor(
        legacy_for(core, descriptor=True), segments=segments, spacings_km=(0.4, 0.8),
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, data, *, compressed=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    open_text = gzip.open if compressed else open
    with open_text(temporary, "wt", encoding="utf-8") as stream:
        json.dump(data, stream, allow_nan=False, default=json_value, indent=None if compressed else 2)
        stream.write("\n")
    temporary.replace(path)
