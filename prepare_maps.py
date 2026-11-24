"""Rebuild the paper's input grids from the public WorldCover and GLO-30 tiles."""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from paper_inputs import core_from_arrays, read_json

WORLD_COVER = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
ELEVATION = "https://copernicus-dem-30m.s3.amazonaws.com"
LAND_SCORES = {
    10: 0.5, 20: 0.25, 30: 0.0, 40: 0.25, 50: 1.0, 60: 0.0,
    70: 0.75, 80: 0.75, 90: 0.75, 95: 0.75, 100: 0.25,
}
PROFILES = ("base", "core100")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def obtain_tile(raw_dir, filename, url, expected, allow_download):
    path = raw_dir / filename
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"Missing {path}. Supply the tile or use --download.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{filename}.part")
        print(f"Downloading {filename}", flush=True)
        with urlopen(url, timeout=60) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
        if digest(temporary) != expected:
            raise ValueError(f"Downloaded tile checksum differs from the paper input: {filename}")
        temporary.replace(path)
    elif digest(path) != expected:
        raise ValueError(f"Tile checksum differs from the paper input: {filename}")
    return path


def warp(source, target, transform, crs, resampling, nodata):
    with rasterio.open(source) as dataset:
        reproject(
            rasterio.band(dataset, 1), target,
            src_transform=dataset.transform, src_crs=dataset.crs,
            src_nodata=dataset.nodata, dst_transform=transform, dst_crs=crs,
            dst_nodata=nodata, resampling=resampling, num_threads=2,
        )


def aggregate(array):
    return array.reshape(300, 10, 300, 10).mean(axis=(1, 3), dtype=np.float64)


def build_window(case, endpoints, raw_dir, output, hashes, expected_models, allow_download):
    case_id = case["id"]
    tile = case["worldcover"]
    name = f"WorldCover_2021_v200_{tile}.tif"
    cover = obtain_tile(raw_dir, name, f"{WORLD_COVER}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif",
                        hashes[name], allow_download)
    x0, _, _, y1 = case["bounds_utm_m"]
    crs = case["crs"]
    fine_transform = from_origin(x0, y1, 10, 10)
    cover_classes = np.zeros((3000, 3000), np.uint8)
    warp(cover, cover_classes, fine_transform, crs, Resampling.nearest, 0)
    if not np.isin(cover_classes, tuple(LAND_SCORES)).all():
        raise ValueError(f"Land-cover tile does not cover {case_id}")
    q_land = np.zeros(cover_classes.shape, np.float32)
    for code, score in LAND_SCORES.items():
        q_land[cover_classes == code] = score

    dem_transform = from_origin(x0 - 300, y1 + 300, 30, 30)
    elevation = np.full((1020, 1020), np.nan, np.float32)
    for tile in case["dem"]:
        stem = f"Copernicus_DSM_COG_10_{tile}_DEM"
        name = f"{stem}.tif"
        source = obtain_tile(raw_dir, name, f"{ELEVATION}/{stem}/{name}", hashes[name], allow_download)
        projected = np.full_like(elevation, np.nan)
        warp(source, projected, dem_transform, crs, Resampling.bilinear, np.nan)
        elevation = np.where(np.isfinite(elevation), elevation, projected)
    if not np.isfinite(elevation).all():
        raise ValueError(f"DEM tiles do not cover the padded window for {case_id}")
    dy, dx = np.gradient(elevation, 30, 30)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    q_terrain = np.full(cover_classes.shape, np.nan, np.float32)
    reproject(
        np.minimum(slope / 30.0, 1), q_terrain,
        src_transform=dem_transform, src_crs=crs,
        dst_transform=fine_transform, dst_crs=crs, resampling=Resampling.average,
    )
    if not np.isfinite(q_terrain).all():
        raise ValueError(f"Slope raster contains missing cells for {case_id}")

    hard = (cover_classes == 50).reshape(300, 10, 300, 10).sum(axis=(1, 3)) == 100
    points = np.asarray(endpoints, dtype=float)
    common = dict(
        resistance=np.flipud(aggregate(0.7 * q_land + 0.3 * q_terrain)),
        q_land=np.flipud(aggregate(q_land)), q_terrain=np.flipud(aggregate(q_terrain)),
        start=(points[0] - 0.05) / 0.1, goal=(points[1] - 0.05) / 0.1,
        resolution_m=100, origin_km=[0.05, 0.05], domain_km=[0.0, 0.0, 30.0, 30.0],
        alpha=1.0, kappa=0.5,
    )
    for profile in PROFILES:
        data = dict(common, forbidden=np.flipud(hard) if profile == "core100" else np.zeros_like(hard))
        model_hash = core_from_arrays(data).model_sha256
        expected = expected_models.get(f"{case_id}/{profile}")
        if expected and expected != model_hash:
            raise ValueError(f"Rebuilt model differs from the frozen inputs for {case_id}/{profile}: {model_hash}")
        target = output / case_id / profile / "input_100m.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".part.npz")
        np.savez_compressed(temporary, **data)
        temporary.replace(target)
        print(f"{case_id}/{profile} model SHA-256 {model_hash}", flush=True)


def main():
    config = read_json(ROOT / "config/study_windows.json")
    cases = {case["id"]: case for case in config["validations"] + config["development"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/inputs"))
    parser.add_argument("--case", choices=cases, action="append", help="Repeat to select specific study windows.")
    parser.add_argument("--download", action="store_true", help="Download missing public WorldCover and GLO-30 tiles.")
    args = parser.parse_args()
    hashes = read_json(ROOT / "config/input_tile_sha256.json")
    for case_id in args.case or cases:
        build_window(cases[case_id], config["endpoints_local_km"], args.raw, args.output,
                     hashes, config["expected_model_sha256"], args.download)


if __name__ == "__main__":
    main()
