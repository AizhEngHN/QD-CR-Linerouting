# QD-CR for alternative transmission line routing

This repository accompanies *Quality–diversity optimization with corridor replanning for alternative transmission line routing*. QD-CR combines perturbed graph proposals and corridor replanning with quality-ordered archive competition. A segmented, multiscale occupancy descriptor guides archive retention. The final selector delivers up to five routes subject to a minimum pairwise spatial difference.

## Quick start

Use Python 3.12. From the repository root:

```bash
python -m pip install -r requirements.txt
python run_qdcr.py --demo --output out/demo
python analyze.py out/demo --replay
```

The demo runs the paper's 200-evaluation QD-CR configuration on a small synthetic map. The result file contains the portfolios at 50, 100 and 200 route evaluations. The summary and replay are written to the same output directory.

## Rebuild the study maps

```bash
python -m pip install -r requirements-gis.txt
python prepare_maps.py --download
```

The script downloads the public tiles to `data/raw` and writes projected Base and Core100 inputs to `data/inputs`. Existing tiles can be used with `--raw /path/to/tiles`. Use `--case validation_henan_01` to build a single window. The [window configuration](config/study_windows.json) includes the six study windows, virtual endpoints, benchmark seeds and expected model hashes. Tile download checksums are in [input_tile_sha256.json](config/input_tile_sha256.json).

## Run the experiments

```bash
python run_qdcr.py --case validation_henan_01 --profile core100 --mode eval --output out/henan
python run_qdcr.py --suite ablation --mode eval --output out/ablation-eval
python analyze.py out/ablation-eval --replay
python run_qdcr.py --suite capacity-scale --mode eval --output out/capacity-scale
python run_qdcr.py --suite probability --mode time --output out/probability-time
python analyze.py out/probability-time
```

The main configuration uses archive capacity 20, Gaussian scale 3 km, and corridor-replanning probability 0.8 within the mutation branch. The evaluation protocol records checkpoints at 50, 100, and 200 complete route evaluations. The time protocol uses a 120 s end-to-end budget, with QD-CR search ending at 100 s and final selection ending at 119 s. The time limit covers initialization and selection. Timed route counts depend on hardware and system load. On Linux, `--jobs N --cpus C1 ... CN` runs parallel workers on the specified CPUs. Probability settings for a given task and seed run in random order on the same assigned CPU. Use one hardware thread from each physical core for parallel timing runs.

The `comparison` suite includes LION, Wang's ACO, the Ramírez clustering and refinement pipeline, and the Chen-inspired A* adaptation. Install the additional feature-extraction dependencies with `python -m pip install -r requirements-ramirez.txt` and supply the [original VGG16 checkpoint](https://download.pytorch.org/models/vgg16-397923af.pth) with `--weights /path/to/vgg16-397923af.pth`.

```bash
python run_qdcr.py --suite comparison --mode eval --weights /path/to/vgg16-397923af.pth --output out/comparison-eval
python run_qdcr.py --suite main --mode time --weights /path/to/vgg16-397923af.pth --output out/main-time
python analyze.py out/comparison-eval
python analyze.py out/main-time
```

The `main` suite runs the comparison and complete-search ablations in one batch. The `probability` suite supports both budgets. Use a distinct output directory for each suite and budget. The analysis scripts report five-route delivery, cost, separation, composite ranks, cost-matched results, and paired comparisons as applicable. The `--replay` option separately compares retention rules on recorded candidate streams.

## Source layout and provenance

| Path | Contents |
| --- | --- |
| `src/` | Route cost and geometry, spatial descriptor, archive retention, mutation, and final selection |
| `experiments/` | Budget workers, comparator implementations, replay, and statistics |
| `config/` | Study windows, public input checksums, and frozen source checksums |
| `prepare_maps.py` | Map construction from public data |
| `run_qdcr.py` and `analyze.py` | Portable run and analysis entry points |

The release extracts the paper's worker functions from the frozen experiment scripts and replaces the server-specific loaders and orchestration with local inputs. The suites execute fresh runs. The paper's fixed-evaluation tables used some previously recorded runs. The [frozen source checksums](config/frozen_source_sha256.json) identify the source versions used for the release.

## Geographic data attribution

- [ESA WorldCover 2021 v200](https://doi.org/10.5281/zenodo.7254221). © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium.
- [Copernicus DEM GLO-30](https://doi.org/10.5270/ESA-c5d3d65). Terrain inputs were produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
