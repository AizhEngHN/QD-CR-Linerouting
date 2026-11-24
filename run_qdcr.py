"""Run QD-CR, the paper's comparison methods, or a complete experiment suite."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from paper_inputs import read_json, write_json

BASELINES = ("LION2021", "Wang2024", "Ramirez2025_native", "Chen2026_perturbed")
CONTROLS = ("D-no-seg", "SpatialGreedy", "TopCost", "NoCorridor")
SCALES = ("K10", "K30", "SIG2", "SIG5")
MAIN = ("Full", *BASELINES, *CONTROLS)
PROFILES = ("base", "core100")
PROBABILITIES = (0.1, 0.2, 0.4, 0.6, 0.8)


def configuration(label, probability=None):
    settings = dict(capacity=20, sigma=3.0, prob=0.8)
    overrides = {
        "K10": {"capacity": 10}, "K30": {"capacity": 30},
        "SIG2": {"sigma": 2.0}, "SIG5": {"sigma": 5.0},
        "NoCorridor": {"prob": 0.0},
    }
    settings.update(overrides.get(label, {}))
    if probability is not None:
        settings["prob"] = probability
    return settings


def task(case, profile, seed, mode, label, probability=None):
    arm = label if label in (*BASELINES, "SpatialGreedy") else "QD_Hybrid"
    return dict(case=case, profile=profile, seed=seed, mode=mode, label=label,
                arm=arm, configuration=configuration(label, probability))


def task_blocks(args, study):
    if args.suite == "single":
        case = args.case[0] if args.case else study["validations"][0]["id"]
        profile = args.profile[0] if args.profile else "base"
        seed = args.seed[0] if args.seed else study["seeds"][0]
        if any(len(values or ()) > 1 for values in (args.case, args.profile, args.seed)):
            raise ValueError("A single run uses one case, profile, and seed.")
        return [[task(case, profile, seed, args.mode, args.variant, args.probability)]]
    if args.probability is not None:
        raise ValueError("--probability applies to a single run.")
    if args.suite == "capacity-scale" and args.mode != "eval":
        raise ValueError("The paper's capacity/scale sensitivity uses the evaluation budget.")
    cases = args.case or [case["id"] for case in study["validations"]]
    seeds = args.seed or study["seeds"]
    triples = list(itertools.product(cases, args.profile or PROFILES, seeds))
    if args.suite == "probability":
        rng = random.Random(202611241)
        rng.shuffle(triples)
        blocks = []
        for case, profile, seed in triples:
            levels = list(PROBABILITIES)
            rng.shuffle(levels)
            blocks.append([task(case, profile, seed, args.mode, f"p={p:.1f}", p) for p in levels])
        return blocks

    arms = {
        "main": MAIN, "comparison": ("Full", *BASELINES),
        "ablation": ("Full", *CONTROLS), "capacity-scale": ("Full", *SCALES),
    }[args.suite]
    rows = [task(case, profile, seed, args.mode, label)
            for case, profile, seed in triples for label in arms
            if label not in ("LION2021", "Ramirez2025_native") or seed in seeds[:3]]
    random.Random(2026112308).shuffle(rows)
    return [[row] for row in rows]


def make_demo(output):
    import numpy as np

    rng = np.random.default_rng(2026)
    y, x = np.indices((61, 61))
    field = 0.2 + 0.25 * np.exp(-((x - 30) ** 2 + (y - 30) ** 2) / 210)
    field += rng.uniform(0.0, 0.1, field.shape)
    inputs = output / "demo_inputs"
    folder = inputs / "demo" / "base"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / "input_100m.npz", resistance=field, q_land=field, q_terrain=np.zeros_like(field),
        forbidden=np.zeros_like(field, dtype=bool), start=[5, 5], goal=[55, 55], resolution_m=100,
        origin_km=[0.05, 0.05], domain_km=[0.0, 0.0, 6.1, 6.1], alpha=1.0, kappa=0.5,
    )
    return inputs


def execute(request_path, started, cpu):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    request = read_json(request_path)
    item = request["task"]
    result_path = request_path.with_name(request_path.name.replace("_task.json", "_result.json"))
    trace_path = request_path.with_name(request_path.name.replace("_task.json", "_data.json.gz"))
    from threadpoolctl import threadpool_limits
    from paper_inputs import load_case

    def load_map(case, profile):
        return load_case(request["inputs"], case, profile)

    with threadpool_limits(limits=1):
        if item["arm"] == "Ramirez2025_native":
            from ramirez_worker import worker

            native = worker(item, started, result_path, load_map, Path(request["weights"]))
            result = {k: v for k, v in native.items() if k not in ("records", "ledger")}
            trace = {k: native[k] for k in ("task", "records", "ledger", "checkpoint")}
        else:
            from fixed_worker import worker as fixed
            from timed_worker import worker as timed

            run_worker = fixed if item["mode"] == "eval" else timed
            result, trace = run_worker(item, started, load_map)
    result["cpu"] = cpu
    write_json(trace_path, trace, compressed=True)
    write_json(result_path, result)


def run_blocks(args, blocks):
    output = args.output.resolve()
    inputs = args.inputs.resolve()
    weights = args.weights.resolve() if args.weights else None
    # Reserve the same CPU for every configuration in a task/seed block.
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    cpus = args.cpus or allowed[:args.jobs]
    if args.cpus and any(cpu not in allowed for cpu in cpus):
        raise ValueError("--cpus requires Linux and CPUs in the current affinity mask.")
    if cpus and len(cpus) < args.jobs:
        raise ValueError("Specify at least one CPU per parallel worker.")
    if not cpus:
        cpus = [None] * args.jobs
    tasks = list(itertools.chain.from_iterable(blocks))
    if any(item["arm"] == "Ramirez2025_native" for item in tasks) and (not weights or not weights.is_file()):
        raise FileNotFoundError("Supply the original VGG16 checkpoint using --weights for the Ramírez baseline.")
    for case, profile in {(item["case"], item["profile"]) for item in tasks}:
        path = inputs / case / profile / "input_100m.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run prepare_maps.py to construct the input.")

    plan = dict(suite=args.suite, tasks=tasks, timed_protocol=dict(
        search_cutoff_seconds=100, selection_deadline_seconds=119, end_to_end_seconds=120,
    ))
    plan_path = output / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError(f"Existing output has a different task plan: {output}")
    write_json(plan_path, plan)
    runs = output / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = "1"
    indexes = iter(range(len(tasks)))
    indexed_blocks = [[(next(indexes), item) for item in block] for block in blocks]

    def run_lane(lane):
        cpu = cpus[lane]
        for block in indexed_blocks[lane::args.jobs]:
            for index, item in block:
                request = runs / f"{index:04d}_task.json"
                result = runs / f"{index:04d}_result.json"
                trace = runs / f"{index:04d}_data.json.gz"
                spec = dict(task=item, inputs=str(inputs), weights=str(weights) if weights else None)
                if result.exists():
                    if not trace.is_file() or not request.is_file() or read_json(request) != spec:
                        raise ValueError(f"Existing result does not match its inputs: {result}")
                    previous = read_json(result)
                    if item["mode"] == "time" and len(block) > 1 and previous["cpu"] != cpu:
                        raise ValueError(f"Timed block resumed with a different CPU at {result}. Use a new output directory.")
                    continue
                write_json(request, spec)
                command = [sys.executable, str(Path(__file__).resolve()), "--execute", str(request),
                           "--started", repr(time.perf_counter())]
                if cpu is not None:
                    command.extend(("--cpu", str(cpu)))
                log_path = runs / f"{index:04d}_worker.log"
                with log_path.open("ab") as log:
                    subprocess.run(command, check=True, env=env, stdout=log, stderr=log,
                                   timeout=150 if item["mode"] == "time" else 1050)
                if not result.is_file() or not trace.is_file():
                    raise RuntimeError(f"Missing worker output. See {log_path}")
                print(f"{index + 1}/{len(tasks)} {item['label']} {item['case']}/{item['profile']} {item['seed']}", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run_lane, range(args.jobs)))


def main():
    study = read_json(ROOT / "config/study_windows.json")
    ids = [case["id"] for case in study["validations"] + study["development"]]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Run QD-CR on a small synthetic map.")
    parser.add_argument("--suite", choices=("single", "main", "comparison", "ablation", "capacity-scale", "probability"),
                        default="single")
    parser.add_argument("--variant", choices=("Full", *BASELINES, *CONTROLS, *SCALES), default="Full")
    parser.add_argument("--probability", type=float, help="Override the corridor-replanning probability in a single run.")
    parser.add_argument("--mode", choices=("eval", "time"), default="eval")
    parser.add_argument("--case", action="append", choices=ids)
    parser.add_argument("--profile", action="append", choices=PROFILES)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--inputs", type=Path, default=Path("data/inputs"))
    parser.add_argument("--output", type=Path, default=Path("out/demo"))
    parser.add_argument("--weights", type=Path, help="Path to the public VGG16 weights.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--cpus", nargs="+", type=int)
    parser.add_argument("--execute", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--started", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--cpu", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute:
        execute(args.execute, args.started, args.cpu)
        return
    if args.jobs < 1 or (args.cpus and len(set(args.cpus)) != len(args.cpus)):
        parser.error("Use a positive worker count and distinct CPUs.")
    if args.probability is not None and not 0 <= args.probability <= 1:
        parser.error("--probability must be between 0 and 1.")
    if args.demo:
        if args.suite != "single":
            parser.error("--demo supports --suite single.")
        args.case, args.profile = ["demo"], ["base"]
        args.inputs = make_demo(args.output)
    run_blocks(args, task_blocks(args, study))


if __name__ == "__main__":
    main()
