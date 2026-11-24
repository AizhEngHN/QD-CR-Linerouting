"""Summarize completed runs and optionally replay the fixed candidate streams."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from run_qdcr import BASELINES, CONTROLS, SCALES
from paper_inputs import load_case, read_json, write_json
from paper_statistics import metrics, paired, paired_main
from replay_worker import replay


def final_checkpoint(result):
    checkpoint = (result["checkpoint"] if result["task"]["arm"] == "Ramirez2025_native"
                  else result["checkpoints"][-1])
    if result["task"]["mode"] == "time" and not checkpoint["on_time"]:
        return dict(checkpoint, delivered=0, five=False, J_mean=None, D_min=None)
    return checkpoint


def summarize_comparison(group):
    labels = ["Full", *BASELINES]
    cases = sorted({key[:2] for key in group["Full"]})
    by_task = {
        f"{case}/{profile}": {
            label: metrics([row for key, row in group[label].items() if key[:2] == (case, profile)])
            for label in labels
        }
        for case, profile in cases
    }
    complete = [label for label in labels if all(row["five"] for row in group[label].values())]
    mean_rank = {}
    if complete:
        ranks = []
        for task in by_task.values():
            cost = rankdata([task[label]["mean_J"] for label in complete])
            separation = rankdata([-task[label]["mean_Dmin"] for label in complete])
            ranks.append(rankdata(cost + separation))
        mean_rank = dict(zip(complete, np.mean(ranks, axis=0).tolist()))
    return dict(
        by_task=by_task,
        mean_composite_rank=mean_rank,
        paired={label: paired_main(group["Full"], group[label]) for label in BASELINES},
    )


def replay_stream(root, index, task, checkpoint):
    stem = f"{index:04d}"
    spec = read_json(root / "runs" / f"{stem}_task.json")
    with gzip.open(root / "runs" / f"{stem}_data.json.gz", "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    records = {row["route_id"]: row for row in trace["records"]}
    rows = [records["bank:0"]]
    rows.extend(records[event["route_id"]] for event in trace["ledger"]
                if event["valid"] and not event.get("late_excluded", False))

    def load_map(case, profile):
        return load_case(spec["inputs"], case, profile)

    choices = ("Full", "D-no-seg", "SpatialGreedy", "TopCost")
    results = {label: replay(task, rows, label, load_map) for label in choices}
    own = results[task["label"]]
    if (own["selected"] != [row["route_id"] for row in checkpoint["selected"]]
            or (own["J"] is not None and abs(own["J"] - checkpoint["J_mean"]) > 1e-9)
            or (own["Dmin"] is not None and abs(own["Dmin"] - checkpoint["D_min"]) > 1e-9)):
        raise ValueError(f"Original portfolio did not reproduce for run {stem}")
    write_json(root / "replay" / f"{stem}.json",
               dict(task=task, source=task["label"], valid_input=len(rows), results=results,
                    own_replay_matches_original=True))
    return results


def summarize_replays(traces):
    report = {}
    for source, streams in traces.items():
        report[source] = {}
        for label in ("Full", "D-no-seg", "SpatialGreedy", "TopCost"):
            pairs = [(row["Full"], row[label]) for row in streams
                     if row["Full"]["count"] == row[label]["count"] == 5]
            differences = [(a["J"] - b["J"], a["Dmin"] - b["Dmin"]) for a, b in pairs]
            report[source][label] = dict(
                streams=len(streams), five=sum(row[label]["count"] == 5 for row in streams),
                matched_five=len(pairs),
                equal_scores=sum(abs(cost) <= 1e-9 and abs(sep) <= 1e-9 for cost, sep in differences),
                mean_full_minus_control=np.mean(differences, axis=0).tolist() if differences else None,
            )
    return report


def summarize(root, replay_enabled):
    plan = read_json(root / "plan.json")
    data = defaultdict(dict)
    replays = defaultdict(list)
    for index, task in enumerate(plan["tasks"]):
        result_path = root / "runs" / f"{index:04d}_result.json"
        result = read_json(result_path)
        if result["task"] != task:
            raise ValueError(f"Result task does not match plan: {result_path}")
        checkpoint = final_checkpoint(result)
        key = (task["case"], task["profile"], task["seed"])
        data[task["label"]][key] = checkpoint
        if replay_enabled and task["mode"] == "eval" and task["label"] in ("Full", "SpatialGreedy"):
            replays[task["label"]].append(replay_stream(root, index, task, checkpoint))
    report = dict(suite=plan["suite"], mode=plan["tasks"][0]["mode"],
                  aggregate={label: metrics(list(group.values())) for label, group in data.items()})
    if set(("Full", *BASELINES)) <= data.keys():
        report["comparison"] = summarize_comparison(data)
    rng = np.random.default_rng(536614)
    if "Full" in data:
        report["ablation_full_minus_control"] = {
            label: paired(data["Full"], data[label], rng) for label in CONTROLS if label in data
        }
        report["capacity_scale_minus_reference"] = {
            label: paired(data[label], data["Full"], rng) for label in SCALES if label in data
        }
    if "p=0.8" in data:
        report["probability_minus_reference"] = {
            label: paired(data[label], data["p=0.8"], rng) for label in sorted(data) if label != "p=0.8"
        }
    if replays:
        report["fixed_stream_replay"] = summarize_replays(replays)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--replay", action="store_true", help="Replay eval-mode QD-CR and Spatial greedy streams.")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        report = summarize(args.run_directory, args.replay)
    target = args.run_directory / "summary.json"
    write_json(target, report)
    print(target)


if __name__ == "__main__":
    main()
