"""Paired statistics and cost-matching rules used for the paper."""
from collections import Counter, defaultdict
from statistics import mean

import numpy as np



def metrics(rows):
    five = [r for r in rows if r["five"]]
    return {
        "runs": len(rows),
        "five_route_deliveries": len(five),
        "mean_J": mean(r["J_mean"] for r in five) if five else None,
        "mean_Dmin": mean(r["D_min"] for r in five) if five else None,
        "mean_evaluations": mean(r["evaluations"] for r in rows),
        "late": sum(not r.get("on_time", True) for r in rows),
        "certified_five_portfolios": sum(r["selection"]["certified"] for r in five),
    }


def outcome(a, b):
    if a["delivered"] != b["delivered"]:
        return "win" if a["delivered"] > b["delivered"] else "loss"
    if not a["five"]:
        return "incomplete"
    cost, separation = a["J_mean"] - b["J_mean"], a["D_min"] - b["D_min"]
    tol = 1e-9
    if abs(cost) <= tol and abs(separation) <= tol:
        return "tie"
    if cost <= tol and separation >= -tol:
        return "win"
    if cost >= -tol and separation <= tol:
        return "loss"
    return "tradeoff"


def paired(variant, reference, rng):
    assert set(variant) == set(reference)
    strata = defaultdict(list)
    counts = Counter()
    for item in sorted(variant):
        a, b = variant[item], reference[item]
        counts[outcome(a, b)] += 1
        if a["five"] and b["five"]:
            strata[item[:2]].append((a["J_mean"] - b["J_mean"], a["D_min"] - b["D_min"]))
    report = {"outcomes": {c: counts[c] for c in ("win", "tie", "loss", "tradeoff", "incomplete")},
              "matched_five": sum(map(len, strata.values())), "mean_delta": None, "ci95": None,
              "task_mean_delta": {}}
    if strata:
        rows = {"/".join(k): np.asarray(v) for k, v in sorted(strata.items())}
        report["mean_delta"] = np.vstack(list(rows.values())).mean(axis=0).tolist()
        report["task_mean_delta"] = {k: v.mean(axis=0).tolist() for k, v in rows.items()}
        if len(rows) == 6:
            samples = sum(v[rng.integers(0, len(v), (10000, len(v)))].sum(axis=1) for v in rows.values())
            report["ci95"] = np.quantile(samples / report["matched_five"], [.025, .975], axis=0).T.tolist()
    return report


def paired_main(full, comparator):
    q, b = [], []
    for task in sorted({k[:2] for k in full}):
        seeds = sorted(k for k in full if k[:2] == task)
        references = sorted(k for k in comparator if k[:2] == task)
        if not references or (len(seeds) == len(references) and seeds != references):
            raise ValueError(f"Unmatched comparison runs for {task}")
        for i,k in enumerate(seeds):
            q.append(full[k])
            b.append(comparator[k] if len(references) == len(seeds) else comparator[references[i % len(references)]])
    outcomes = Counter(outcome(a,z) for a,z in zip(q,b))
    cost_matches = {}
    for tolerance in (.5,1.):
        differences = [a["D_min"]-z["D_min"] for a,z in zip(q,b)
                       if a["five"] and z["five"] and abs(a["J_mean"]-z["J_mean"]) <= tolerance]
        cost_matches[str(tolerance)] = {"n":len(differences), "higher":sum(d > 1e-9 for d in differences),
            "mean_delta_Dmin":mean(differences) if differences else None}
    return {"pairs":len(q), "outcomes":dict(outcomes), "cost_matches":cost_matches}
