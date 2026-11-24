"""Fixed-B retention and shared cost-constrained geometric portfolio selection.

Pair matrices may be precomputed for replay; the policy accesses only retained
records plus the current offer. This is not a timed online-search benchmark.
"""
from __future__ import annotations
import itertools
import math
import numpy as np
from replay_retention import RetentionPolicy


class SpatialBudgetRetention(RetentionPolicy):
    def __init__(self, initial, selector, budget, indices, distances, capacity=20):
        initial=list(initial)
        if selector not in ('competition','greedy','topk') or not initial:
            raise ValueError('Invalid selector/initialization')
        if any(r.cost>budget for r in initial):
            raise ValueError('Initial archive exceeds budget')
        self.selector=selector; self.budget=budget; self.indices=indices; self.distances=distances
        super().__init__(initial,method='cost_ordered',capacity=len(initial))
        self.capacity=capacity

    def _distance(self, a, b):
        return float(self.distances[self.indices[a.route_id],self.indices[b.route_id]])

    def offer(self, candidate):
        before=self._records.copy(); routes=before.copy(); accepted=False
        duplicate=next((r for r in routes if r.geometry_key==candidate.geometry_key),None)
        reason='outside_fixed_budget'
        if candidate.cost<=self.budget:
            if duplicate is not None:
                if self._raw_cost_order(candidate)<self._raw_cost_order(duplicate):
                    routes=[r for r in routes if r.route_id!=duplicate.route_id]+[candidate]
                    reason='same_geometry_better'
                else:
                    reason='same_geometry_not_better'
            else:
                routes.append(candidate);reason='new_geometry'
            if len(routes)<=self.capacity:
                selected=routes
            elif self.selector=='competition':
                selected,_=self._competition(routes)
            elif self.selector=='topk':
                selected=sorted(routes,key=self._cost_order)[:self.capacity]
            else:
                ordered=sorted(routes,key=self._cost_order)
                selected=[ordered[0]];remaining=ordered[1:]
                nearest={r.route_id:self._distance(r,selected[0]) for r in remaining}
                while remaining and len(selected)<self.capacity:
                    chosen=min(remaining,key=lambda r:(-nearest[r.route_id],self._raw_cost_order(r)))
                    selected.append(chosen);remaining=[r for r in remaining if r.route_id!=chosen.route_id]
                    for r in remaining:nearest[r.route_id]=min(nearest[r.route_id],self._distance(r,chosen))
            self._records=selected
            accepted=any(r.route_id==candidate.route_id for r in selected)
        self.stats['offers']+=1
        self.stats['accepted' if accepted else 'rejected']+=1
        self.stats['geometry_duplicate_offers']+=int(duplicate is not None)
        ids={r.route_id for r in self._records}
        return dict(route_id=candidate.route_id,accepted=accepted,reason=reason,
                    retained_ids=[r.route_id for r in self._records],
                    deleted_ids=[r.route_id for r in before if r.route_id not in ids])


def portfolio_score(selected, distance, costs):
    if len(selected)<2:
        return (0.,0.,-float(np.mean(costs[selected])))
    values=[distance(a,b) for a,b in itertools.combinations(selected,2)]
    return min(values),float(np.mean(values)),-float(np.mean(costs[selected]))


def _strictly_better(a,b,tol=1e-12):
    for x,y in zip(a,b):
        if x>y+tol:return True
        if x<y-tol:return False
    return False


def select_spatial_portfolio(pool, costs, distance, stable_keys, nominal_cost, average_cap,
                             target=5, swap_passes=5):
    """Same heuristic for every pool, with cheapest-completion feasibility."""
    ordered=sorted(set(pool),key=lambda i:(costs[i],stable_keys[i]))
    if not ordered:return dict(selected=[],status='empty_pool',swap_passes=0)
    cheapest=ordered[0];total_cap=average_cap*nominal_cost*target
    if len(ordered)<target:
        partial=[]
        for i in ordered:
            if sum(costs[j] for j in partial+[i])/(len(partial)+1)<=average_cap*nominal_cost+1e-12:
                partial.append(i)
        return dict(selected=partial,status='fewer_than_five_qualified',swap_passes=0)
    if sum(costs[i] for i in ordered[:target])>total_cap+1e-12:
        return dict(selected=[],status='five_cheapest_exceed_average_cap',swap_passes=0)
    selected=[cheapest]
    while len(selected)<target:
        candidates=[]
        for i in ordered:
            if i in selected:continue
            trial=selected+[i]
            completion=[j for j in ordered if j not in trial][:target-len(trial)]
            if sum(costs[j] for j in trial+completion)<=total_cap+1e-12:
                candidates.append(i)
        if not candidates:raise AssertionError('Cheapest-completion invariant failed')
        chosen=min(candidates,key=lambda i:(-min(distance(i,j) for j in selected),costs[i],stable_keys[i]))
        selected.append(chosen)
    passes=0
    for _ in range(swap_passes):
        baseline=portfolio_score(selected,distance,costs);best_score=baseline;best=None;best_key=None
        for old in selected:
            if old==cheapest:continue
            for new in ordered:
                if new in selected:continue
                trial=[new if j==old else j for j in selected]
                if sum(costs[j] for j in trial)>total_cap+1e-12:continue
                score=portfolio_score(trial,distance,costs)
                key=tuple(sorted(stable_keys[j] for j in trial))
                if _strictly_better(score,best_score) or (best is not None and
                    not _strictly_better(best_score,score) and not _strictly_better(score,best_score) and key<best_key):
                    best_score,best,best_key=score,trial,key
        if best is None:break
        selected=best;passes+=1
    return dict(selected=selected,status='complete_five',swap_passes=passes,
                score=list(portfolio_score(selected,distance,costs)))


def effective_count(selected, distance, threshold=.2):
    for size in range(len(selected),0,-1):
        if any(all(distance(a,b)>=threshold for a,b in itertools.combinations(subset,2))
               for subset in itertools.combinations(selected,size)):
            return size
    return 0


def exact_portfolio_audit(pool,costs,distance,stable_keys,nominal_cost,average_cap,target=5):
    """Small bounded-pool audit only; never used to improve one method's output."""
    pool=sorted(set(pool),key=lambda i:(costs[i],stable_keys[i]))
    if len(pool)<target:return None
    cheapest=pool[0];best=None;count=0
    for tail in itertools.combinations(pool[1:],target-1):
        selected=[cheapest,*tail]
        if sum(costs[j] for j in selected)>target*nominal_cost*average_cap+1e-12:continue
        score=portfolio_score(selected,distance,costs);count+=1
        if best is None or _strictly_better(score,best):best=score
    return dict(feasible_subsets=count,best_score=list(best) if best is not None else None)
