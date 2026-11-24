"""Exact fixed-cardinality, anchored, cost-constrained max-min portfolios.

Small pools enumerate every feasible combination. Larger pools binary-search
pairwise distance thresholds with budgeted clique feasibility, then enumerate
optimal-threshold cliques for the same mean-distance/cost tie objectives.
No external optimizer is required. Numerical comparisons use declared 1e-12
tolerance; a timed-out result is explicitly uncertified, never called optimal.
"""
from __future__ import annotations
import itertools
import math
import time
import numpy as np


TOL=1e-12


class SelectionTimeout(RuntimeError):
    pass


def _better(score,reference):
    if reference is None:return True
    for a,b in zip(score,reference):
        if a>b+TOL:return True
        if a<b-TOL:return False
    return False


def _equal(a,b):
    return not _better(a,b) and not _better(b,a)


def select_exact_portfolio(pool,costs,distance,stable_keys,nominal_cost,average_cap,
                           target=5,deadline=math.inf,engine='auto'):
    """Return exact primary/secondary objectives, or an explicit timeout.

The caller supplies already legal, per-route-budget-qualified unique geometry.
The cheapest representative is mandatory. The same objective is used for any
pool size, including full history. Input indices identify actual representations
and costs; no cost of a forgotten representation is substituted.
"""
    if target<2 or engine not in ('auto','enumerate','clique'):
        raise ValueError('Invalid target/engine')
    if nominal_cost<=0 or average_cap<=0 or not math.isfinite(nominal_cost*average_cap):
        raise ValueError('Positive finite nominal cost and cap required')
    costs=np.asarray(costs,float)
    if any(not math.isfinite(costs[i]) or costs[i]<=0 for i in pool):
        raise ValueError('Positive finite candidate costs required')
    ordered=sorted(set(pool),key=lambda i:(costs[i],stable_keys[i]));start=time.perf_counter()
    base=dict(optimization='anchored fixed-cardinality lexicographic max(D_min,D_mean,-meanJ)',
              tolerance=TOL,optimal=False,engine=None,enumerated_feasible_sets=0,
              feasibility_nodes=0,selection_seconds=0.,primary_gap=None)
    if len(ordered)<target:
        partial=[]
        for i in ordered:
            if sum(costs[j] for j in partial+[i])<=(len(partial)+1)*nominal_cost*average_cap+TOL:partial.append(i)
        return dict(base,selected=partial,status='fewer_than_five_qualified',optimal=True,
                    selection_seconds=time.perf_counter()-start)
    limit=target*nominal_cost*average_cap
    if sum(costs[i] for i in ordered[:target])>limit+TOL:
        return dict(base,selected=[],status='five_cheapest_exceed_average_cap',optimal=True,
                    selection_seconds=time.perf_counter()-start)
    n=len(ordered);local_cost=costs[ordered];best=None;best_set=None;best_key=None
    matrix=np.full((n,n),np.nan);np.fill_diagonal(matrix,0)
    def tick():
        if time.perf_counter()>deadline:raise SelectionTimeout()
    def d(i,j):
        if np.isnan(matrix[i,j]):
            tick();value=float(distance(ordered[i],ordered[j]))
            if not math.isfinite(value) or value < -TOL or value>1+TOL:raise ValueError('Geometric difference must be in [0,1]')
            matrix[i,j]=matrix[j,i]=max(0.,min(1.,value))
        return matrix[i,j]
    def consider(selection,score=None):
        nonlocal best,best_set,best_key
        base['enumerated_feasible_sets']+=1
        if score is None:
            values=[d(i,j) for i,j in itertools.combinations(selection,2)]
            score=(float(min(values)),float(np.mean(values)),-float(np.mean(local_cost[list(selection)])))
        key=tuple(sorted(stable_keys[ordered[i]] for i in selection))
        if _better(score,best) or (best is not None and _equal(score,best) and key<best_key):
            best,best_set,best_key=tuple(score),list(selection),key
    initial=list(range(target))
    try:
        consider(initial)
        if engine=='enumerate' or (engine=='auto' and n<=24):
            base['engine']='enumeration'
            for tail in itertools.combinations(range(1,n),target-1):
                tick();sel=(0,*tail)
                if local_cost[list(sel)].sum()<=limit+TOL:consider(sel)
        else:
            base['engine']='budgeted_clique_threshold'
            for i in range(n):
                for j in range(i):d(i,j)
            thresholds=np.unique(matrix[np.triu_indices(n,1)])
            node_count=0
            def masks_for(threshold):
                graph=matrix>=threshold-TOL;np.fill_diagonal(graph,False)
                masks=[sum(1<<int(j) for j in np.flatnonzero(graph[i])) for i in range(n)]
                candidates=masks[0]&~1
                return masks,candidates
            def lower_cost(mask,needed):
                total=0.
                for _ in range(needed):
                    if not mask:return math.inf
                    bit=mask&-mask;j=bit.bit_length()-1;total+=local_cost[j];mask^=bit
                return total
            def feasible(threshold):
                masks,candidates=masks_for(threshold)
                def visit(mask,needed,spent,chosen):
                    nonlocal node_count
                    node_count+=1;tick()
                    if needed==0:return chosen
                    if mask.bit_count()<needed or spent+lower_cost(mask,needed)>limit+TOL:return None
                    while mask:
                        bit=mask&-mask;i=bit.bit_length()-1;mask^=bit
                        if spent+local_cost[i]>limit+TOL:break
                        result=visit(mask&masks[i],needed-1,spent+local_cost[i],chosen+[i])
                        if result is not None:return result
                    return None
                return visit(candidates,target-1,local_cost[0],[0])
            lo,hi=0,len(thresholds)-1;opt_index=-1;feasible_seed=None
            while lo<=hi:
                mid=(lo+hi)//2;solution=feasible(thresholds[mid])
                if solution is not None:opt_index=mid;feasible_seed=solution;lo=mid+1
                else:hi=mid-1
            assert opt_index>=0
            optimum=float(thresholds[opt_index]);consider(feasible_seed)
            masks,candidates=masks_for(optimum)
            def visit_optimal(mask,needed,spent,chosen):
                nonlocal node_count
                node_count+=1;tick()
                if mask.bit_count()<needed or spent+lower_cost(mask,needed)>limit+TOL:return
                if needed==0:consider(chosen);return
                if needed==2:
                    indices=[];remaining=mask
                    while remaining:
                        bit=remaining&-remaining;indices.append(bit.bit_length()-1);remaining^=bit
                    ids=np.array(indices,int);aa,bb=np.triu_indices(len(ids),1)
                    a,b=ids[aa],ids[bb]
                    valid=(spent+local_cost[a]+local_cost[b]<=limit+TOL)&(matrix[a,b]>=optimum-TOL)
                    a,b=a[valid],b[valid]
                    if not len(a):return
                    fixed=list(itertools.combinations(chosen,2))
                    fixed_values=[matrix[x,y] for x,y in fixed]
                    cross=matrix[np.ix_(ids,chosen)]
                    link_sum=np.zeros(n);link_min=np.ones(n)
                    link_sum[ids]=cross.sum(axis=1);link_min[ids]=cross.min(axis=1)
                    sums=sum(fixed_values)+link_sum[a]+link_sum[b]+matrix[a,b]
                    mins=np.minimum(np.minimum(link_min[a],link_min[b]),matrix[a,b])
                    if fixed_values:mins=np.minimum(mins,min(fixed_values))
                    means=sums/(target*(target-1)/2)
                    # Preserve every tie that could matter to the cost/stable-key
                    # stages, while avoiding Python work for dominated pairs.
                    max_min=float(mins.max());keep=mins>=max_min-TOL
                    max_mean=float(means[keep].max());keep&=means>=max_mean-TOL
                    candidate_cost=spent+local_cost[a]+local_cost[b]
                    min_cost=float(candidate_cost[keep].min());keep&=candidate_cost<=min_cost+target*TOL
                    base['enumerated_feasible_sets']+=len(a)-int(keep.sum())
                    for k in np.flatnonzero(keep):
                        consider(chosen+[int(a[k]),int(b[k])],(float(mins[k]),float(means[k]),-float(candidate_cost[k]/target)))
                    return
                while mask:
                    bit=mask&-mask;i=bit.bit_length()-1;mask^=bit
                    visit_optimal(mask&masks[i],needed-1,spent+local_cost[i],chosen+[i])
            visit_optimal(candidates,target-1,local_cost[0],[0]);base['feasibility_nodes']=node_count
        tick()
        return dict(base,selected=[ordered[i] for i in best_set],status='complete_five',optimal=True,
                    score=list(best),primary_gap=0.,selection_seconds=time.perf_counter()-start)
    except SelectionTimeout:
        return dict(base,selected=[ordered[i] for i in best_set] if best_set else [],
                    status='timeout_uncertified',optimal=False,score=list(best) if best else None,
                    selection_seconds=time.perf_counter()-start)
