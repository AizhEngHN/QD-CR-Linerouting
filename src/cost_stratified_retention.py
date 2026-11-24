"""Predeclared cost-stratified spatial greedy, with K-matched online state.

Four strata are fixed against pre-search p0, not current best. Each selects
cheapest-seeded spatial max-min representatives; missing quotas are filled
globally from current+candidate. This is a control, not a new QD claim.
"""
from __future__ import annotations
import math
from closedloop_spatial_run import OnlineFixedBudgetRetention,FixedBudgetCandidateRun


class CostStratifiedRetention(OnlineFixedBudgetRetention):
    def __init__(self,initial,nominal_cost,budget,quotas=(5,5,5,5)):
        if not math.isfinite(nominal_cost) or nominal_cost<=0 or len(quotas)!=4 or any(int(q)!=q or q<=0 for q in quotas):
            raise ValueError('Invalid nominal cost/quotas')
        if abs(budget-1.10*nominal_cost)>1e-10:
            raise ValueError('This preregistered control requires fixed B=1.10*p0')
        self.nominal_cost=nominal_cost;self.quotas=tuple(quotas)
        self.bounds=(nominal_cost,1.02*nominal_cost,1.04*nominal_cost,budget)
        self.policy_role='cost_stratified_spatial_greedy'
        super().__init__(initial,'competition',budget,capacity=sum(quotas))

    def stratum(self,r):
        return next((i for i,b in enumerate(self.bounds) if r.cost<=b),3)

    def _farthest_fill(self,selected,remaining,target):
        remaining=list(remaining)
        if not selected and remaining:
            first=min(remaining,key=self._cost_order);selected.append(first)
            remaining=[r for r in remaining if r.route_id!=first.route_id]
        nearest={r.route_id:min(self._distance(r,s) for s in selected) for r in remaining}
        while remaining and len(selected)<target:
            chosen=min(remaining,key=lambda r:(-nearest[r.route_id],self._raw_cost_order(r)))
            selected.append(chosen);remaining=[r for r in remaining if r.route_id!=chosen.route_id]
            for r in remaining:nearest[r.route_id]=min(nearest[r.route_id],self._distance(r,chosen))
        return selected

    def _competition(self,routes):
        selected=[]
        for i,quota in enumerate(self.quotas):
            bucket=[r for r in routes if self.stratum(r)==i]
            selected.extend(self._farthest_fill([],bucket,quota))
        chosen={r.route_id for r in selected}
        remaining=[r for r in routes if r.route_id not in chosen]
        selected=self._farthest_fill(selected,remaining,self.capacity)
        return selected,dict(stratum_counts=[sum(self.stratum(r)==i for r in selected) for i in range(4)])


class CostStratifiedCandidateRun(FixedBudgetCandidateRun):
    def __init__(self,core,descriptor,bank,budget,seconds,nominal_cost):
        # Preserve the exact frozen M dispatch and original parent representation.
        super().__init__('SpatialGreedy_M',core,descriptor,bank,budget,seconds)
        self.policy=CostStratifiedRetention(self.policy.records,nominal_cost,budget)
