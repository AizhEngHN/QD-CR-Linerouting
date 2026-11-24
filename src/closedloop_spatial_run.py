"""Online fixed-budget wrapper around the frozen candidate generator.

CandidateRun.generate is inherited unchanged: same parent draw, integer
encoding, 0.2 corridor probability, and long-path fallback. Only the previously
agreed fixed-B eligibility/retention interface is supplied here. Logs are never
available to parent selection; pair caches retain current-archive entries only.
"""
from __future__ import annotations
import copy,time,math
import numpy as np
from replay_retention import RetentionPolicy,RouteRecord
from spatial_corridor_selection import SpatialBudgetRetention
from run_gis_algorithm_comparison import CandidateRun
from gis_comparison_adapter import restore_endpoints


class OnlineFixedBudgetRetention(SpatialBudgetRetention):
    def __init__(self,initial,selector,budget,capacity=20):
        super().__init__(initial,selector,budget,{},None,capacity)
    def _distance(self,a,b):
        return RetentionPolicy._distance(self,a,b)
    def offer(self,candidate):
        result=super().offer(candidate)
        self._prune_caches()
        return result


class FixedBudgetCandidateRun(CandidateRun):
    def __init__(self,method,core,descriptor,bank,budget,seconds):
        self.method,self.core,self.descriptor=method,core,descriptor
        self.started=time.perf_counter();self.deadline=self.started+seconds
        self.records={r['route_id']:copy.deepcopy(r) for r in bank}
        carriers=[RouteRecord(r['route_id'],r['path'],r['fitness'],descriptor.compute(r['path'])) for r in bank]
        self.policy=OnlineFixedBudgetRetention(carriers,'greedy' if method=='SpatialGreedy_M' else 'competition',budget,20)
        self.ledger=[];self.offers=[];self.model_calls=0;self.budget=budget

    def offer(self,path,operator):
        slot=len(self.ledger);rid=f'{self.method}:{slot}'
        row=dict(slot=slot,route_id=rid,path=None,operator=operator,valid=False,fitness=None,
                 reason='construction_failed',late_excluded=False,within_B=False)
        self.ledger.append(row)
        if path is None:
            row.update(late_excluded=time.perf_counter()>self.deadline,finished_seconds=time.perf_counter()-self.started)
            return None
        path=restore_endpoints(path,self.core);cost,details=self.core.evaluate(path);self.model_calls+=1
        row.update(path=path,valid=bool(details['valid']),fitness=cost if math.isfinite(cost) else None,
                   reason=details['failure_reason'],within_B=bool(details['valid'] and cost<=self.budget))
        if not details['valid'] or time.perf_counter()>self.deadline:
            row.update(late_excluded=time.perf_counter()>self.deadline,finished_seconds=time.perf_counter()-self.started)
            return None
        rec=dict(route_id=rid,path=path,fitness=cost,cost_details=details,cost_unit=self.core.cost_unit,model_sha256=self.core.model_sha256)
        before=self.policy._records.copy();before_stats=self.policy.stats.copy()
        if cost<=self.budget:
            vector=self.descriptor.compute(path)
            if time.perf_counter()>self.deadline:
                row.update(late_excluded=True,finished_seconds=time.perf_counter()-self.started);return None
            carrier=RouteRecord(rid,path,cost,vector);event=self.policy.offer(carrier)
        else:
            event=dict(route_id=rid,accepted=False,reason='outside_fixed_budget',deleted_ids=[],retained_ids=[r.route_id for r in before])
        if time.perf_counter()>self.deadline:
            self.policy._records=before;self.policy.stats=before_stats;self.policy._prune_caches()
            row.update(late_excluded=True,finished_seconds=time.perf_counter()-self.started);return None
        self.records[rid]=rec
        self.offers.append(dict(slot=slot,**event))
        row['finished_seconds']=time.perf_counter()-self.started
        if row['finished_seconds']>self.deadline-self.started:
            self.records.pop(rid);self.offers.pop();self.policy._records=before
            self.policy.stats=before_stats;self.policy._prune_caches();row['late_excluded']=True
            return None
        return rec
