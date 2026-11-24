"""Candidate generator shared by the frozen online search classes."""
import copy
import random
import time

import numpy as np

from corridor_model import canonical_path
from corridor_mutation import CorridorMutation
from corridor_sensitivity_replay import SharedDistanceCache
from gis_comparison_adapter import encode_integer, restore_endpoints, underfilled_policy
from paper_route_baselines import guided_aco, multiweight_routes, soft_penalty_routes
from qd_individual import Individual
from qd_mutation import MutationOperators
from replay_retention import RouteRecord


class CandidateRun:
    def __init__(self,method,core,descriptor,bank,delta,seconds):
        self.method,self.core,self.descriptor=method,core,descriptor
        self.started=time.perf_counter();self.deadline=self.started+seconds
        self.shared=SharedDistanceCache();self.records={r['route_id']:copy.deepcopy(r) for r in bank}
        self.vectors={rid:descriptor.compute(r['path']) for rid,r in self.records.items()}
        self.carriers={rid:RouteRecord(rid,tuple(map(tuple,r['path'])),r['fitness'],self.vectors[rid]) for rid,r in self.records.items()}
        self.policy=underfilled_policy(self.carriers.values(),selector='competition' if method=='QD_DAM' else 'greedy',capacity=20,online_tolerance=delta,shared_cache=self.shared)
        self.ledger=[];self.offers=[];self.model_calls=0

    def offer(self,path,operator):
        rid=f'{self.method}:{len(self.ledger)}'
        if path is None:
            self.ledger.append(dict(route_id=rid,path=None,operator=operator,valid=False,fitness=None,reason='construction_failed',late_excluded=time.perf_counter()>self.deadline));return None
        path=restore_endpoints(path,self.core)
        cost,details=self.core.evaluate(path);self.model_calls+=1
        row=dict(route_id=rid,path=path,operator=operator,valid=details['valid'],fitness=cost if details['valid'] else None,reason=details['failure_reason'],late_excluded=False)
        self.ledger.append(row)
        if not details['valid']:
            row['late_excluded']=time.perf_counter()>self.deadline;return None
        if time.perf_counter()>self.deadline:
            row['late_excluded']=True;return None
        vector=self.descriptor.compute(path)
        if time.perf_counter()>self.deadline:
            row['late_excluded']=True;return None
        rec=dict(route_id=rid,path=path,fitness=cost,cost_details=details,cost_unit=self.core.cost_unit,model_sha256=self.core.model_sha256)
        carrier=RouteRecord(rid,tuple(map(tuple,path)),cost,vector)
        before=self.policy._records.copy();before_stats=copy.deepcopy(self.policy.stats)
        event=self.policy.offer(carrier)
        if time.perf_counter()>self.deadline:
            row['late_excluded']=True;self.policy._records=before;self.policy.stats=before_stats;return None
        self.records[rid]=rec;self.vectors[rid]=vector;self.carriers[rid]=carrier
        self.offers.append(dict(route_id=rid,accepted=event['accepted'],deleted_ids=event['deleted_ids'],retained_ids=event['retained_ids']))
        return rec

    def generate(self,legacy,graph,seed):
        random.seed(seed);np.random.seed(seed);metadata={}
        if self.method in ('QD_DAM','SpatialGreedy_M'):
            local=MutationOperators(legacy,max_span_km=2.)
            corridor=CorridorMutation(legacy,max_span_km=2.,deadline=self.deadline)
            while time.perf_counter()<self.deadline:
                parent=random.choice(self.policy.records)
                ind=Individual(encode_integer(parent.path));ind.fitness=parent.cost
                child=corridor.mutate(ind) if random.random()<.2 else local.mutate(ind)
                if len(child.path)>300:child=ind.copy()
                self.offer(child.path,'frozen_M')
            metadata=dict(corridor_stats=corridor.stats,operator_radius_cells=3,operator_radius_km=3*self.core.corridor_map.resolution_km)
        elif self.method=='Wang2024_ACO':metadata=guided_aco(graph,self.offer,seed,self.deadline,10**9)
        elif self.method=='Ramirez2025_common':metadata=multiweight_routes(graph,self.offer,self.deadline)
        elif self.method.startswith('LION_soft_penalty'):metadata=soft_penalty_routes(graph,self.offer,self.deadline,10**9)
        elif self.method!='InitOnly':raise ValueError(self.method)
        elapsed=time.perf_counter()-self.started
        return dict(method=self.method,attempt_slots=len(self.ledger),actual_complete_curve_evaluations=self.model_calls,
            valid_candidates=sum(r['valid'] for r in self.ledger),late_excluded=sum(r['late_excluded'] for r in self.ledger),
            timely_valid_candidates=sum(r['valid'] and not r['late_excluded'] for r in self.ledger),
            unique_generated_geometries=len({canonical_path(r['path']) for r in self.ledger if r['valid'] and not r['late_excluded']}),
            search_elapsed_seconds=elapsed,search_limit_seconds=self.deadline-self.started,
            deadline_exceeded=elapsed>self.deadline-self.started,archive_count=len(self.policy.records),metadata=metadata)
