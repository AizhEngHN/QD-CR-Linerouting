"""Geometry/cost plumbing for the common exact final selector.

Diagnostic caches are downstream of completed search and unavailable to any
generator. Every pool deduplicates only representations actually in that pool.
"""
from __future__ import annotations
import itertools,time
import numpy as np
from corridor_model import canonical_path
from corridor_spatial_geometry import clip_endpoint_disks,geometric_difference,mean_nearest_distance
from exact_spatial_portfolio import select_exact_portfolio
from spatial_corridor_selection import effective_count


class SharedPhysicalGeometry:
    def __init__(self,core):
        self.core=core;self.g=core.corridor_map
        self.endpoints=[self.g.to_physical(x) for x in (self.g.start,self.g.goal)]
        self.keys={};self.curves=[];self.distances={};self.means={};self.geometry_seconds=0.
    def register(self,path):
        key=canonical_path(path)
        if key not in self.keys:
            self.keys[key]=len(self.curves)
            self.curves.append(clip_endpoint_disks([self.g.to_physical(p) for p in key],self.endpoints,1.))
        return self.keys[key]
    def distance(self,a,b):
        if a==b:return 0.
        pair=tuple(sorted((a,b)))
        if pair not in self.distances:
            start=time.perf_counter();self.distances[pair]=geometric_difference(self.curves[a],self.curves[b],.5)
            self.geometry_seconds+=time.perf_counter()-start
        return self.distances[pair]
    def mean_distance(self,a,b):
        if a==b:return (0.,0.)
        pair=tuple(sorted((a,b)))
        if pair not in self.means:self.means[pair]=mean_nearest_distance(self.curves[a],self.curves[b],.05)
        return self.means[pair]


class ExactFinalWorkspace:
    def __init__(self,records,geometry,nominal_cost,budget_cap):
        self.records=list(records);self.geometry=geometry;self.nominal=nominal_cost;self.budget=budget_cap
        self.byid={r['route_id']:i for i,r in enumerate(records)}
        if len(self.byid)!=len(records):raise ValueError('Duplicate route IDs')
        self.costs=np.array([r['fitness'] for r in records])
        self.keys=[(canonical_path(r['path']),r['route_id']) for r in records]
        self.gids={};self.cache={}
    def gid(self,i):
        if i not in self.gids:self.gids[i]=self.geometry.register(self.records[i]['path'])
        return self.gids[i]
    def distance(self,a,b):return self.geometry.distance(self.gid(a),self.gid(b))
    def dedup(self,ids):
        best={}
        for rid in ids:
            i=self.byid[rid]
            if self.costs[i]>self.budget:continue
            key=self.keys[i][0]
            if key not in best or (self.costs[i],self.keys[i])<(self.costs[best[key]],self.keys[best[key]]):best[key]=i
        return sorted(best.values(),key=lambda i:(self.costs[i],self.keys[i]))
    def select(self,ids,average_cap):
        pool=self.dedup(ids);key=(tuple(pool),average_cap)
        if key not in self.cache:
            self.cache[key]=select_exact_portfolio(pool,self.costs,self.distance,self.keys,self.nominal,average_cap)
        result=self.cache[key];chosen=result['selected'];n=len(chosen);complete=n==5
        means=[self.geometry.mean_distance(self.gid(a),self.gid(b)) for a,b in itertools.combinations(chosen,2)]
        effective=effective_count(chosen,self.distance,.2)
        return dict(pool_ids=[self.records[i]['route_id'] for i in pool],selected_ids=[self.records[i]['route_id'] for i in chosen],
            selected_records=[self.records[i] for i in chosen],delivered_count=n,complete_five=complete,
            effective_count=effective,effective_five=effective==5,pool_qualified_unique=len(pool),
            D_min=result.get('score',[None,None])[0] if complete else None,
            D_mean=result.get('score',[None,None])[1] if complete else None,
            mean_cost_ratio=float(self.costs[chosen].mean()/self.nominal) if chosen else None,
            maximum_cost_ratio=float(self.costs[chosen].max()/self.nominal) if chosen else None,
            best_pool_cost_ratio=float(self.costs[pool].min()/self.nominal) if pool else None,
            mean_nearest_km=float(np.mean([x[0] for x in means])) if means else None,
            mean_nearest_error_bound_km=float(np.mean([x[1] for x in means])) if means else None,
            certificate=result,nominal_cost=self.nominal,budget_cap=self.budget)


def prefix_state(data,attempts,initial_ids):
    """Recover the actual archive checkpoint, never replay future candidates."""
    ledger=data['ledger'];assert 0<=attempts<=len(ledger)
    assert all(not r['late_excluded'] for r in ledger[:attempts])
    ids=list(initial_ids)
    for event in data['retention_events']:
        if event['slot']<attempts:ids=event['retained_ids']
    history=list(initial_ids)+[r['route_id'] for r in ledger[:attempts] if r['valid'] and not r['late_excluded']]
    actual={r['route_id'] for r in data['records']}
    assert set(ids)<=set(history)<=actual
    return ids,history
