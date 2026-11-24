"""GIS transport/graph interfaces around unchanged route mechanisms."""
from __future__ import annotations
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

CODE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(CODE/'src'),str(CODE/'experiments')]
from corridor_model import CorridorMap,CorridorEvaluator,CorridorWeights,_segment_parts
from paper_route_baselines import TurnGraph
from corridor_sensitivity_replay import CachedQualityMatchedRetention

GIS=CODE/'experiments/results/gis_cases_20261102'


def underfilled_policy(records,capacity=20,**kwargs):
    records=list(records)
    if not 0<len(records)<=capacity:raise ValueError('Initial unique count must be between 1 and capacity')
    # The frozen constructor requires an exactly full initial population.
    # Initialize its legal state with the actual count, then expose the target
    # capacity before any offer. Below capacity its unchanged wrapper keeps
    # every distinct geometry; no duplicates or artificial routes are inserted.
    policy=CachedQualityMatchedRetention(records,capacity=len(records),**kwargs)
    policy.capacity=capacity
    return policy


def load_core(case_id,resolution=100):
    with np.load(GIS/case_id/f'model_input_{resolution}m.npz') as d:
        return CorridorEvaluator(CorridorMap(d['forbidden'],d['resistance'],d['start'],d['goal'],
            resolution_m=float(d['resolution_m']),origin_km=d['origin_km'],domain_km=d['domain_km']),CorridorWeights(1.,.5))


def legacy_for(core,descriptor=False):
    g=core.corridor_map
    endpoints=(g.start,g.goal) if descriptor else tuple(tuple(int(round(v)) for v in p) for p in (g.start,g.goal))
    if not descriptor:
        assert max(abs(v-round(v)) for p in (g.start,g.goal) for v in p)<=1e-8
    return SimpleNamespace(grid=np.where(g.forbidden,-1.,g.resistance),grid_width=g.width,grid_height=g.height,
        substation_a=endpoints[0],substation_b=endpoints[1],
        config=SimpleNamespace(resolution_m=g.resolution_m,width_km=g.width*g.resolution_km,height_km=g.height*g.resolution_km),_initial_route_pool={})


def encode_integer(path):
    encoded=[]
    for p in path:
        q=tuple(int(round(v)) for v in p)
        if max(abs(a-b) for a,b in zip(p,q))>1e-8:
            raise ValueError('Nonintegral common-graph path cannot enter frozen integer M')
        encoded.append(q)
    return encoded


def restore_endpoints(path,core):
    points=[list(p) for p in path]
    for i,expected in ((0,core.corridor_map.start),(-1,core.corridor_map.goal)):
        if max(abs(a-b) for a,b in zip(points[i],expected))>1e-8:
            raise ValueError('Proposal endpoints changed beyond the coding tolerance')
        points[i]=list(expected)
    return points


def densify_integer(path,core,max_span_km=2.):
    encoded=encode_integer(path);out=[encoded[0]]
    limit=max_span_km/core.corridor_map.resolution_km
    for a,b in zip(encoded,encoded[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1];gcd=math.gcd(abs(dx),abs(dy))
        if not gcd:raise ValueError('Repeated point')
        need=max(1,math.ceil(math.dist(a,b)/limit-1e-12))
        divisors=[n for n in range(need,gcd+1) if gcd%n==0]
        if not divisors:raise ValueError('No collinear integer subdivision satisfies proposal span')
        n=divisors[0]
        out.extend([(a[0]+dx*j//n,a[1]+dy*j//n) for j in range(1,n+1)])
    if len(out)>300:raise ValueError('Frozen proposal representation exceeds 300 points')
    return restore_endpoints(out,core)


class GISTurnGraph(TurnGraph):
    """Same turn graph with explicit physical spacing and terminal attachment."""
    def __init__(self,core,spacing_km=1.,**kwargs):
        self.physical_spacing_km=float(spacing_km)
        super().__init__(core,spacing=spacing_km/core.corridor_map.resolution_km,**kwargs)
        g=core.corridor_map;terminal_ids={self.start,self.goal};added=[]
        lattice=[i for i in range(len(self.points)) if i not in terminal_ids]
        for i in (self.start,self.goal):
            nearest=sorted(lattice,key=lambda j:(math.dist(self.points[i],self.points[j]),j))[:8]
            for j in nearest:
                if any(k==j for k,_,_ in self.adj[i]):continue
                a,b=self.points[i],self.points[j]
                if self.corridor_path is not None and not all(self.in_corridor(p) for p in np.linspace(a,b,9)):continue
                touched,intervals=_segment_parts(a,b,g.shape)
                if any(g.forbidden[y,x] for x,y in touched):continue
                length=math.dist(a,b)*g.resolution_km
                exposure=length*math.fsum(f*math.fsum(float(g.resistance[y,x]) for x,y in cells)/len(cells) for f,cells in intervals)
                self.adj[i].append((j,length,exposure));self.adj[j].append((i,length,exposure));added.append((i,j))
        for edges in self.adj:edges.sort()
        self.terminal_attachment_edges=added


def transformed_path(path,source,target):
    a,b=source.corridor_map,target.corridor_map
    result=[[(v-b.origin_km[i])/b.resolution_km for i,v in enumerate(a.to_physical(p))] for p in path]
    # Same virtual physical terminals, within floating representation residual.
    result[0]=list(b.start);result[-1]=list(b.goal)
    return result
