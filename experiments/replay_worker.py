"""Fixed-candidate-stream replay extracted from the paper analysis."""
import itertools
import math
import time

from closedloop_spatial_run import OnlineFixedBudgetRetention
from cost_first_bounded_solver import select_bounded_exact
from exact_final_workspace import SharedPhysicalGeometry
from paper_inputs import descriptor_for
from replay_retention import RouteRecord



def replay(task, rows, kind, load_map):
 core,_=load_map(task['case'],task['profile']);d=descriptor_for(core,kind);sel='greedy' if kind=='SpatialGreedy' else 'topk' if kind=='TopCost' else 'competition';start=time.perf_counter();first=rows[0];policy=OnlineFixedBudgetRetention([RouteRecord(first['route_id'],first['path'],first['fitness'],d.compute(first['path']))],sel,math.inf,20)
 for x in rows[1:]:policy.offer(RouteRecord(x['route_id'],x['path'],x['fitness'],d.compute(x['path'])))
 ids={x.route_id for x in policy.records};pool=[x for x in rows if x['route_id'] in ids];geom=SharedPhysicalGeometry(core);out=select_bounded_exact(pool,geom,deadline=time.perf_counter()+60,tau=.2);chosen=out.pop('selected');dist=[geom.distance(geom.register(a['path']),geom.register(b['path'])) for a,b in itertools.combinations(chosen,2)]
 return dict(kind=kind,pool=len(pool),count=len(chosen),J=sum(x['fitness'] for x in chosen)/len(chosen) if chosen else None,Dmin=min(dist) if dist else None,selected=[x['route_id'] for x in chosen],certified=out.get('certified'),status=out.get('status'),elapsed=time.perf_counter()-start)
