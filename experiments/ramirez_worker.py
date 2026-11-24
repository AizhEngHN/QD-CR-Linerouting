"""Ramírez baseline adaptation used in the paper."""
import json
import math
import time

import numpy as np

from corridor_model import _segment_parts
from cost_first_bounded_solver import select_bounded_exact
from exact_final_workspace import SharedPhysicalGeometry
from gis_model_validation_inputs import ModelParameterGraph
from paper_route_baselines import SearchExpired, TurnGraph, VGGRouteCluster, multiweight_routes, refine_rrtstar

from paper_inputs import write_json



def attach_isolated_terminals(graph,deadline):
 for endpoint in (graph.start,graph.goal):
  if graph.adj[endpoint]:continue
  a=graph.points[endpoint]
  nearest=sorted((math.dist(a,b),j) for j,b in enumerate(graph.points) if j!=endpoint)
  for dist,j in nearest:
   if time.perf_counter()>=deadline:raise SearchExpired()
   if dist>4*2.5:break
   b=graph.points[j]
   if not all(graph.in_corridor(p) for p in np.linspace(a,b,9)):continue
   touched,intervals=_segment_parts(a,b,graph.grid.shape)
   if any(graph.grid.forbidden[y,x] for x,y in touched):continue
   length=dist*graph.grid.resolution_km
   exposure=length*math.fsum(f*math.fsum(float(graph.grid.resistance[y,x]) for x,y in cells)/len(cells) for f,cells in intervals)
   graph.adj[endpoint].append((j,length,exposure));graph.adj[j].append((endpoint,length,exposure))
  graph.adj[endpoint].sort()
 for edges in graph.adj:edges.sort()
 return bool(graph.adj[graph.start] and graph.adj[graph.goal])


def worker(task, started, output, load_map, weights_path):
 mode=task['mode'];end=started+(120 if mode=='time' else 900);gen_end=started+(65 if mode=='time' else 840);core,_=load_map(task['case'],task['profile'])
 graph=ModelParameterGraph(core,spacing_km=1.,deadline=gen_end);records=[];ledger=[];count=0
 def offer(path,operator):
  nonlocal count
  i=len(ledger);row={'id':f'native:{i}','operator':operator,'valid':False}
  ledger.append(row)
  if path is None or count>=200:return None
  val,det=core.evaluate(path);count+=1;row.update(valid=bool(det['valid']),fitness=float(val) if det['valid'] else None,path=path)
  if det['valid']:
   r=dict(route_id=row['id'],path=path,fitness=float(val));records.append(r);return r
  return None
 search_meta=multiweight_routes(graph,offer,gen_end)
 write_json(output.with_name(output.stem+'_partial.json'),dict(task=task,ledger=ledger,records=records,search_meta=search_meta,count=count))
 phase=time.perf_counter();cluster=VGGRouteCluster(weights_path)
 ids,cluster_info=cluster.representatives(records,core.corridor_map.shape,task['seed']);lookup={r['route_id']:r for r in records}
 ref_end=min(time.perf_counter()+25,end-2) if mode=='time' else min(time.perf_counter()+25,end-60);refined=[];details=[]
 for j,rid in enumerate(ids):
  r=lookup[rid];guide=r['path'];fs='not_started';rs='not_started'
  if time.perf_counter()<ref_end:
   try:
    fine=TurnGraph(core,spacing=2.5,deadline=ref_end,corridor_path=guide)
    attached=attach_isolated_terminals(fine,ref_end)
    fs='no_path' if attached else 'terminal_unconnected'
    path,_=fine.shortest(alpha=r.get('generation_alpha',1.),deadline=ref_end)
    if path is not None:guide=path;fs='fine_graph'
   except SearchExpired:fs='deadline'
  if time.perf_counter()<ref_end and count<199:
   smooth=refine_rrtstar(core,guide,task['seed']+j,ref_end)
   value,det=core.evaluate(smooth);count+=1
   if det['valid']:guide=smooth;rs='valid'
   else:rs='invalid'
  if count<200:
   value,det=core.evaluate(guide);count+=1
   if det['valid']:refined.append(dict(route_id='refined:'+rid,path=guide,fitness=float(value)))
  details.append(dict(id=rid,graph=fs,rrt=rs))
 now=time.perf_counter();deadline=min(started+119,now+60) if mode=='time' else now+60;geom=SharedPhysicalGeometry(core)
 sel=select_bounded_exact(refined,geom,deadline=deadline,tau=.2);selected=sel.pop('selected')
 ds=[geom.distance(geom.register(a['path']),geom.register(b['path'])) for i,a in enumerate(selected) for b in selected[i+1:]]
 cp=dict(evaluations=count,generated_evaluations=len([x for x in ledger if x.get('path') is not None]),delivered=len(selected),five=len(selected)==5,J_mean=float(np.mean([x['fitness'] for x in selected])) if selected else None,D_min=min(ds) if ds else None,selected=selected,selection=sel)
 json.dumps(cp,default=lambda x:x.item() if hasattr(x,'item') else x.tolist());finish=time.perf_counter()
 cp.update(T_total=finish-started,on_time=finish<=end)
 return dict(task=task,checkpoint=cp,cluster=cluster_info,refinement=details,search_meta=search_meta,graph_expansions=graph.expansions,records=records,ledger=ledger)
