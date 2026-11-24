"""Paper search worker adapted to accept a local map loader and configuration."""
import math
import random
import time
import json

import numpy as np
from scipy.spatial import cKDTree

from closedloop_spatial_run import FixedBudgetCandidateRun
from corridor_model import canonical_path
from corridor_mutation import CorridorMutation
from cost_first_bounded_solver import select_bounded_exact
from exact_final_workspace import SharedPhysicalGeometry
from gis_comparison_adapter import densify_integer, encode_integer
from gis_model_validation_inputs import ModelParameterGraph, weighted_legacy
from no_reference_search import NoReferenceRun
from paper_route_baselines import SearchExpired, guided_aco, multiweight_routes, soft_penalty_routes
from qd_individual import Individual
from qd_mutation import MutationOperators

from paper_inputs import descriptor_for

BUDGETS = (50, 100, 200)
SAFETY = 900.
SELECT_SECONDS = 60.


class BudgetReached(Exception):
    pass



def worker(task, started, load_map):
    configuration = task['configuration']
    end=started+SAFETY;seed=task['seed'];arm=task['arm'];rng=np.random.default_rng(seed);random.seed(seed+1000)
    core,_=load_map(task['case'],task['profile']);desc=descriptor_for(core, task['label']);legacy=weighted_legacy(core)
    graph=ModelParameterGraph(core,spacing_km=1.,deadline=end);t_init=time.perf_counter(); records=[];ledger=[];actor=None;count=0;checkpoints=[];select_total=0.;reason='natural_stop';metadata={}
    def pool_now():
        return [actor.records[r.route_id] for r in actor.policy.records] if actor is not None else records.copy()
    def snapshot(target,reached):
        nonlocal select_total
        pool=pool_now();geom=SharedPhysicalGeometry(core);t=time.perf_counter()
        sel=select_bounded_exact(pool,geom,deadline=t+SELECT_SECONDS,tau=.2);selected=sel.pop('selected')
        ds=[geom.distance(geom.register(a['path']),geom.register(b['path'])) for i,a in enumerate(selected) for b in selected[i+1:]]
        final=time.perf_counter();dur=final-t;select_total+=dur
        checkpoints.append(dict(target=target,reached=reached,evaluations=count,attempts=len(ledger) if actor is None else len(actor.ledger)+1,
          graph_expansions=graph.expansions,pool_size=len(pool),unique_pool=len({canonical_path(x['path']) for x in pool}),
          delivered=len(selected),five=len(selected)==5,J_mean=float(np.mean([x['fitness'] for x in selected])) if selected else None,
          J_best=min((x['fitness'] for x in selected),default=None),D_min=min(ds) if ds else None,
          T_init=t_init-started,T_generation=t-started-select_total+dur,T_select=dur,T_previous_select=select_total-dur,T_total=final-started,
          selected=selected,selection=sel))
    def after_evaluation():
        if count in BUDGETS:snapshot(count,True)
        if count>=200:raise BudgetReached()
    def offer(path,operator):
        nonlocal count
        idx=len(ledger);row=dict(id=f'{arm}:{idx}',operator=operator,valid=False,evaluated=False)
        ledger.append(row)
        if path is None:return None
        value,detail=core.evaluate(path);count+=1;row.update(evaluated=True,valid=bool(detail['valid']),fitness=float(value) if detail['valid'] else None,path=path)
        rec=None
        if detail['valid']:
            rec=dict(route_id=row['id'],path=path,fitness=float(value));records.append(rec)
        after_evaluation();return rec
    try:
      if arm in ('QD_Hybrid','SpatialGreedy'):
        path,_=graph.shortest(deadline=end)
        bank=[]
        if path is not None:
            path=densify_integer(path,core);val,det=core.evaluate(path);count+=1
            if det['valid']:bank=[dict(route_id='bank:0',path=path,fitness=float(val),cost_details=det,cost_unit=core.cost_unit,model_sha256=core.model_sha256)]
        if not bank:reason='initialization_failed'
        else:
            actor=(FixedBudgetCandidateRun('SpatialGreedy_M',core,desc,bank,math.inf,SAFETY) if arm=='SpatialGreedy' else NoReferenceRun('QD',core,desc,bank,SAFETY));actor.deadline=end
            actor.policy.capacity = configuration['capacity']
            if task['label'] == 'TopCost':
                actor.policy.selector = 'topk'
            local=MutationOperators(legacy,max_span_km=2.);corridor=CorridorMutation(legacy,max_span_km=2.,deadline=end);attempt=0;global_count=0
            while time.perf_counter()<end and attempt<1000:
                if attempt%4!=3:
                    centers=rng.uniform(graph.xy.min(axis=0),graph.xy.max(axis=0),size=(3,2));penalty=np.exp(-((graph.xy[:,None,:]-centers[None,:,:])**2).sum(axis=2)/(2 * configuration["sigma"] ** 2)).sum(axis=1)
                    if global_count%4==3:
                        anchor=min(actor.policy.records,key=lambda r:r.cost);samples=[]
                        for a,b in zip(anchor.path,anchor.path[1:]):
                            aa=np.asarray(a)*core.corridor_map.resolution_km;bb=np.asarray(b)*core.corridor_map.resolution_km
                            samples.extend(np.linspace(aa,bb,max(2,math.ceil(np.linalg.norm(bb-aa)/.1)+1)))
                        penalty+=np.maximum(0.,1-cKDTree(samples).query(graph.xy)[0]/.6)
                    path,_=graph.shortest(penalty=penalty,deadline=end);global_count+=1
                    if path is not None:
                        try:path=densify_integer(path,core)
                        except ValueError:path=None
                    before=actor.model_calls;actor.offer(path,'random_graph')
                else:
                    parent=random.choice(actor.policy.records);ind=Individual(encode_integer(parent.path));ind.fitness=parent.cost
                    child=corridor.mutate(ind) if random.random() < configuration["prob"] else local.mutate(ind)
                    if len(child.path)>300:child=ind.copy()
                    before=actor.model_calls;actor.offer(child.path,'local_mutation')
                attempt+=1
                if actor.model_calls>before:count+=1;after_evaluation()
            if attempt>=1000:reason='safety_attempt_limit'
      elif arm=='LION2021':metadata=soft_penalty_routes(graph,offer,end,2000,.6)
      elif arm=='Wang2024':metadata=guided_aco(graph,offer,seed,end,2000)
      elif arm=='Ramirez2025_generator':metadata=multiweight_routes(graph,offer,end)
      else:
        attempts=0
        while attempts<2000 and time.perf_counter()<end:
            if attempts==0:penalty=None
            else:
                centers=rng.uniform(graph.xy.min(axis=0),graph.xy.max(axis=0),size=(3,2))
                penalty=np.exp(-((graph.xy[:,None,:]-centers[None,:,:])**2).sum(axis=2)/(2 * configuration["sigma"] ** 2)).sum(axis=1)
            path,_=graph.shortest(penalty=penalty,deadline=end);offer(path,'unperturbed' if attempts==0 else 'random_gaussian_search_cost');attempts+=1
        metadata=dict(attempts=attempts)
    except BudgetReached:reason='evaluation_budget_200'
    except SearchExpired:reason='safety_time_limit'
    if time.perf_counter()>=end and reason=='natural_stop':reason='safety_time_limit'
    elif metadata.get('attempts',0)>=2000 and reason=='natural_stop':reason='safety_attempt_limit'
    missing=[b for b in BUDGETS if not any(x['target']==b for x in checkpoints)]
    if missing:
        # Select once at natural stop and copy the identical result for unreached targets.
        snapshot(missing[0],False)
        for b in missing[1:]:checkpoints.append(dict(checkpoints[-1],target=b))
    metadata.update(stop_reason=reason,final_evaluations=count,wall_seconds=time.perf_counter()-started)
    return dict(task=task,stop=metadata,checkpoints=checkpoints),dict(task=task,ledger=actor.ledger if actor else ledger,
      records=list(actor.records.values()) if actor else records,checkpoints=checkpoints)
