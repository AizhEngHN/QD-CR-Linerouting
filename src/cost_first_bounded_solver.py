"""Cost-pruned exact anchored selector with deterministic feasible fallback."""
import math,time
from corridor_model import canonical_path
from no_reference_selection import select_cost_first

def select_bounded_exact(records,geometry,deadline=math.inf,target=5,tau=.2):
    started=time.perf_counter()
    fallback=select_cost_first(records,geometry,deadline=deadline,target=target,tau=tau)
    incumbent=fallback['selected']
    unique={}
    for r in records:
        k=canonical_path(r['path'])
        if k not in unique or (r['fitness'],r['route_id'])<(unique[k]['fitness'],unique[k]['route_id']):unique[k]=r
    ordered=sorted(unique.values(),key=lambda r:(r['fitness'],canonical_path(r['path']),r['route_id']))
    valid=[];gids=[];invalid=[]
    for r in ordered:
        if time.perf_counter()>=deadline:return dict(selected=incumbent,certified=False,status='timeout',invalid=invalid,seconds=time.perf_counter()-started)
        try:gid=geometry.register(r['path'])
        except ValueError:invalid.append(r['route_id']);continue
        valid.append(r);gids.append(gid)
    if not valid:return dict(selected=[],certified=True,status='empty',invalid=invalid,seconds=time.perf_counter()-started,n=0)
    costs=[float(r['fitness']) for r in valid];n=len(valid);best=None;nodes=0
    rid_to_i={r['route_id']:i for i,r in enumerate(valid)}
    cache={}
    def compatible(a,b):
        if time.perf_counter()>=deadline:raise TimeoutError
        key=(a,b) if a<b else (b,a)
        if key not in cache:cache[key]=geometry.distance(gids[key[0]],gids[key[1]])>=tau
        if time.perf_counter()>=deadline:raise TimeoutError
        return cache[key]
    try:
        for k in range(min(target,n),0,-1):
            old=[rid_to_i[r['route_id']] for r in incumbent if r['route_id'] in rid_to_i]
            if len(old)==k and old and old[0]==0:
                best=(math.fsum(costs[j] for j in old),tuple(sorted(old)))
            else:best=None
            def dfs(chosen,start,total):
                nonlocal best,nodes
                nodes+=1
                if time.perf_counter()>=deadline:raise TimeoutError
                need=k-len(chosen)
                if need==0:
                    key=(total,tuple(chosen))
                    if best is None or key<best:best=key
                    return
                if n-start<need:return
                if best is not None and total+math.fsum(costs[start:start+need])>best[0]+1e-10:return
                for i in range(start,n-need+1):
                    if best is not None and total+math.fsum(costs[i:i+need])>best[0]+1e-10:break
                    if all(compatible(j,i) for j in chosen):dfs(chosen+[i],i+1,total+costs[i])
            dfs([0],1,costs[0])
            if best is not None:
                selected=[valid[i] for i in best[1]]
                return dict(selected=selected,certified=True,status='complete',invalid=invalid,seconds=time.perf_counter()-started,
                   n=n,nodes=nodes,geometry_pairs=len(cache),objective=best[0])
    except TimeoutError:
        if best is not None:
            candidate=[valid[i] for i in best[1]]
            if len(candidate)>len(incumbent) or (len(candidate)==len(incumbent) and math.fsum(r['fitness'] for r in candidate)<math.fsum(r['fitness'] for r in incumbent)):
                incumbent=candidate
        return dict(selected=incumbent,certified=False,status='timeout',invalid=invalid,seconds=time.perf_counter()-started,n=n,nodes=nodes,geometry_pairs=len(cache))
    raise AssertionError('single anchor must be feasible')
