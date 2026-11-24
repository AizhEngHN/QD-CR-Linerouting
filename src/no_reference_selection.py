"""Common cost-first geometric selector with no cost reference or cap."""
import math,time
from corridor_model import canonical_path

def select_cost_first(records,geometry,deadline=math.inf,target=5,tau=.2):
    """Return timely greedy result; an expired call never invents a route."""
    unique={};invalid_geometry=[]
    for r in records:
        if time.perf_counter()>deadline:return dict(selected=[],status='timeout',invalid_geometry=invalid_geometry)
        key=canonical_path(r['path'])
        if key not in unique or (r['fitness'],r['route_id'])<(unique[key]['fitness'],unique[key]['route_id']):unique[key]=r
    ordered=sorted(unique.values(),key=lambda r:(r['fitness'],canonical_path(r['path']),r['route_id']))
    selected=[];gids=[]
    for r in ordered:
        if time.perf_counter()>deadline:return dict(selected=selected,status='timeout',invalid_geometry=invalid_geometry)
        try:gid=geometry.register(r['path'])
        except ValueError:invalid_geometry.append(r['route_id']);continue
        if all(geometry.distance(gid,old)>=tau for old in gids):selected.append(r);gids.append(gid)
        if len(selected)>=target:break
    if time.perf_counter()>deadline:return dict(selected=selected,status='timeout',invalid_geometry=invalid_geometry)
    return dict(selected=selected,status='complete' if len(selected)==target else 'underfilled',invalid_geometry=invalid_geometry)
