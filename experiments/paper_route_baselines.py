"""Paper-mechanism implementations, explicitly adapted to corridor model v1.

Ramirez2025: weighted shortest paths, pretrained VGG16, k-means/centroid reps.
Wang2024: Eq5 transition, Eq6 ant-cycle, guided/coarse/hard-mask mechanisms.
Wiedemann2021: cumulative soft corridor penalization, turn-aware shortest paths.
Missing-paper implementation choices are in paper_baseline_plan_20261102.txt.
"""
from __future__ import annotations
import heapq,math,time,random
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from corridor_model import _segment_parts,canonical_path


class SearchExpired(RuntimeError):pass

class TurnGraph:
    """Sparse control-point graph. Costs integrate the original fine field.

    Coarse graph is a search restriction, never a map/model replacement.
    (previous,current) labels preserve angular costs. Returned full curves are
    always revalidated by the authoritative evaluator, including self crossing.
    """
    def __init__(self,core,spacing=5,deadline=math.inf,corridor_path=None,corridor_radius_km=.25):
        self.core=core;g=core.corridor_map;self.grid=g
        self.corridor_path=corridor_path;self.corridor_radius_km=corridor_radius_km
        points={(float(x),float(y)) for y in np.arange(0,g.shape[0]-1+1e-9,spacing) for x in np.arange(0,g.shape[1]-1+1e-9,spacing)}
        if corridor_path is not None:points={p for p in points if self.in_corridor(p)}
        points.update([g.start,g.goal]);self.points=sorted(points);lookup={p:i for i,p in enumerate(self.points)}
        self.start=lookup[g.start];self.goal=lookup[g.goal];self.adj=[[] for _ in self.points];self.expansions=0
        offsets=[(dx*spacing,dy*spacing) for dx in range(-2,3) for dy in range(-2,3) if (abs(dx),abs(dy)) in ((0,1),(1,0),(1,1),(1,2),(2,1))]
        for i,a in enumerate(self.points):
            if time.perf_counter()>deadline:raise SearchExpired()
            for dx,dy in offsets:
                b=(a[0]+dx,a[1]+dy);j=lookup.get(b)
                if j is None or j<=i:continue
                if corridor_path is not None and not all(self.in_corridor(p) for p in np.linspace(a,b,9)):continue
                touched,intervals=_segment_parts(a,b,g.shape)
                if any(g.forbidden[y,x] for x,y in touched):continue
                length=math.dist(a,b)*g.resolution_km
                exposure=length*math.fsum(f*math.fsum(float(g.resistance[y,x]) for x,y in cells)/len(cells) for f,cells in intervals)
                self.adj[i].append((j,length,exposure));self.adj[j].append((i,length,exposure))
        for v in self.adj:v.sort()
        self.xy=np.asarray(self.points,float)*g.resolution_km
    def in_corridor(self,p):
        if self.corridor_path is None:return True
        points=np.asarray(self.corridor_path,float);a=points[:-1];v=points[1:]-a
        t=np.clip(np.sum((np.asarray(p)-a)*v,axis=1)/np.sum(v*v,axis=1),0,1)
        return bool(np.min(np.linalg.norm(a+t[:,None]*v-p,axis=1))*self.grid.resolution_km<=self.corridor_radius_km+1e-12)
    def turn(self,previous,current,nxt):
        if previous<0:return 0.
        a=self.xy[current]-self.xy[previous];b=self.xy[nxt]-self.xy[current]
        return math.atan2(abs(float(a[0]*b[1]-a[1]*b[0])),float(a@b))/(math.pi/2)
    def shortest(self,alpha=1.,kappa=.5,penalty=None,deadline=math.inf):
        if alpha<0 or kappa<0:raise ValueError('nonnegative weights required')
        penalty=np.zeros(len(self.points)) if penalty is None else np.asarray(penalty,float)
        if penalty.shape!=(len(self.points),) or not np.isfinite(penalty).all() or (penalty<0).any():raise ValueError('invalid penalty')
        start=(-1,self.start);dist={start:0.};parent={};heap=[(0.,0.,start)];found=None
        while heap:
            if self.expansions%128==0 and time.perf_counter()>deadline:raise SearchExpired()
            _,cost,state=heapq.heappop(heap)
            if cost!=dist[state]:continue
            self.expansions+=1;prev,i=state
            if i==self.goal:found=state;break
            for j,length,exposure in self.adj[i]:
                if j==prev:continue
                val=cost+length+alpha*exposure+kappa*self.turn(prev,i,j)+length*(penalty[i]+penalty[j])*.5
                nxt=(i,j)
                if val<dist.get(nxt,math.inf):
                    dist[nxt]=val;parent[nxt]=state
                    h=math.dist(self.xy[j],self.xy[self.goal]);heapq.heappush(heap,(val+h,val,nxt))
        if found is None:return None,None
        path=[];node=found
        while node!=start:path.append(self.points[node[1]]);node=parent[node]
        path.append(self.points[self.start]);path.reverse()
        return path,dist[found]


def guided_aco(graph,offer,seed,deadline,max_attempts=125,ants=20,evaporation=.3):
    """Wang Eq5/6 plus declared quantitative completion of prose mechanisms.

    eta=1/edge length, alpha=1,beta=3; positive direction multiplier exp(2cos).
    Guidance reset to current->goal if its preferred half-plane is empty.
    Forward/revisit handling and parameters are disclosed completions, not
    claimed author defaults. Each attempted ant (also failed) uses one slot.
    """
    rng=np.random.default_rng(seed);tau={(i,j):1. for i,v in enumerate(graph.adj) for j,_,_ in v};attempts=0;resets=0
    while attempts<max_attempts and time.perf_counter()<deadline:
        completed=[]
        for _ in range(min(ants,max_attempts-attempts)):
            if time.perf_counter()>=deadline:break
            ids=[graph.start];seen=set(ids);guide=graph.xy[graph.goal]-graph.xy[graph.start]
            while ids[-1]!=graph.goal:
                if time.perf_counter()>=deadline:break
                i=ids[-1];options=[(j,l,e) for j,l,e in graph.adj[i] if j not in seen]
                if not options:break
                def cosine(j):
                    d=graph.xy[j]-graph.xy[i];return float(d@guide)/(np.linalg.norm(d)*np.linalg.norm(guide))
                preferred=[o for o in options if cosine(o[0])>=0]
                if not preferred:guide=graph.xy[graph.goal]-graph.xy[i];resets+=1
                weights=np.array([tau[i,j]*(1/l)**3*math.exp(2*cosine(j)) for j,l,e in options]);weights/=weights.sum()
                j=options[int(rng.choice(len(options),p=weights))][0];ids.append(j);seen.add(j)
            path=[graph.points[i] for i in ids] if ids[-1]==graph.goal else None
            rec=offer(path,'aco_ant');attempts+=1
            if rec is not None:completed.append((ids,rec['fitness']))
        for edge in tau:tau[edge]=max(1e-12,tau[edge]*(1-evaporation))
        for ids,cost in completed:
            for i,j in zip(ids,ids[1:]):tau[i,j]+=1./cost
    return dict(attempts=attempts,guidance_resets=resets,pheromone_edges=len(tau),deposit='1/J (adapted from original Q/L)',ants_per_iteration=ants)


def multiweight_routes(graph,offer,deadline):
    pairs=[(float(a),float(b)) for a in np.geomspace(.3,30,13) for b in np.geomspace(.01,1,7)]
    # Fixed ratio order spans the factor range; original paper does not publish
    # its exact adaptively chosen 91 pairs. Never tune pairs by validation.
    evaluated=[]
    for a,b in pairs:
        if time.perf_counter()>=deadline:break
        try:path,_=graph.shortest(alpha=a/b,kappa=.5,deadline=deadline)
        except SearchExpired:break
        rec=offer(path,'multiweight');evaluated.append((a,b))
        if rec is not None:rec['generation_alpha']=a/b
    return dict(factor_pairs=evaluated,requested_factor_pairs=91)


def soft_penalty_routes(graph,offer,deadline,max_attempts=125,radius_km=.6):
    penalty=np.zeros(len(graph.points));attempts=0
    for _ in range(max_attempts):
        if time.perf_counter()>=deadline:break
        try:path,_=graph.shortest(penalty=penalty,deadline=deadline)
        except SearchExpired:break
        offer(path,'soft_penalty');attempts+=1
        if path is None:break
        # Sample full segments at <=.1 km, so this is a corridor buffer rather
        # than a penalty only at sparsely spaced control points.
        samples=[]
        for a,b in zip(path,path[1:]):
            aa=np.asarray(a)*graph.grid.resolution_km;bb=np.asarray(b)*graph.grid.resolution_km
            samples.extend(np.linspace(aa,bb,max(2,math.ceil(np.linalg.norm(bb-aa)/.1)+1)))
        distances=cKDTree(samples).query(graph.xy)[0]
        penalty+=np.maximum(0.,1-distances/radius_km)
    return dict(attempts=attempts,radius_km=radius_km,penalty='cumulative triangular spatial penalty, unit strength per route')


class VGGRouteCluster:
    def __init__(self,weights_path,device='cpu'):
        import torch
        from torchvision.models import vgg16
        self.torch=torch;self.device=device
        # Official weights, safe tensor-only deserialization. No random feature
        # fallback is permitted if the required pretrained asset is missing.
        model=vgg16(weights=None);model.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True))
        self.features=model.features.to(device).eval()
        self.pool=model.avgpool.to(device).eval()
        del model
    def features_for(self,paths,shape):
        from PIL import Image,ImageDraw
        tensors=[];torch=self.torch
        for path in paths:
            im=Image.new('RGB',(shape[1],shape[0]),'black');ImageDraw.Draw(im).line([tuple(p) for p in path],fill='white',width=1)
            im=im.resize((224,224),Image.Resampling.BILINEAR)
            x=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255
            x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None];tensors.append(x)
        out=[]
        with torch.inference_mode():
            for i in range(0,len(tensors),4):
                values=self.pool(self.features(torch.stack(tensors[i:i+4]).to(self.device))).flatten(1)
                out.append(values.cpu().numpy())
        return np.concatenate(out) if out else np.empty((0,25088))
    def representatives(self,records,shape,seed):
        from sklearn.cluster import KMeans
        unique={canonical_path(r['path']):r for r in sorted(records,key=lambda r:(r['fitness'],r['route_id']),reverse=True)}
        routes=list(unique.values())
        if not routes:return [],dict(k=0)
        features=self.features_for([r['path'] for r in routes],shape);models=[];inertias=[]
        for k in range(1,min(5,len(routes))+1):
            km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(features);models.append(km);inertias.append(float(km.inertia_))
        if len(models)<3 or inertias[0]==inertias[-1]:idx=len(models)-1
        else:
            # Explicit deterministic elbow completion: max distance below chord
            # of normalized k/inertia; prefer smaller k on an exact tie.
            xx=np.linspace(0,1,len(models));yy=(np.asarray(inertias)-inertias[-1])/(inertias[0]-inertias[-1]);idx=int(np.argmax(1-xx-yy))
        model=models[idx];ids=[]
        for cluster in range(idx+1):
            members=np.flatnonzero(model.labels_==cluster);j=min(members,key=lambda j:(np.linalg.norm(features[j]-model.cluster_centers_[cluster]),routes[j]['fitness'],routes[j]['route_id']));ids.append(routes[j]['route_id'])
        return ids,dict(k=idx+1,inertias=inertias,feature_layer='ImageNet VGG16 conv+avgpool flatten, 25088D; paper did not specify layer',elbow='normalized max chord gap',seed=seed)


def refine_rrtstar(core,guide,seed,deadline,iterations=150,buffer_km=.25):
    """Seeded geometric RRT* inside the paper's guide corridor (adaptation).

    Seed the feasible guide for a meaningful finite-time incumbent. Rewiring
    propagates geometric costs to descendants and forbids ancestry cycles.
    All straight links are checked on H, plus nine corridor samples. Final J
    and simplicity remain the caller's authoritative check. No claim of
    asymptotic optimality for this seeded, capped, corridor-sampled adaptation.
    """
    g=core.corridor_map;rng=np.random.default_rng(seed);points=[np.asarray(p,float) for p in canonical_path(guide)];parents=[-1]+list(range(len(points)-1));cost=[0.]
    for a,b in zip(points,points[1:]):cost.append(cost[-1]+np.linalg.norm(b-a))
    goal=len(points)-1;guide=np.asarray(guide,float);aa=guide[:-1];vv=guide[1:]-aa;radius=buffer_km/g.resolution_km
    def inside(p):
        ts=np.clip(np.sum((p-aa)*vv,axis=1)/np.sum(vv*vv,axis=1),0,1)
        return np.min(np.linalg.norm(aa+ts[:,None]*vv-p,axis=1))<=radius+1e-12
    def valid(a,b):
        if not all(inside(p) for p in np.linspace(a,b,9)):return False
        touched,_=_segment_parts(a,b,g.shape)
        return not any(g.forbidden[y,x] for x,y in touched)
    for _ in range(iterations):
        if time.perf_counter()>deadline:break
        j=int(rng.integers(len(aa)));sample=aa[j]+rng.random()*vv[j]+rng.uniform(-radius,radius,size=2)
        if not g.contains(sample) or not inside(sample):continue
        arr=np.array(points);nearest=int(np.argmin(np.linalg.norm(arr-sample,axis=1)));delta=sample-points[nearest];distance=np.linalg.norm(delta)
        if distance<1e-12:continue
        sample=points[nearest]+delta*min(1.,radius/distance)
        near=[i for i,p in enumerate(points) if np.linalg.norm(p-sample)<=2*radius];options=[i for i in near if valid(points[i],sample)]
        if not options:continue
        parent=min(options,key=lambda i:cost[i]+np.linalg.norm(points[i]-sample));idx=len(points);points.append(sample);parents.append(parent);cost.append(cost[parent]+np.linalg.norm(points[parent]-sample))
        ancestors=set();j=parent
        while j>=0:ancestors.add(j);j=parents[j]
        for j in options:
            val=cost[idx]+np.linalg.norm(points[j]-sample)
            if j in ancestors or val>=cost[j]-1e-12:continue
            parents[j]=idx;cost[j]=val
            queue=[j]
            while queue:
                node=queue.pop()
                for child in range(len(points)):
                    if parents[child]==node:cost[child]=cost[node]+np.linalg.norm(points[child]-points[node]);queue.append(child)
    ids=[];j=goal
    while j>=0:ids.append(j);j=parents[j]
    return [points[i].tolist() for i in reversed(ids)]
