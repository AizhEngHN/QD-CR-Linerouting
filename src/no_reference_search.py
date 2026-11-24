"""No-reference online search adapters; preserve frozen QD and mutation."""
import math,time,copy
from closedloop_spatial_run import FixedBudgetCandidateRun,OnlineFixedBudgetRetention
from cost_stratified_retention import CostStratifiedRetention
from gis_comparison_adapter import restore_endpoints

class RankStratifiedRetention(CostStratifiedRetention):
    """Four online cost-rank strata, five slots each, no reference or cap.

    Ranks are computed only on the current 20 members and candidate; they are
    not global quantiles of past candidates. The cheapest seed in each stratum
    and within-stratum descriptor max-min follow the existing greedy control.
    """
    def __init__(self,initial,quotas=(5,5,5,5)):
        self.quotas=tuple(quotas);self.policy_role='online_rank_stratified_spatial_greedy'
        OnlineFixedBudgetRetention.__init__(self,initial,'competition',math.inf,capacity=sum(quotas))
    def _competition(self,routes):
        ordered=sorted(routes,key=self._cost_order);n=len(ordered);selected=[]
        buckets=[[] for _ in self.quotas]
        for i,r in enumerate(ordered):buckets[min(len(buckets)-1,i*len(buckets)//n)].append(r)
        for bucket,quota in zip(buckets,self.quotas):selected.extend(self._farthest_fill([],bucket,quota))
        chosen={r.route_id for r in selected};remaining=[r for r in routes if r.route_id not in chosen]
        return self._farthest_fill(selected,remaining,self.capacity),dict(bucket_sizes=[len(b) for b in buckets])

class NoReferenceRun(FixedBudgetCandidateRun):
    def __init__(self,method,core,descriptor,bank,seconds):
        dispatch='QD_DAM' if method=='QD' else 'SpatialGreedy_M'
        super().__init__(dispatch,core,descriptor,bank,math.inf,seconds)
        if method=='RankStratified':self.policy=RankStratifiedRetention(self.policy.records)
        elif method!='QD':raise ValueError('Unknown method')

class LIONHistoryRun(FixedBudgetCandidateRun):
    """LION generator, no QD retention/descriptor on its own proposals."""
    def __init__(self,method,core,descriptor,bank,seconds):
        super().__init__(method,core,descriptor,bank,math.inf,seconds)
    def offer(self,path,operator):
        slot=len(self.ledger);rid=f'{self.method}:{slot}'
        row=dict(slot=slot,route_id=rid,path=None,operator=operator,valid=False,fitness=None,
                 reason='construction_failed',late_excluded=False)
        self.ledger.append(row)
        if path is None:
            row['late_excluded']=time.perf_counter()>self.deadline;return None
        path=restore_endpoints(path,self.core);cost,details=self.core.evaluate(path);self.model_calls+=1
        row.update(path=path,valid=bool(details['valid']),fitness=cost if math.isfinite(cost) else None,reason=details['failure_reason'])
        if not details['valid'] or time.perf_counter()>self.deadline:
            row['late_excluded']=time.perf_counter()>self.deadline;return None
        rec=dict(route_id=rid,path=path,fitness=cost,cost_details=details,cost_unit=self.core.cost_unit,model_sha256=self.core.model_sha256)
        self.records[rid]=rec;row['finished_seconds']=time.perf_counter()-self.started
        if time.perf_counter()>self.deadline:
            self.records.pop(rid);row['late_excluded']=True;return None
        return rec
