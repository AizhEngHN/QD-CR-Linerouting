"""Exact pair-distance memoization around frozen quality-matched retention.

This changes where a deterministic norm is computed, never selection semantics.
Per-policy local caches/pruning and logical distance counters are preserved.
The shared cache is for one fixed physical descriptor representation/stream;
costs may change between model profiles but a route ID's D vector may not.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import weakref

import numpy as np

CODE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(CODE/'src'),str(CODE/'experiments')]
from quality_matched_retention import QualityMatchedRetention
from replay_retention import RouteRecord


class SharedDistanceCache:
    """Bind IDs to immutable feature fingerprints and memoize exact norms.

    ``register_all`` can validate the entire input stream before replay,
    including offers whose high cost means they never enter a distance query.
    A weak identity cache avoids re-hashing the same immutable feature object;
    it does not keep the large arrays alive between cost-profile replays.
    """
    def __init__(self, namespace='fixed_physical_D_stream'):
        if not isinstance(namespace,str) or not namespace:
            raise ValueError('namespace must be nonempty text')
        self.namespace=namespace
        self._fingerprints={}
        self._verified_arrays={}
        self._distances={}
        self._dimension=None
        self.stats=dict(register_calls=0,feature_fingerprint_computations=0,
            verified_object_hits=0,distance_requests=0,shared_pair_cache_hits=0,
            actual_pair_norm_computations=0,same_id_zero_queries=0)

    def register(self,record):
        if not isinstance(record,RouteRecord):
            raise TypeError('shared distances require immutable RouteRecord objects')
        self.stats['register_calls']+=1
        array=record.feature
        if self._dimension is not None and array.size != self._dimension:
            raise ValueError('shared cache descriptor dimension changed')
        identity=id(array)
        previous=self._verified_arrays.get(identity)
        if previous is not None and previous[0]() is array:
            digest=previous[1]
            self.stats['verified_object_hits']+=1
        else:
            # RouteRecord guarantees nonempty finite C-contiguous float64 data
            # backed by immutable bytes, so this fingerprint stays valid.
            digest=hashlib.sha256(memoryview(array).cast('B')).hexdigest()
            self.stats['feature_fingerprint_computations']+=1
            def remove(reference, key=identity):
                cached=self._verified_arrays.get(key)
                if cached is not None and cached[0] is reference:
                    del self._verified_arrays[key]
            reference=weakref.ref(array,remove)
            self._verified_arrays[identity]=(reference,digest)
        expected=self._fingerprints.get(record.route_id)
        if expected is not None and expected != digest:
            raise ValueError('route_id was reused with a different descriptor feature')
        self._fingerprints[record.route_id]=digest
        self._dimension=array.size
        return digest

    def register_all(self,records):
        for record in records:
            self.register(record)

    def distance(self,first,second):
        self.register(first);self.register(second)
        self.stats['distance_requests']+=1
        if first.route_id==second.route_id:
            self.stats['same_id_zero_queries']+=1
            return 0.
        pair=tuple(sorted((first.route_id,second.route_id)))
        if pair not in self._distances:
            # Deliberately identical arithmetic to the frozen implementation.
            # Do not substitute Gram matrices, squared-distance shortcuts,
            # dimension reduction or different BLAS contractions here.
            self._distances[pair]=float(np.linalg.norm(first.feature-second.feature))
            self.stats['actual_pair_norm_computations']+=1
        else:
            self.stats['shared_pair_cache_hits']+=1
        return self._distances[pair]

    def snapshot(self,include_fingerprints=False):
        result=dict(namespace=self.namespace,descriptor_dimension=self._dimension,
            registered_route_ids=len(self._fingerprints),shared_pair_entries=len(self._distances),
            verified_live_array_objects=len(self._verified_arrays),stats=dict(self.stats),
            counter_scope='Shared memoized pair norms only; excludes direct consistency norms inside frozen offer and fingerprint hashing.')
        if include_fingerprints:
            result['feature_fingerprints']=dict(sorted(self._fingerprints.items()))
        return result


class CachedQualityMatchedRetention(QualityMatchedRetention):
    """Frozen policy with only its scalar distance computation memoized.

    ``stats.distance_computations`` remains the original per-policy local miss
    count, including when a shared memo supplied the value.  The shared object
    separately records actual norm calculations. Offer/quality/fill/ranking/
    snapshot/pruning all remain inherited from the existing source classes.
    """
    def __init__(self,initial_records,selector='competition',capacity=None,
                 online_tolerance=.1,tie_salt='20261031',*,shared_cache):
        if not isinstance(shared_cache,SharedDistanceCache):
            raise TypeError('shared_cache must be SharedDistanceCache')
        self.shared_cache=shared_cache
        initial_records=list(initial_records)
        shared_cache.register_all(initial_records)
        super().__init__(initial_records,selector=selector,capacity=capacity,
            online_tolerance=online_tolerance,tie_salt=tie_salt)

    def _distance(self,first,second):
        # The ordering of logical-counter updates and local cache lifetime is
        # copied exactly; only the miss calculation is delegated to the memo.
        if first.route_id==second.route_id:
            self.shared_cache.register(first);self.shared_cache.register(second)
            return 0.
        pair=tuple(sorted((first.route_id,second.route_id)))
        if pair not in self._distance_cache:
            self._distance_cache[pair]=self.shared_cache.distance(first,second)
            self.stats['distance_computations']+=1
            self.stats['max_distance_cache_entries']=max(
                self.stats['max_distance_cache_entries'],len(self._distance_cache))
        else:
            # Validate object identity/fingerprint even on local cache hits.
            self.shared_cache.register(first);self.shared_cache.register(second)
            self.stats['distance_cache_hits']+=1
        return self._distance_cache[pair]
