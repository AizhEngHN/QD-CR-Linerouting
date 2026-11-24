"""Identical online quality/fill wrapper for two spatial retention selectors.

Only the over-capacity quality-qualified selection differs: nearest-cost-
predecessor competition versus best-seeded max-min spatial greedy selection.
The descriptor, route evaluator and candidate generator are external and frozen.
No historical candidates or future common evaluation reference are read here.
"""
from __future__ import annotations

import math

import numpy as np

from replay_retention import RetentionPolicy, RouteRecord


class QualityMatchedRetention(RetentionPolicy):
    """A bounded route set with shared eligibility, ties, dedup and cost fill.

    The constructor requires exactly K geometrically distinct input records.
    Every offer, including a cheaper encoding of current geometry, is processed
    through the shared wrapper. The online threshold is based only on the
    cheapest route currently available in ``current + candidate``.
    """

    def __init__(self, initial_records, selector='competition', capacity=None,
                 online_tolerance=.1, tie_salt='20261031'):
        if selector not in ('competition', 'greedy'):
            raise ValueError("selector must be 'competition' or 'greedy'")
        if not math.isfinite(online_tolerance) or online_tolerance < 0:
            raise ValueError('online_tolerance must be finite and nonnegative')
        # Reuse immutable records, stable cost/tie order and bounded distances.
        # The old module is not modified; its quality-filtering _greedy method
        # is deliberately not called, so filtering happens only once here.
        super().__init__(initial_records,
                         method='cost_ordered' if selector == 'competition' else 'spatial_greedy',
                         capacity=capacity, tie_salt=tie_salt,
                         greedy_tolerance=online_tolerance)
        self.selector = selector
        self.online_tolerance = float(online_tolerance)
        self.stats.update(quality_selection_calls=0, quality_selection_opportunities=0,
                          quality_fill_slots=0,
                          maximum_quality_fill_slots=0,
                          online_best_improvements=0)
        self._last_quality = None
        chosen, detail = self._select_pool(self._records)
        self._records = chosen
        self._last_quality = detail
        self._prune_caches()

    def _select_qualified_greedy(self, qualified):
        ordered = sorted(qualified, key=self._cost_order)
        selected = [ordered[0]]
        remaining = ordered[1:]
        nearest = {r.route_id: self._distance(r, selected[0]) for r in remaining}
        while remaining and len(selected) < self.capacity:
            chosen = min(remaining, key=lambda r: (-nearest[r.route_id],
                                                  self._raw_cost_order(r)))
            selected.append(chosen)
            remaining = [r for r in remaining if r.route_id != chosen.route_id]
            for route in remaining:
                nearest[route.route_id] = min(nearest[route.route_id],
                                              self._distance(route, chosen))
        return selected

    def _select_pool(self, routes):
        """Pure common wrapper apart from the qualified over-cap selector."""
        ordered = sorted(routes, key=self._cost_order)
        best = ordered[0]
        cutoff = best.cost * (1.0 + self.online_tolerance)
        qualified = [r for r in ordered if r.cost <= cutoff
                     or r.route_id == best.route_id]
        selection_applied = len(qualified) > self.capacity
        extra = {}
        if not selection_applied:
            selected = qualified.copy()
        elif self.selector == 'competition':
            selected, extra = self._competition(qualified)
        else:
            selected = self._select_qualified_greedy(qualified)
        chosen_quality_ids = [r.route_id for r in selected]
        chosen_ids = set(chosen_quality_ids)
        fill = []
        for route in ordered:
            if len(selected) >= self.capacity:
                break
            if route.route_id not in chosen_ids:
                selected.append(route)
                fill.append(route.route_id)
                chosen_ids.add(route.route_id)
        detail = dict(online_best_cost=best.cost, online_quality_cutoff=cutoff,
                      online_tolerance=self.online_tolerance,
                      quality_eligible_count=len(qualified),
                      quality_retained_count=len(chosen_quality_ids),
                      online_qualified_count=len(chosen_quality_ids),
                      selected_quality_ids=chosen_quality_ids,
                      quality_pool_ids=[r.route_id for r in qualified],
                      fill_ids=fill, quality_fill_slots=len(fill), fill_count=len(fill),
                      pool_order_ids=[r.route_id for r in ordered],
                      quality_selection_applied=selection_applied,
                      selection_branch='all_qualified_plus_cost_fill' if not selection_applied
                                       else 'qualified_over_capacity',
                      **extra)
        return selected, detail

    def offer(self, candidate):
        if not isinstance(candidate, RouteRecord):
            raise TypeError('candidate must be a RouteRecord')
        if candidate.feature.size != self.dimension:
            raise ValueError('candidate feature dimension differs')
        same_id = next((r for r in self._records if r.route_id == candidate.route_id), None)
        if same_id is not None and (same_id.path_key != candidate.path_key
                or same_id.cost != candidate.cost
                or not np.array_equal(same_id.feature, candidate.feature)):
            raise ValueError('a currently retained route_id cannot identify different data')
        before = self._records.copy()
        previous_best = min(r.cost for r in before)
        self.stats['offers'] += 1
        self.stats['equal_cost_offers'] += int(any(r.cost == candidate.cost for r in before))
        duplicate_index = next((i for i, r in enumerate(before)
                                if r.geometry_key == candidate.geometry_key), None)
        routes = before.copy()
        duplicate_of = None
        candidate_used = duplicate_index is None
        if duplicate_index is None:
            routes.append(candidate)
            input_reason = 'new_geometry'
        else:
            previous = before[duplicate_index]
            duplicate_of = previous.route_id
            self.stats['geometry_duplicate_offers'] += 1
            self.stats['exact_duplicate_offers'] += int(previous.path_key == candidate.path_key)
            if np.linalg.norm(candidate.feature - previous.feature) > self.separation:
                raise ValueError('same geometry has inconsistent features')
            if self._raw_cost_order(candidate) < self._raw_cost_order(previous):
                routes[duplicate_index] = candidate
                self.stats['geometry_replacements'] += 1
                candidate_used = True
                input_reason = 'geometry_cheaper_or_stable_tie'
            else:
                input_reason = 'geometry_not_better'
        # Even a same-geometry lower cost can move the eligibility threshold.
        # Both selectors therefore always execute this common wrapper.
        self._records, detail = self._select_pool(routes)
        self._last_quality = detail
        selected_ids = {r.route_id for r in self._records}
        deleted_ids = [r.route_id for r in before if r.route_id not in selected_ids]
        accepted = bool(candidate_used and candidate.route_id in selected_ids)
        if detail['online_best_cost'] < previous_best:
            self.stats['online_best_improvements'] += 1
        self.stats['quality_selection_calls'] += int(detail['quality_selection_applied'])
        self.stats['quality_selection_opportunities'] += int(detail['quality_selection_applied'])
        self.stats['quality_fill_slots'] += detail['quality_fill_slots']
        self.stats['maximum_quality_fill_slots'] = max(self.stats['maximum_quality_fill_slots'],
                                                       detail['quality_fill_slots'])
        self.stats['accepted' if accepted else 'rejected'] += 1
        self.stats['selection_changes'] += int(bool(deleted_ids))
        self._prune_caches()
        event = dict(candidate_id=candidate.route_id, candidate_cost=candidate.cost,
                     method='quality_matched_' + self.selector,
                     selector=self.selector, accepted=accepted,
                     reason=input_reason if duplicate_index is not None else
                            ('quality_wrapper_selected' if accepted else 'quality_wrapper_not_selected'),
                     input_reason=input_reason, geometry_duplicate_of=duplicate_of,
                     deleted_ids=deleted_ids,
                     retained_ids=[r.route_id for r in self._records],
                     retained_count=len(self._records),
                     geometry_unique_count=len({r.geometry_key for r in self._records}),
                     **detail)
        candidate_competition = detail.get('competition', {}).get(candidate.route_id)
        if candidate_competition is not None:
            event.update(candidate_competition)
        else:
            event.update(reference_id=None, score=None, score_infinite=False)
        return event

    def snapshot(self, include_records=True):
        value = super().snapshot(include_records=include_records)
        value.update(method='quality_matched_' + self.selector,
                     selector=self.selector, online_tolerance=self.online_tolerance,
                     quality=dict(self._last_quality))
        return value
