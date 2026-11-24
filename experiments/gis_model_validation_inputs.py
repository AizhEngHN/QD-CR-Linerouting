"""Routing model adapter extracted from the frozen GIS experiments."""
import math

import numpy as np

from gis_comparison_adapter import GISTurnGraph, legacy_for


class ModelParameterGraph(GISTurnGraph):
    """Frozen shortest-path mechanism, with current model coefficients."""
    def shortest(self,alpha=1.,kappa=.5,penalty=None,deadline=math.inf):
        # alpha here is a relative generation preference (1 for a baseline,
        # a/b for the frozen multiweight sequence), not a second objective.
        return super().shortest(alpha=self.core.weights.resistance_weight*alpha,
            kappa=self.core.weights.turn_equivalent_km,penalty=penalty,deadline=deadline)

def weighted_legacy(core):
    legacy=legacy_for(core)
    legacy.grid=np.where(core.corridor_map.forbidden,-1.,core.weights.resistance_weight*core.corridor_map.resistance)
    return legacy
