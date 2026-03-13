from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PredictionStrategy(ABC):
    """
    Interface for aggregating K neighbours' ratings into a single prediction.

    Implement `predict` to add a new aggregation method.
    """

    @abstractmethod
    def predict(
        self,
        similarities: np.ndarray,  # (k,) — similarity of each neighbour
        ratings: np.ndarray,  # (k,) — neighbour's rating for the target item
    ) -> float:
        """Make a prediction based on the similarities and ratings of K neighbours.

        :param similarities: Similarity scores of the K neighbours to the target user.
        :param ratings: Ratings given by the K neighbours for the target item.
        :return: Predicted rating for the target item.
        """
        ...

    def __call__(self, similarities: np.ndarray, ratings: np.ndarray) -> float:
        return self.predict(similarities, ratings)


class WeightedMeanPrediction(PredictionStrategy):
    """
    Similarity-weighted mean:  ŷ = Σ sim_i · r_i / Σ sim_i

    Falls back to an unweighted mean when all similarities are zero.
    """

    def predict(self, similarities: np.ndarray, ratings: np.ndarray) -> float:
        weights = np.clip(similarities, 0.0, None)
        total = weights.sum()
        if total == 0.0:
            return float(ratings.mean())
        return float(np.dot(weights, ratings) / total)


STRATEGY_REGISTRY: dict[str, PredictionStrategy] = {
    "w_mean": WeightedMeanPrediction(),
}
