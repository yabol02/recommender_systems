from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


class PredictionStrategy(ABC):
    """Interface for aggregating K neighbours' ratings into a single prediction.

    The optional `**context` dict may carry pre-computed user statistics needed by bias-corrected strategies.
    Simple strategies ignore it entirely, so existing code remains unaffected.

    Expected context keys (all optional):
        `query_user_mean`: float- mean rating of the query user
        `query_user_std`: float- std of ratings of the query user
        `neighbour_means`: ndarray (k,) - mean rating of each neighbour
        `neighbour_stds`: ndarray (k,) - std  of ratings of each neighbour
    """

    @abstractmethod
    def predict(
        self,
        similarities: np.ndarray,  # (k,) similarity to each neighbour
        ratings: np.ndarray,  # (k,) neighbour's rating for the target item
        **context,
    ) -> float:
        """Aggregate neighbour information into a predicted rating.

        :param similarities: Similarity scores of the K nearest neighbours.
        :param ratings: Ratings given by those neighbours for the target item.
        :param context: Optional user statistics for bias correction.
        :return: Predicted rating.
        """
        ...

    def __call__(
        self, similarities: np.ndarray, ratings: np.ndarray, **context
    ) -> float:
        return self.predict(similarities, ratings, **context)


class SimpleAveragePrediction(PredictionStrategy):
    """Unweighted mean of the K neighbours' ratings.

    Ignores similarity values entirely; useful as an ablation baseline.
    """

    def predict(
        self, similarities: np.ndarray, ratings: np.ndarray, **context
    ) -> float:
        return float(ratings.mean())


class WeightedMeanPrediction(PredictionStrategy):
    """Similarity-weighted mean:  ŷ = Σ sim_i·r_i / Σ sim_i

    Negative similarities are clipped to 0.
    Falls back to an unweighted mean when all similarities are zero or negative.
    """

    def predict(
        self, similarities: np.ndarray, ratings: np.ndarray, **context
    ) -> float:
        weights = np.clip(similarities, 0.0, None)
        total = weights.sum()
        if total == 0.0:
            return float(ratings.mean())
        return float(np.dot(weights, ratings) / total)


class MeanCenteredWeightedMean(PredictionStrategy):
    """Bias-corrected weighted mean (mean-centered).

    Corrects for each user's systematic rating tendency (generous / harsh raters):
        ŷ_u,i = r̄_u + Σ_v w_v · (r_v,i − r̄_v) / Σ|w_v|

    The neighbour ratings are centered around their own means before weighting, and the query user's mean is added back.
    Degrades gracefully to `WeightedMeanPrediction` if context keys are absent.
    """

    def predict(
        self, similarities: np.ndarray, ratings: np.ndarray, **context
    ) -> float:
        weights = np.clip(similarities, 0.0, None)
        total = weights.sum()

        query_mean: float | None = context.get("query_user_mean")
        nb_means: np.ndarray | None = context.get("neighbour_means")

        if query_mean is None or nb_means is None:
            return float(
                ratings.mean() if total == 0.0 else np.dot(weights, ratings) / total
            )

        centered = ratings - nb_means
        if total == 0.0:
            return float(query_mean + centered.mean())
        return float(query_mean + np.dot(weights, centered) / total)


class ZScoreWeightedMean(PredictionStrategy):
    """Bias-corrected weighted mean (z-score normalisation).

    Extends mean-centering by also normalising for rating spread:
        ŷ_u,i = r̄_u + σ_u · Σ_v w_v · (r_v,i − r̄_v)/σ_v / Σ|w_v|

    Each neighbour's deviation is expressed in units of their own std; the result is then rescaled to the query user's std.
    More aggressive than mean-centering; can help when users have very different rating dispersions.

    Degrades gracefully to `MeanCenteredWeightedMean` (or `WeightedMeanPrediction`) if any required context keys are absent.
    """

    def predict(
        self, similarities: np.ndarray, ratings: np.ndarray, **context
    ) -> float:
        weights = np.clip(similarities, 0.0, None)
        total = weights.sum()

        query_mean: float | None = context.get("query_user_mean")
        query_std: float | None = context.get("query_user_std")
        nb_means: np.ndarray | None = context.get("neighbour_means")
        nb_stds: np.ndarray | None = context.get("neighbour_stds")

        if any(v is None for v in [query_mean, query_std, nb_means, nb_stds]):
            # Degrade to mean-centered or plain weighted mean
            if query_mean is not None and nb_means is not None:
                centered = ratings - nb_means
                return float(
                    query_mean
                    + (
                        centered.mean()
                        if total == 0.0
                        else np.dot(weights, centered) / total
                    )
                )
            return float(
                ratings.mean() if total == 0.0 else np.dot(weights, ratings) / total
            )

        safe_nb_stds = np.where(nb_stds > 1e-9, nb_stds, 1.0)
        normalised = (ratings - nb_means) / safe_nb_stds

        safe_query_std = float(query_std) if float(query_std) > 1e-9 else 1.0

        if total == 0.0:
            return float(query_mean + safe_query_std * normalised.mean())
        return min(
            10,
            max(
                1,
                float(
                    query_mean + safe_query_std * np.dot(weights, normalised) / total
                ),
            ),
        )


class ModeVotation(PredictionStrategy):
    """Similarity-weighted majority vote among K neighbours' ratings.

    Each neighbour casts a vote for their (rounded) integer rating, weighted
    by their similarity to the query user.  The rating with the highest
    accumulated weight wins.  Ties are broken by picking the lowest rating
    among the winners (conservative / pessimistic tie-breaking).

    Falls back to an unweighted vote when all similarities are zero or negative.
    """

    def predict(
        self, similarities: np.ndarray, ratings: np.ndarray, **context
    ) -> float:
        if ratings.size == 0:
            return 0.0

        discrete = np.round(ratings).astype(int)

        weights = np.clip(similarities, 0.0, None)
        total_weight = weights.sum()

        # Fall back to uniform weights if all similarities are non-positive
        if total_weight == 0.0:
            weights = np.ones(len(ratings), dtype=np.float64)

        # Shift ratings so the minimum maps to index 0 (handles any rating scale)
        offset = discrete.min()
        shifted = discrete - offset

        weighted_counts = np.bincount(shifted, weights=weights)

        # Winner: highest accumulated weight; ties broken by lowest rating
        winner_shifted = int(np.argmax(weighted_counts))
        return float(winner_shifted + offset)


class PersonalDistribution(PredictionStrategy):
    """Quantile-mapping prediction onto the query user's personal rating distribution.

    Standard bias-correction strategies (like mean-centered and z-score) assume that rating distributions can be approximated by their mean and std.
    Normalising by mean/std and re-scaling back can land the prediction on a value the user has never actually given.
    This strategy maps the weighted mean deviation of the neighbours onto the query user's historical rating distribution via quantile matching.
    """

    def predict(
        self, similarities: np.ndarray, ratings: np.ndarray, **context
    ) -> float:
        weights = np.clip(similarities, 0.0, None)
        total = weights.sum()

        user_hist: np.ndarray | None = context.get("query_user_sorted_ratings")
        nb_means: np.ndarray | None = context.get("neighbour_means")
        query_mean: float | None = context.get("query_user_mean")

        if user_hist is None or len(user_hist) < 2:
            # Fall back to mean-centered or plain weighted mean
            if query_mean is not None and nb_means is not None:
                centered = ratings - nb_means
                return float(
                    query_mean
                    + (
                        centered.mean()
                        if total == 0.0
                        else np.dot(weights, centered) / total
                    )
                )
            return float(
                ratings.mean() if total == 0.0 else np.dot(weights, ratings) / total
            )

        # Weighted mean deviation from neighbour means
        if nb_means is not None:
            centered = ratings - nb_means
            if total == 0.0:
                ref_deviation = float(centered.mean())
            else:
                ref_deviation = float(np.dot(weights, centered) / total)
        else:
            ref_deviation = float(
                ratings.mean() if total == 0.0 else np.dot(weights, ratings) / total
            )

        # Turn the deviation into a percentile
        if nb_means is not None:
            deviations = ratings - nb_means  # (k,) centred deviations
        else:
            deviations = ratings - ratings.mean()

        dev_std = float(deviations.std())
        if dev_std < 1e-9:
            # All neighbours agree - map directly to the user's median
            percentile = 0.5
        else:
            # Assuming a normal distribution of deviations, compute the percentile of the reference deviation
            z = ref_deviation / (dev_std * math.sqrt(2.0))
            percentile = float(0.5 * (1.0 + math.erf(z)))
            percentile = max(0.0, min(1.0, percentile))

        # Map percentile onto the user's sorted rating history
        predicted = float(np.quantile(user_hist, percentile))
        return predicted


STRATEGY_REGISTRY: dict[str, PredictionStrategy] = {
    "mean": SimpleAveragePrediction(),
    "w_mean": WeightedMeanPrediction(),
    "mean_centered": MeanCenteredWeightedMean(),
    "z_score": ZScoreWeightedMean(),
    "vote_mode": ModeVotation(),
    "personal": PersonalDistribution(),
}
