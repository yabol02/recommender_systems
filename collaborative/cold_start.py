from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np


class ColdStartStatus(Enum):
    OK = auto()  # user and item both present in train
    COLD_USER = auto()  # user not seen during training
    UNKNOWN_ITEM = auto()  # item not seen during training
    BOTH = auto()  # neither user nor item is in train


class ColdStartHandler(ABC):
    """Interface for producing fallback predictions in cold-start scenarios."""

    def setup(self, train_data: np.ndarray) -> None:
        """Pre-compute any statistics needed from training data.

        Call this once before any `predict` calls.

        :param train_data: Training array of shape (n, 3) - (user_id, item_id, rating).
        """

    @abstractmethod
    def predict(self, uid: int, iid: int, status: ColdStartStatus) -> float:
        """Return a fallback prediction for a cold-start case.

        :param uid: User ID from the test set.
        :param iid: Item ID from the test set.
        :param status: The cold-start status of this case.
        :return: Predicted rating.
        """
        ...


class StaticFallback(ColdStartHandler):
    """Always returns a fixed constant regardless of context.

    :param value: The constant prediction to return (default: 5.5 for 1-10 scale).
    """

    def __init__(self, value: float = 5.5) -> None:
        self.value = value

    def predict(self, uid: int, iid: int, status: ColdStartStatus) -> float:
        return self.value


class MeanFallback(ColdStartHandler):
    """Context-aware mean fallback.

    Applies the most informative available mean for each cold-start case:
    * `COLD_USER` - Item mean - we know how others rated this item
    * `UNKNOWN_ITEM` - User mean - we know how this user rates things
    * `BOTH` - Global mean - only global info available

    :param global_mean_init: Default global mean before `setup` is called (default: 5.5 for 1-10 scale).
    """

    def __init__(self, global_mean_init: float = 5.5) -> None:
        self._global_mean: float = global_mean_init
        self._item_means: dict[int, float] = {}
        self._user_means: dict[int, float] = {}

    def setup(self, train_data: np.ndarray) -> None:
        uids = train_data[:, 0].astype(int)
        iids = train_data[:, 1].astype(int)
        ratings = train_data[:, 2].astype(np.float64)

        self._global_mean = float(ratings.mean())

        # User means
        u_unique, u_inv = np.unique(uids, return_inverse=True)
        u_sums = np.bincount(u_inv, weights=ratings)
        u_cnts = np.bincount(u_inv).astype(np.float64)
        self._user_means = dict(zip(u_unique.tolist(), (u_sums / u_cnts).tolist()))

        # Item means
        i_unique, i_inv = np.unique(iids, return_inverse=True)
        i_sums = np.bincount(i_inv, weights=ratings)
        i_cnts = np.bincount(i_inv).astype(np.float64)
        self._item_means = dict(zip(i_unique.tolist(), (i_sums / i_cnts).tolist()))

    def predict(self, uid: int, iid: int, status: ColdStartStatus) -> float:
        if status == ColdStartStatus.COLD_USER:
            return self._item_means.get(iid, self._global_mean)
        if status == ColdStartStatus.UNKNOWN_ITEM:
            return self._user_means.get(uid, self._global_mean)
        # BOTH - neither user nor item is known
        return self._global_mean


class MedianFallback(ColdStartHandler):
    """Context-aware median fallback.

    Applies the most informative available median for each cold-start case:
    * `COLD_USER` - Item median - we know how others rated this item
    * `UNKNOWN_ITEM` - User median - we know how this user rates things
    * `BOTH` - Global median - only global info available

    More robust to outliers than mean. Ratings scale: 1-10.

    :param global_median_init: Default global median before `setup` is called (default: 5.5 for 1-10 scale).
    """

    def __init__(self, global_median_init: float = 5.5) -> None:
        self._global_median: float = global_median_init
        self._item_medians: dict[int, float] = {}
        self._user_medians: dict[int, float] = {}

    def setup(self, train_data: np.ndarray) -> None:
        uids = train_data[:, 0].astype(int)
        iids = train_data[:, 1].astype(int)
        ratings = train_data[:, 2].astype(np.float64)

        self._global_median = float(np.median(ratings))

        # User medians
        u_unique = np.unique(uids)
        self._user_medians = {}
        for u in u_unique:
            mask = uids == u
            self._user_medians[int(u)] = float(np.median(ratings[mask]))

        # Item medians
        i_unique = np.unique(iids)
        self._item_medians = {}
        for i in i_unique:
            mask = iids == i
            self._item_medians[int(i)] = float(np.median(ratings[mask]))

    def predict(self, uid: int, iid: int, status: ColdStartStatus) -> float:
        if status == ColdStartStatus.COLD_USER:
            return self._item_medians.get(iid, self._global_median)
        if status == ColdStartStatus.UNKNOWN_ITEM:
            return self._user_medians.get(uid, self._global_median)
        # BOTH - neither user nor item is known
        return self._global_median


class ModeFallback(ColdStartHandler):
    """Context-aware mode (most frequent value) fallback.

    Applies the most informative available mode for each cold-start case:
    * `COLD_USER` - Item mode - most common rating for this item
    * `UNKNOWN_ITEM` - User mode - most common rating by this user
    * `BOTH` - Global mode - most common rating overall

    Useful when dealing with discrete rating scales (e.g., 1-10).

    :param global_mode_init: Default global mode before `setup` is called (default: 5.5 for 1-10 scale).
    """

    def __init__(self, global_mode_init: float = 5.5) -> None:
        self._global_mode: float = global_mode_init
        self._item_modes: dict[int, float] = {}
        self._user_modes: dict[int, float] = {}

    def setup(self, train_data: np.ndarray) -> None:
        uids = train_data[:, 0].astype(int)
        iids = train_data[:, 1].astype(int)
        ratings = train_data[:, 2].astype(int)  # Treat ratings as integers for mode

        # Global mode
        unique_ratings, counts = np.unique(ratings, return_counts=True)
        self._global_mode = float(unique_ratings[np.argmax(counts)])

        # User modes
        u_unique = np.unique(uids)
        self._user_modes = {}
        for u in u_unique:
            mask = uids == u
            user_ratings = ratings[mask]
            unique_vals, counts = np.unique(user_ratings, return_counts=True)
            self._user_modes[int(u)] = float(unique_vals[np.argmax(counts)])

        # Item modes
        i_unique = np.unique(iids)
        self._item_modes = {}
        for i in i_unique:
            mask = iids == i
            item_ratings = ratings[mask]
            unique_vals, counts = np.unique(item_ratings, return_counts=True)
            self._item_modes[int(i)] = float(unique_vals[np.argmax(counts)])

    def predict(self, uid: int, iid: int, status: ColdStartStatus) -> float:
        if status == ColdStartStatus.COLD_USER:
            return self._item_modes.get(iid, self._global_mode)
        if status == ColdStartStatus.UNKNOWN_ITEM:
            return self._user_modes.get(uid, self._global_mode)
        # BOTH - neither user nor item is known
        return self._global_mode


class PopularityFallback(ColdStartHandler):
    """Popularity-based fallback using item rating counts.

    Recommends the most frequently rated items, as popularity is often
    a strong signal for new users (addressed in Pinterest & Netflix papers).

    * `COLD_USER` - Item popularity score (higher = more rated)
    * `UNKNOWN_ITEM` - Global popularity average
    * `BOTH` - Global popularity average

    :param global_popularity_init: Default popularity before `setup` is called.
    """

    def __init__(self, global_popularity_init: float = 5.5) -> None:
        self._global_popularity: float = global_popularity_init
        self._item_popularity: dict[int, float] = {}
        self._rating_counts: dict[int, int] = {}

    def setup(self, train_data: np.ndarray) -> None:
        iids = train_data[:, 1].astype(int)
        ratings = train_data[:, 2].astype(np.float64)

        # Count ratings per item
        unique_items, counts = np.unique(iids, return_counts=True)
        max_count = counts.max()
        
        # Normalize popularity to rating scale (1-10)
        # Maps highest count to 10.0, lowest to 1.0
        for item, count in zip(unique_items, counts):
            normalized = 1.0 + (count / max_count) * 9.0 if max_count > 0 else 1.0
            self._item_popularity[int(item)] = float(normalized)
            self._rating_counts[int(item)] = int(count)
        
        # Global average popularity
        if len(self._item_popularity) > 0:
            self._global_popularity = float(np.mean(list(self._item_popularity.values())))

    def predict(self, uid: int, iid: int, status: ColdStartStatus) -> float:
        if status == ColdStartStatus.COLD_USER:
            return self._item_popularity.get(iid, self._global_popularity)
        # UNKNOWN_ITEM or BOTH
        return self._global_popularity


class WeightedMeanFallback(ColdStartHandler):
    """Weighted mean fallback using rating count as weight.

    Applies weighted means where items/users with more ratings
    have higher confidence. Based on collaborative filtering confidence weighting.

    * `COLD_USER` - Item mean weighted by rating count
    * `UNKNOWN_ITEM` - User mean weighted by rating count
    * `BOTH` - Global mean

    :param global_mean_init: Default global mean before `setup` is called (default: 5.5 for 1-10 scale).
    """

    def __init__(self, global_mean_init: float = 5.5) -> None:
        self._global_mean: float = global_mean_init
        self._item_means: dict[int, float] = {}
        self._user_means: dict[int, float] = {}
        self._item_counts: dict[int, int] = {}
        self._user_counts: dict[int, int] = {}

    def setup(self, train_data: np.ndarray) -> None:
        uids = train_data[:, 0].astype(int)
        iids = train_data[:, 1].astype(int)
        ratings = train_data[:, 2].astype(np.float64)

        self._global_mean = float(ratings.mean())

        # User means and counts
        u_unique = np.unique(uids)
        for u in u_unique:
            mask = uids == u
            user_ratings = ratings[mask]
            self._user_means[int(u)] = float(user_ratings.mean())
            self._user_counts[int(u)] = int(mask.sum())

        # Item means and counts
        i_unique = np.unique(iids)
        for i in i_unique:
            mask = iids == i
            item_ratings = ratings[mask]
            self._item_means[int(i)] = float(item_ratings.mean())
            self._item_counts[int(i)] = int(mask.sum())

    def _dampen(self, mean: float, count: int, global_mean: float, damping: float = 5.0) -> float:
        """Apply damping: shrink toward global mean for low-count cases."""
        weight = count / (count + damping)
        return weight * mean + (1 - weight) * global_mean

    def predict(self, uid: int, iid: int, status: ColdStartStatus) -> float:
        if status == ColdStartStatus.COLD_USER:
            item_mean = self._item_means.get(iid, self._global_mean)
            count = self._item_counts.get(iid, 0)
            return self._dampen(item_mean, count, self._global_mean)
        
        if status == ColdStartStatus.UNKNOWN_ITEM:
            user_mean = self._user_means.get(uid, self._global_mean)
            count = self._user_counts.get(uid, 0)
            return self._dampen(user_mean, count, self._global_mean)
        
        return self._global_mean


class HybridFallback(ColdStartHandler):
    """Hybrid fallback combining multiple strategies intelligently.

    Implements a weighted ensemble approach:
    * `COLD_USER` - 80% item mean + 20% global mean
    * `UNKNOWN_ITEM` - 80% user mean + 20% global mean
    * `BOTH` - Global mean

    Inspired by ensemble methods in modern recommender systems (e.g., Netflix).

    :param global_mean_init: Default global mean before `setup` is called (default: 5.5 for 1-10 scale).
    :param item_weight: Weight for item statistics in COLD_USER case (0-1).
    :param user_weight: Weight for user statistics in UNKNOWN_ITEM case (0-1).
    """

    def __init__(
        self,
        global_mean_init: float = 5.5,
        item_weight: float = 0.8,
        user_weight: float = 0.8,
    ) -> None:
        self._global_mean: float = global_mean_init
        self._item_means: dict[int, float] = {}
        self._user_means: dict[int, float] = {}
        self._item_weight = item_weight
        self._user_weight = user_weight

    def setup(self, train_data: np.ndarray) -> None:
        uids = train_data[:, 0].astype(int)
        iids = train_data[:, 1].astype(int)
        ratings = train_data[:, 2].astype(np.float64)

        self._global_mean = float(ratings.mean())

        # User means
        u_unique = np.unique(uids)
        for u in u_unique:
            mask = uids == u
            self._user_means[int(u)] = float(ratings[mask].mean())

        # Item means
        i_unique = np.unique(iids)
        for i in i_unique:
            mask = iids == i
            self._item_means[int(i)] = float(ratings[mask].mean())

    def predict(self, uid: int, iid: int, status: ColdStartStatus) -> float:
        if status == ColdStartStatus.COLD_USER:
            item_mean = self._item_means.get(iid, self._global_mean)
            return (
                self._item_weight * item_mean +
                (1 - self._item_weight) * self._global_mean
            )
        
        if status == ColdStartStatus.UNKNOWN_ITEM:
            user_mean = self._user_means.get(uid, self._global_mean)
            return (
                self._user_weight * user_mean +
                (1 - self._user_weight) * self._global_mean
            )
        
        return self._global_mean


class PercentileFallback(ColdStartHandler):
    """Percentile-based fallback (25th, 50th, or 75th percentile).

    Alternative to mean/median that uses a specific percentile.
    * 25th percentile - Conservative (lower ratings)
    * 50th percentile - Equivalent to MedianFallback
    * 75th percentile - Optimistic (higher ratings)

    Ratings scale: 1-10

    :param percentile: Which percentile to use (0-100). Default 50 (median).
    :param global_percentile_init: Default percentile before `setup` is called (default: 5.5 for 1-10 scale).
    """

    def __init__(self, percentile: float = 50.0, global_percentile_init: float = 5.5) -> None:
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")
        self._percentile = percentile
        self._global_percentile: float = global_percentile_init
        self._item_percentiles: dict[int, float] = {}
        self._user_percentiles: dict[int, float] = {}

    def setup(self, train_data: np.ndarray) -> None:
        uids = train_data[:, 0].astype(int)
        iids = train_data[:, 1].astype(int)
        ratings = train_data[:, 2].astype(np.float64)

        self._global_percentile = float(np.percentile(ratings, self._percentile))

        # User percentiles
        u_unique = np.unique(uids)
        for u in u_unique:
            mask = uids == u
            user_ratings = ratings[mask]
            self._user_percentiles[int(u)] = float(np.percentile(user_ratings, self._percentile))

        # Item percentiles
        i_unique = np.unique(iids)
        for i in i_unique:
            mask = iids == i
            item_ratings = ratings[mask]
            self._item_percentiles[int(i)] = float(np.percentile(item_ratings, self._percentile))

    def predict(self, uid: int, iid: int, status: ColdStartStatus) -> float:
        if status == ColdStartStatus.COLD_USER:
            return self._item_percentiles.get(iid, self._global_percentile)
        if status == ColdStartStatus.UNKNOWN_ITEM:
            return self._user_percentiles.get(uid, self._global_percentile)
        return self._global_percentile


class TrimmedMeanFallback(ColdStartHandler):
    """Trimmed mean fallback - media without extreme values.

    Removes a percentage of extreme values (top and bottom) before computing mean.
    Combines robustness of median with more data utilization.
    
    * `COLD_USER` - Item trimmed mean (excludes top/bottom 20%)
    * `UNKNOWN_ITEM` - User trimmed mean (excludes top/bottom 20%)
    * `BOTH` - Global trimmed mean

    More stable than mean, simpler than median-damping.

    :param trim_percent: Percentage to trim from each tail (0-50). Default 20.
    :param global_trimmed_mean_init: Default trimmed mean before `setup` (default: 5.5).
    """

    def __init__(self, trim_percent: float = 20.0, global_trimmed_mean_init: float = 5.5) -> None:
        if not 0 <= trim_percent <= 50:
            raise ValueError("trim_percent must be between 0 and 50")
        self._trim_percent = trim_percent
        self._global_trimmed_mean: float = global_trimmed_mean_init
        self._item_trimmed_means: dict[int, float] = {}
        self._user_trimmed_means: dict[int, float] = {}

    def setup(self, train_data: np.ndarray) -> None:
        from scipy.stats import trim_mean
        
        uids = train_data[:, 0].astype(int)
        iids = train_data[:, 1].astype(int)
        ratings = train_data[:, 2].astype(np.float64)

        # Global trimmed mean
        self._global_trimmed_mean = float(trim_mean(ratings, self._trim_percent / 100))

        # User trimmed means
        u_unique = np.unique(uids)
        for u in u_unique:
            mask = uids == u
            user_ratings = ratings[mask]
            if len(user_ratings) > 0:
                self._user_trimmed_means[int(u)] = float(trim_mean(user_ratings, self._trim_percent / 100))

        # Item trimmed means
        i_unique = np.unique(iids)
        for i in i_unique:
            mask = iids == i
            item_ratings = ratings[mask]
            if len(item_ratings) > 0:
                self._item_trimmed_means[int(i)] = float(trim_mean(item_ratings, self._trim_percent / 100))

    def predict(self, uid: int, iid: int, status: ColdStartStatus) -> float:
        if status == ColdStartStatus.COLD_USER:
            return self._item_trimmed_means.get(iid, self._global_trimmed_mean)
        if status == ColdStartStatus.UNKNOWN_ITEM:
            return self._user_trimmed_means.get(uid, self._global_trimmed_mean)
        return self._global_trimmed_mean


class MedianDampingFallback(ColdStartHandler):
    """Median with damping/confidence weighting by rating count.

    Combines median's robustness with confidence-based shrinkage.
    When fewer ratings exist, shrink toward global median for stability.
    
    * `COLD_USER` - Item median, dampened by item rating count
    * `UNKNOWN_ITEM` - User median, dampened by user rating count
    * `BOTH` - Global median

    Best for balancing robustness and data efficiency.

    :param global_median_init: Default global median before `setup` (default: 5.5).
    :param damping_factor: Shrinkage strength (higher = more conservative). Default 3.0.
    """

    def __init__(self, global_median_init: float = 5.5, damping_factor: float = 3.0) -> None:
        self._global_median: float = global_median_init
        self._item_medians: dict[int, float] = {}
        self._user_medians: dict[int, float] = {}
        self._item_counts: dict[int, int] = {}
        self._user_counts: dict[int, int] = {}
        self._damping_factor = damping_factor

    def setup(self, train_data: np.ndarray) -> None:
        uids = train_data[:, 0].astype(int)
        iids = train_data[:, 1].astype(int)
        ratings = train_data[:, 2].astype(np.float64)

        self._global_median = float(np.median(ratings))

        # User medians and counts
        u_unique = np.unique(uids)
        for u in u_unique:
            mask = uids == u
            user_ratings = ratings[mask]
            self._user_medians[int(u)] = float(np.median(user_ratings))
            self._user_counts[int(u)] = int(mask.sum())

        # Item medians and counts
        i_unique = np.unique(iids)
        for i in i_unique:
            mask = iids == i
            item_ratings = ratings[mask]
            self._item_medians[int(i)] = float(np.median(item_ratings))
            self._item_counts[int(i)] = int(mask.sum())

    def _dampen(self, median: float, count: int, global_median: float) -> float:
        """Apply damping: shrink toward global median for low-count cases."""
        weight = count / (count + self._damping_factor)
        return weight * median + (1 - weight) * global_median

    def predict(self, uid: int, iid: int, status: ColdStartStatus) -> float:
        if status == ColdStartStatus.COLD_USER:
            item_median = self._item_medians.get(iid, self._global_median)
            count = self._item_counts.get(iid, 0)
            return self._dampen(item_median, count, self._global_median)
        
        if status == ColdStartStatus.UNKNOWN_ITEM:
            user_median = self._user_medians.get(uid, self._global_median)
            count = self._user_counts.get(uid, 0)
            return self._dampen(user_median, count, self._global_median)
        
        return self._global_median


class IQROutlierFallback(ColdStartHandler):
    """IQR-based outlier detection with robust median fallback.

    Removes statistical outliers (beyond 1.5 * IQR) before computing median.
    Cleanest approach: robust to anomalous ratings.
    
    * `COLD_USER` - Item median (outliers removed)
    * `UNKNOWN_ITEM` - User median (outliers removed)
    * `BOTH` - Global median (outliers removed)

    Best for noisy data with spammy or glitchy ratings.

    :param global_median_init: Default global median before `setup` (default: 5.5).
    :param iqr_multiplier: IQR threshold multiplier (1.5 is standard). Default 1.5.
    """

    def __init__(self, global_median_init: float = 5.5, iqr_multiplier: float = 1.5) -> None:
        self._global_median: float = global_median_init
        self._item_medians: dict[int, float] = {}
        self._user_medians: dict[int, float] = {}
        self._iqr_multiplier = iqr_multiplier

    def _remove_outliers(self, values: np.ndarray) -> np.ndarray:
        """Remove values outside 1.5*IQR whiskers."""
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - self._iqr_multiplier * iqr
        upper_bound = q3 + self._iqr_multiplier * iqr
        return values[(values >= lower_bound) & (values <= upper_bound)]

    def setup(self, train_data: np.ndarray) -> None:
        uids = train_data[:, 0].astype(int)
        iids = train_data[:, 1].astype(int)
        ratings = train_data[:, 2].astype(np.float64)

        # Global median (after removing outliers)
        clean_ratings = self._remove_outliers(ratings)
        self._global_median = float(np.median(clean_ratings)) if len(clean_ratings) > 0 else 5.5

        # User medians (outliers removed)
        u_unique = np.unique(uids)
        for u in u_unique:
            mask = uids == u
            user_ratings = ratings[mask]
            clean_user_ratings = self._remove_outliers(user_ratings)
            if len(clean_user_ratings) > 0:
                self._user_medians[int(u)] = float(np.median(clean_user_ratings))
            else:
                self._user_medians[int(u)] = self._global_median

        # Item medians (outliers removed)
        i_unique = np.unique(iids)
        for i in i_unique:
            mask = iids == i
            item_ratings = ratings[mask]
            clean_item_ratings = self._remove_outliers(item_ratings)
            if len(clean_item_ratings) > 0:
                self._item_medians[int(i)] = float(np.median(clean_item_ratings))
            else:
                self._item_medians[int(i)] = self._global_median

    def predict(self, uid: int, iid: int, status: ColdStartStatus) -> float:
        if status == ColdStartStatus.COLD_USER:
            return self._item_medians.get(iid, self._global_median)
        if status == ColdStartStatus.UNKNOWN_ITEM:
            return self._user_medians.get(uid, self._global_median)
        return self._global_median


COLD_START_REGISTRY: dict[str, type[ColdStartHandler]] = {
    "static": StaticFallback,
    "mean": MeanFallback,
    "median": MedianFallback,
    "mode": ModeFallback,
    "popularity": PopularityFallback,
    "weighted_mean": WeightedMeanFallback,
    "hybrid": HybridFallback,
    "percentile": PercentileFallback,
    "trimmed_mean": TrimmedMeanFallback,
    "median_damping": MedianDampingFallback,
    "iqr_outlier": IQROutlierFallback,
}
