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

    :param value: The constant prediction to return.
    """

    def __init__(self, value: float = 5.0) -> None:
        self.value = value

    def predict(self, uid: int, iid: int, status: ColdStartStatus) -> float:
        return self.value


class MeanFallback(ColdStartHandler):
    """Context-aware mean fallback.

    Applies the most informative available mean for each cold-start case:
    * `COLD_USER` - Item mean - we know how others rated this item
    * `UNKNOWN_ITEM` - User mean - we know how this user rates things
    * `BOTH` - Global mean - only global info available

    :param global_mean_init: Default global mean before `setup` is called (used as an emergency fallback).
    """

    def __init__(self, global_mean_init: float = 5.0) -> None:
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

    More robust to outliers than mean.

    :param global_median_init: Default global median before `setup` is called (used as an emergency fallback).
    """

    def __init__(self, global_median_init: float = 5.0) -> None:
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

    Useful when dealing with discrete rating scales (e.g., 1-5 stars).

    :param global_mode_init: Default global mode before `setup` is called (used as an emergency fallback).
    """

    def __init__(self, global_mode_init: float = 5.0) -> None:
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


COLD_START_REGISTRY: dict[str, type[ColdStartHandler]] = {
    "static": StaticFallback,
    "mean": MeanFallback,
    "median": MedianFallback,
    "mode": ModeFallback,
}
