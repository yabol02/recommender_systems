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


COLD_START_REGISTRY: dict[str, type[ColdStartHandler]] = {
    "static": StaticFallback,
    "mean": MeanFallback,
}
