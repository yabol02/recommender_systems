from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp


class SimilarityFunction(ABC):
    """
    Interface for user-user similarity functions.

    Implement `compute` to add a new metric: it receives a batch of query user-vectors and the
    full training matrix, and returns a (B, N) similarity matrix (higher = more similar).
    """

    @abstractmethod
    def compute(
        self,
        query: sp.csr_matrix,  # (B, n_items) — batch of query users
        train: sp.csr_matrix,  # (N, n_items) — all training users
    ) -> np.ndarray:  # (B, N)
        """
        Computes the similarity between each query user and each training user.

        :param query: A sparse matrix of shape (B, n_items) representing the batch of query user-vectors.
        :param train: A sparse matrix of shape (N, n_items) representing the full training user-item matrix.
        :return: A 2D numpy array of shape (B, N) where each entry (i, j) represents the similarity between query user i and training user j.
        """
        ...

    def __call__(self, query: sp.csr_matrix, train: sp.csr_matrix) -> np.ndarray:
        return self.compute(query, train)


class EuclideanSimilarity(SimilarityFunction):
    """
    sim(u, v) = 1 / (1 + ||u − v||₂)

    Unrated items are treated as 0.  The squared distance is expanded as:
        ||u − v||² = ||u||² + ||v||² − 2·(u · v)
    which lets us stay in sparse-matrix arithmetic until the final subtraction.
    """

    def compute(self, query: sp.csr_matrix, train: sp.csr_matrix) -> np.ndarray:
        sq_q = np.asarray(query.power(2).sum(axis=1))  # (B, 1)
        sq_t = np.asarray(train.power(2).sum(axis=1)).T  # (1, N)
        dot = (query @ train.T).toarray()  # (B, N)

        sq_dist = sq_q + sq_t - 2.0 * dot
        np.clip(sq_dist, 0.0, None, out=sq_dist)  # guard against float noise
        return 1.0 / (1.0 + np.sqrt(sq_dist))  # (B, N)


SIMILARITY_REGISTRY: dict[str, SimilarityFunction] = {
    "euclidean": EuclideanSimilarity(),
}
