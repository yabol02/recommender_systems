from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp


class SimilarityFunction(ABC):
    """Interface for user-user similarity functions.

    Implement `compute` to add a new metric: it receives a batch of query user-vectors and the full training matrix,
    and returns a (B, N) similarity matrix (higher = more similar).
    """

    @abstractmethod
    def compute(
        self,
        query: sp.csr_matrix,  # (B, n_items) - batch of query users
        train: sp.csr_matrix,  # (N, n_items) - all training users
    ) -> np.ndarray:  # (B, N)
        """Compute pairwise similarities between query and training users.

        :param query: Sparse matrix of shape (B, n_items) for the batch of query users.
        :param train: Sparse matrix of shape (N, n_items) for all training users.
        :return: A 2D numpy array of shape (B, N) where each entry (i, j) represents the similarity between query user i and training user j.
        """
        ...

    def __call__(self, query: sp.csr_matrix, train: sp.csr_matrix) -> np.ndarray:
        return self.compute(query, train)


class EuclideanSimilarity(SimilarityFunction):
    """sim(u, v) = 1 / (1 + ||u − v||₂)

    Unrated items are treated as 0.
    The squared distance is expanded via ||u − v||² = ||u||² + ||v||² − 2·(u·v) to stay in sparse arithmetic.
    """

    def compute(self, query: sp.csr_matrix, train: sp.csr_matrix) -> np.ndarray:
        sq_q = np.asarray(query.power(2).sum(axis=1))  # (B, 1)
        sq_t = np.asarray(train.power(2).sum(axis=1)).T  # (1, N)
        dot = (query @ train.T).toarray()  # (B, N)

        sq_dist = sq_q + sq_t - 2.0 * dot
        np.clip(sq_dist, 0.0, None, out=sq_dist)  # guard float noise
        return 1.0 / (1.0 + np.sqrt(sq_dist))


class CosineSimilarity(SimilarityFunction):
    """sim(u, v) = (u · v) / (||u|| · ||v||)

    Unrated items are treated as 0.
    Purely angle-based: does not penalise rating-scale differences between users, making it sensitive to user bias.
    """

    def compute(self, query: sp.csr_matrix, train: sp.csr_matrix) -> np.ndarray:
        dot = (query @ train.T).toarray()  # (B, N)
        q_norms = np.sqrt(np.asarray(query.power(2).sum(axis=1)))  # (B, 1)
        t_norms = np.sqrt(np.asarray(train.power(2).sum(axis=1))).T  # (1, N)
        denom = q_norms * t_norms
        return np.where(denom > 0, dot / denom, 0.0)


class PearsonSimilarity(SimilarityFunction):
    """Pearson Correlation Coefficient via mean-centered cosine similarity.

    Each user's ratings are centered by subtracting their individual mean (computed only over *rated* items); unrated entries remain 0 after centering.
    Cosine similarity on the centered matrix equals the Pearson correlation under this convention.
    """

    def _mean_center_sparse(self, matrix: sp.csr_matrix) -> sp.csr_matrix:
        """Subtract each row's mean (over *rated* items only) from its nonzero entries.

        Unrated entries stay at 0 - they do not contribute to the per-row mean and they are not modified.
        This is the standard pre-processing step for Pearson-based similarity.
        """
        m = matrix.copy().astype(np.float64)
        counts = np.diff(m.indptr)  # number of rated items per user (N,)
        sums = np.asarray(m.sum(axis=1)).ravel()
        means = np.where(counts > 0, sums / counts, 0.0)
        m.data -= np.repeat(
            means, counts
        )  # subtract each row's mean from its nonzero data entries
        return m

    def compute(self, query: sp.csr_matrix, train: sp.csr_matrix) -> np.ndarray:
        q_c = self._mean_center_sparse(query)
        t_c = self._mean_center_sparse(train)

        dot = (q_c @ t_c.T).toarray()  # (B, N)
        q_norms = np.sqrt(np.asarray(q_c.power(2).sum(axis=1)))  # (B, 1)
        t_norms = np.sqrt(np.asarray(t_c.power(2).sum(axis=1))).T  # (1, N)
        denom = q_norms * t_norms
        return np.where(denom > 0, dot / denom, 0.0)


class JMSDSimilarity(SimilarityFunction):
    """JMSD: Jaccard weight × (1 − Mean Squared Difference).

    Combines two complementary signals:
    - **Jaccard** penalises pairs who have co-rated few items in common, preventing overconfident similarities from sparse overlap.
    - **MSD** penalises large rating discrepancies on co-rated items, normalised by the rating scale range.

    JMSD(u,v) = |I_uv| / |I_u ∪ I_v|  ×  (1 − MSD(u,v))

    where  MSD(u,v) = (1/|I_uv|) Σ_{i∈I_uv} ((r_ui − r_vi) / (r_max−r_min))²

    :param r_min: Minimum possible rating in the dataset.
    :param r_max: Maximum possible rating in the dataset.
    """

    def __init__(self, r_min: float = 1.0, r_max: float = 10.0) -> None:
        if r_max <= r_min:
            raise ValueError(f"r_max ({r_max}) must be greater than r_min ({r_min})")
        self.r_min = r_min
        self.r_max = r_max

    def compute(self, query: sp.csr_matrix, train: sp.csr_matrix) -> np.ndarray:
        range_sq = (self.r_max - self.r_min) ** 2

        q_bin = (query != 0).astype(np.float64)  # binary mask: rated = 1
        t_bin = (train != 0).astype(np.float64)

        # Jaccard
        co = (q_bin @ t_bin.T).toarray()  # (B, N)
        q_cnt = np.asarray(q_bin.sum(axis=1))  # (B, 1)
        t_cnt = np.asarray(t_bin.sum(axis=1)).T  # (1, N)
        union = q_cnt + t_cnt - co
        jaccard = np.where(union > 0, co / union, 0.0)

        # dot = Σ r_ui·r_vi  (only co-rated items; unrated entries are 0)
        dot = (query @ train.T).toarray()  # (B, N)
        # sum_sq_q[b,n] = Σ r_{q_b,i}² for items rated by train user n
        sum_sq_q = (query.power(2) @ t_bin.T).toarray()  # (B, N)
        # sum_sq_t[b,n] = Σ r_{t_n,i}² for items rated by query user b
        sum_sq_t = (q_bin @ train.power(2).T).toarray()  # (B, N)

        msd_num = sum_sq_q + sum_sq_t - 2.0 * dot
        np.clip(msd_num, 0.0, None, out=msd_num)  # float noise guard
        msd = np.where(co > 0, msd_num / (co * range_sq), 1.0)
        np.clip(msd, 0.0, 1.0, out=msd)

        return jaccard * (1.0 - msd)


SIMILARITY_REGISTRY: dict[str, SimilarityFunction] = {
    "euclidean": EuclideanSimilarity(),
    "cosine": CosineSimilarity(),
    "pearson": PearsonSimilarity(),
    "jmsd": JMSDSimilarity(),
}
