from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from collaborative.cold_start import ColdStartHandler, ColdStartStatus, MeanFallback

# Configuration


@dataclass
class PMFConfig:
    """Hyperparameters for Probabilistic Matrix Factorization.

    :param n_factors: Dimensionality of the latent space (K).
    :param n_epochs: Maximum number of training epochs.
    :param lr: Learning rate for gradient updates.
    :param reg_user: L2 regularisation on user factors (λ_U).
    :param reg_item: L2 regularisation on item factors (λ_V).
    :param reg_bias: L2 regularisation on bias terms; only used when `use_biases=True`.
    :param tol: Early-stopping threshold - training halts when |Δloss| < tol.
    :param min_rating: Lower bound for clipping predicted ratings.
    :param max_rating: Upper bound for clipping predicted ratings.
    :param init_std: Std of the zero-mean Gaussian used to initialise latent factors.
    :param use_biases: Include a global μ, per-user b_u, and per-item b_i bias term.
    :param batch_size: Mini-batch size for SGD; `None` uses full-batch gradient descent.
    :param seed: Random seed for reproducibility.
    """

    n_factors: int | None = None
    n_epochs: int | None = None
    lr: float | None = None
    reg_user: float | None = None
    reg_item: float | None = None
    reg_bias: float | None = None
    tol: float | None = None
    min_rating: float | None = None
    max_rating: float | None = None
    init_std: float | None = None
    use_biases: bool | None = None
    batch_size: int | None = None  # None -> full-batch GD; int -> mini-batch SGD
    seed: int | None = None

    def __post_init__(self):
        # Assign default values if None
        if self.n_factors is None:
            self.n_factors = 10
        if self.n_epochs is None:
            self.n_epochs = 100
        if self.lr is None:
            self.lr = 0.005
        if self.reg_user is None:
            self.reg_user = 0.02
        if self.reg_item is None:
            self.reg_item = 0.02
        if self.reg_bias is None:
            self.reg_bias = 0.02
        if self.tol is None:
            self.tol = 1e-4
        if self.min_rating is None:
            self.min_rating = 1.0
        if self.max_rating is None:
            self.max_rating = 10.0
        if self.init_std is None:
            self.init_std = 0.1
        if self.use_biases is None:
            self.use_biases = True
        if self.seed is None:
            self.seed = 42

        # Validate parameters
        if self.n_factors <= 0:
            raise ValueError("n_factors must be positive.")
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be positive.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.reg_user < 0 or self.reg_item < 0 or self.reg_bias < 0:
            raise ValueError("Regularisation parameters must be non-negative.")
        if self.tol < 0:
            raise ValueError("tol must be non-negative.")
        if self.min_rating >= self.max_rating:
            raise ValueError("min_rating must be less than max_rating.")
        if self.init_std < 0:
            raise ValueError("init_std must be non-negative.")
        if self.batch_size is not None and (self.batch_size <= 0):
            raise ValueError("batch_size must be positive or None.")


# Model
class PMF:
    """Probabilistic Matrix Factorization (PMF) via gradient descent.

    Computes the MAP estimate of the PMF generative model, which is equivalent to minimising the regularised squared-error objective.
    Both U and V are updated simultaneously at each step: V's gradient uses the pre-step U, and vice versa.
    Training stops when `n_epochs` is reached or |Δloss| < `tol`.

    :param config: Model hyperparameters. Defaults to `PMFConfig()`.
    """

    def __init__(self, config: PMFConfig | None = None) -> None:
        self.config = config or PMFConfig()

        # Learned parameters
        self._U: np.ndarray | None = None  # (n_users, n_factors)
        self._V: np.ndarray | None = None  # (n_items, n_factors)
        self._bu: np.ndarray | None = None  # (n_users,)  user biases
        self._bi: np.ndarray | None = None  # (n_items,)  item biases
        self._mu: float = 0.0  # global mean bias

        # original_id -> compact matrix index
        self._user_idx: dict[int, int] = {}
        self._item_idx: dict[int, int] = {}

        self._fallback: ColdStartHandler | None = None
        self.loss_history: list[float] = []

    @property
    def n_users(self) -> int:
        """Number of users seen during training."""
        return len(self._user_idx)

    @property
    def n_items(self) -> int:
        """Number of items seen during training."""
        return len(self._item_idx)

    @property
    def is_fitted(self) -> bool:
        """Whether `fit()` has been called."""
        return self._U is not None

    def fit(self, train_data: np.ndarray) -> PMF:
        """Fit PMF on training triples.

        :param train_data: Array of shape (n, 3) with columns (user_id, item_id, rating).
        :return: `self` for method chaining.
        """
        conf = self.config
        rng = np.random.default_rng(conf.seed)

        users = train_data[:, 0].astype(int)
        items = train_data[:, 1].astype(int)
        ratings = train_data[:, 2].astype(np.float64)

        # Build compact index maps: original ID -> consecutive integer index
        unique_users = np.unique(users)
        unique_items = np.unique(items)
        self._user_idx = {int(uid): i for i, uid in enumerate(unique_users)}
        self._item_idx = {int(iid): j for j, iid in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)
        n_obs = len(ratings)

        # Pre-translate to compact indices
        u_idx = np.array([self._user_idx[u] for u in users], dtype=np.int32)
        i_idx = np.array([self._item_idx[i] for i in items], dtype=np.int32)

        # Global mean
        self._mu = float(ratings.mean())

        # Initialise latent factors ~ N(0, init_std²)
        self._U = (
            rng.standard_normal((n_users, conf.n_factors)).astype(np.float64)
            * conf.init_std
        )
        self._V = (
            rng.standard_normal((n_items, conf.n_factors)).astype(np.float64)
            * conf.init_std
        )

        # Bias terms start at zero
        self._bu = np.zeros(n_users, dtype=np.float64)
        self._bi = np.zeros(n_items, dtype=np.float64)

        self._fallback = MeanFallback()
        self._fallback.setup(train_data)
        self.loss_history = []
        prev_loss = float("inf")

        _print_header(conf, n_users, n_items, n_obs)

        for epoch in range(1, conf.n_epochs + 1):
            t0 = time.perf_counter()

            if conf.batch_size is None:
                self._gradient_step(u_idx, i_idx, ratings)
            else:
                order = rng.permutation(n_obs)
                for start in range(0, n_obs, conf.batch_size):
                    sl = slice(start, start + conf.batch_size)
                    self._gradient_step(
                        u_idx[order[sl]], i_idx[order[sl]], ratings[order[sl]]
                    )

            loss = self._compute_loss(u_idx, i_idx, ratings)
            self.loss_history.append(loss)

            delta = abs(prev_loss - loss)
            elapsed = time.perf_counter() - t0
            print(
                f"  Epoch {epoch:>4}/{conf.n_epochs} | loss={loss:12.4f} | Δloss={delta:.6f} | {elapsed:.2f}s"
            )

            if epoch > 1 and delta < conf.tol:
                print(f"\n  Early stopping: Δloss={delta:.2e} < tol={conf.tol:.2e}")
                break

            prev_loss = loss

        return self

    def predict_one(self, uid: int, iid: int) -> float:
        """Predict the rating for a single (user, item) pair.

        Falls back to the global mean for unseen IDs.
        Prefer `predict_test` for test-set inference where cold-start handling is needed.

        :param uid: Original user ID.
        :param iid: Original item ID.
        :return: Clipped predicted rating.
        """
        self._check_fitted()
        u = self._user_idx.get(int(uid))
        i = self._item_idx.get(int(iid))
        return self._score(u, i) if (u is not None and i is not None) else self._mu

    def predict_test(
        self,
        test_data: np.ndarray,
        cold_start_handler: ColdStartHandler | None = None,
    ) -> np.ndarray:
        """Generate predictions for the full test set, handling cold-start cases.

        OK cases are handled with a vectorised dot product; cold-start cases are dispatched to the handler one by one.

        :param test_data: Array of shape (n_test, 3) - columns (id, user_id, item_id).
        :param cold_start_handler: Fallback for cold-start cases. Defaults to the `MeanFallback` built during `fit`.
        :return: Array of shape (n_test, 2) - columns (id, predicted_rating).
        """
        self._check_fitted()
        handler = (
            cold_start_handler if cold_start_handler is not None else self._fallback
        )

        test_ids = test_data[:, 0].astype(int)
        test_users = test_data[:, 1].astype(int)
        test_items = test_data[:, 2].astype(int)

        # Map to compact indices; -1 signals an unseen ID
        u_compact = np.array(
            [self._user_idx.get(int(u), -1) for u in test_users], dtype=np.int32
        )
        i_compact = np.array(
            [self._item_idx.get(int(i), -1) for i in test_items], dtype=np.int32
        )

        ok_mask = (u_compact >= 0) & (i_compact >= 0)
        predictions = np.empty(len(test_data), dtype=np.float64)

        # Prediction for known users/items
        if ok_mask.any():
            u_ok, i_ok = u_compact[ok_mask], i_compact[ok_mask]
            scores = np.einsum("nd,nd->n", self._U[u_ok], self._V[i_ok])
            if self.config.use_biases:
                scores = scores + self._mu + self._bu[u_ok] + self._bi[i_ok]
            predictions[ok_mask] = np.clip(
                scores, self.config.min_rating, self.config.max_rating
            )

        # Cold-start fallback
        for idx in np.where(~ok_mask)[0]:
            uid, iid = int(test_users[idx]), int(test_items[idx])
            status = _cold_start_status(u_compact[idx], i_compact[idx])
            predictions[idx] = handler.predict(uid, iid, status)

        return np.column_stack([test_ids, predictions])

    def _gradient_step(
        self,
        u_idx: np.ndarray,
        i_idx: np.ndarray,
        ratings: np.ndarray,
    ) -> None:
        """One vectorised gradient step for a batch (full or mini).

        :param u_idx: Array of shape (batch,) with compact user indices.
        :param i_idx: Array of shape (batch,) with compact item indices.
        :param ratings: Array of shape (batch,) with true ratings.
        """
        conf = self.config
        lr = conf.lr

        errors = ratings - self._raw_scores(u_idx, i_idx)  # (batch,)
        err_col = errors[:, None]  # (batch, 1)  for broadcasting

        # Snapshot factor rows before any in-place modification
        v_snap = self._V[i_idx].copy()  # (batch, n_factors)
        u_snap = self._U[u_idx].copy()

        unique_u = np.unique(u_idx)
        unique_i = np.unique(i_idx)

        # Count hoy many times each user/item appears in the batch
        u_counts = np.bincount(u_idx, minlength=len(self._U)).astype(np.float64)
        i_counts = np.bincount(i_idx, minlength=len(self._V)).astype(np.float64)

        # ∂L/∂U[u] = λ_U·U[u] − Σ_{i∈Ω_u} e_ui · V[i]  ->  U[u] += lr·(Σ e_ui · V[i] − λ_U·U[u])
        delta_U = np.zeros_like(self._U)
        np.add.at(delta_U, u_idx, err_col * v_snap)
        delta_U[unique_u] /= u_counts[unique_u, np.newaxis]
        self._U[unique_u] += lr * (
            delta_U[unique_u] - conf.reg_user * self._U[unique_u]
        )

        # ∂L/∂V[i] = λ_V·V[i] − Σ_{u∈Ω_i} e_ui · U[u]  ->  V[i] += lr·(Σ e_ui · U[u] − λ_V·V[i])
        delta_V = np.zeros_like(self._V)
        np.add.at(delta_V, i_idx, err_col * u_snap)
        delta_V[unique_i] /= i_counts[unique_i, np.newaxis]
        self._V[unique_i] += lr * (
            delta_V[unique_i] - conf.reg_item * self._V[unique_i]
        )

        if conf.use_biases:
            delta_bu = np.zeros_like(self._bu)
            np.add.at(delta_bu, u_idx, errors)
            delta_bu[unique_u] /= u_counts[unique_u]
            self._bu[unique_u] += lr * (
                delta_bu[unique_u] - conf.reg_bias * self._bu[unique_u]
            )

            delta_bi = np.zeros_like(self._bi)
            np.add.at(delta_bi, i_idx, errors)
            delta_bi[unique_i] /= i_counts[unique_i]
            self._bi[unique_i] += lr * (
                delta_bi[unique_i] - conf.reg_bias * self._bi[unique_i]
            )

    def _raw_scores(self, u_idx: np.ndarray, i_idx: np.ndarray) -> np.ndarray:
        """Compute raw (unclipped) predicted ratings for batches of compact indices.

        :param u_idx: Array of shape (batch,) with compact user indices.
        :param i_idx: Array of shape (batch,) with compact item indices.
        :return: Array of shape (batch,) with raw predicted ratings.
        """
        scores = np.einsum("nd,nd->n", self._U[u_idx], self._V[i_idx])
        if self.config.use_biases:
            scores = scores + self._mu + self._bu[u_idx] + self._bi[i_idx]
        return scores

    def _compute_loss(
        self,
        u_idx: np.ndarray,
        i_idx: np.ndarray,
        ratings: np.ndarray,
    ) -> float:
        """Compute the full regularised PMF loss on the given batch.

        :param u_idx: Array of shape (batch,) with compact user indices.
        :param i_idx: Array of shape (batch,) with compact item indices.
        :param ratings: Array of shape (batch,) with true ratings.
        :return: Scalar loss value.
        """
        conf = self.config
        errors = ratings - self._raw_scores(u_idx, i_idx)
        loss = 0.5 * float(np.dot(errors, errors))
        loss += 0.5 * conf.reg_user * float(np.sum(self._U**2))
        loss += 0.5 * conf.reg_item * float(np.sum(self._V**2))
        if conf.use_biases:
            loss += (
                0.5
                * conf.reg_bias
                * (
                    float(np.dot(self._bu, self._bu))
                    + float(np.dot(self._bi, self._bi))
                )
            )
        return loss

    def _score(self, u: int, i: int) -> float:
        """Compute the predicted rating for a compact user-item pair.

        :param u: Compact user index.
        :param i: Compact item index.
        :return: Predicted rating.
        """
        s = float(self._U[u] @ self._V[i])
        if self.config.use_biases:
            s += self._mu + self._bu[u] + self._bi[i]
        return float(np.clip(s, self.config.min_rating, self.config.max_rating))

    def _check_fitted(self) -> None:
        """Raise an error if the model has not been fitted yet."""
        if self._U is None:
            raise RuntimeError("PMF has not been fitted yet. Call fit() first.")


def _cold_start_status(u: int, i: int) -> ColdStartStatus:
    """Derive `ColdStartStatus` from compact indices (-1 signals unseen ID).

    :param u: Compact user index.
    :param i: Compact item index.
    :return: ColdStartStatus.
    """
    if u < 0 and i < 0:
        return ColdStartStatus.BOTH
    if u < 0:
        return ColdStartStatus.COLD_USER
    return ColdStartStatus.UNKNOWN_ITEM


def _print_header(conf: PMFConfig, n_users: int, n_items: int, n_obs: int) -> None:
    """Print a training header summarising the configuration and dataset.

    :param conf: PMFConfig instance with hyperparameters to display.
    :param n_users: Number of unique users.
    :param n_items: Number of unique items.
    :param n_obs: Number of observed ratings.
    """
    mode = (
        f"mini-batch SGD  (batch_size={conf.batch_size:,})"
        if conf.batch_size
        else "full-batch gradient descent"
    )
    print(f"\n{'='*64}")
    print(f"  PMF  |  {n_users:,} users  ·  {n_items:,} items  ·  {n_obs:,} ratings")
    print(
        f"  K={conf.n_factors}  lr={conf.lr}  "
        f"λ_U={conf.reg_user}  λ_V={conf.reg_item}  biases={conf.use_biases}"
    )
    print(f"  {mode}  |  tol={conf.tol}  max_epochs={conf.n_epochs}")
    print(f"{'='*64}")


# Some predefined configurations
PMF_CONFIGS: dict[str, PMFConfig] = {
    # Sensible default: full-batch GD, K=10, biases enabled
    "default": PMFConfig(),
    # Larger latent space for richer datasets
    "deep": PMFConfig(
        n_factors=50, n_epochs=200, lr=0.002, reg_user=0.05, reg_item=0.05
    ),
    # Mini-batch SGD: faster per-epoch, better for very large datasets
    "sgd": PMFConfig(n_factors=20, n_epochs=50, lr=0.01, batch_size=2_048),
    # No biases: pure PMF as in the original Mnih & Salakhutdinov (2007) paper
    "pure_pmf": PMFConfig(use_biases=False, n_factors=10, n_epochs=100),
}


class SVDPlusPlus(PMF):
    """SVD++: biased MF augmented with implicit feedback (Koren, 2008).

    Extends the biased-MF MAP estimate by adding a second set of *implicit* item factors Y that capture *which*
    items a user has rated, irrespective of the rating values.

    The key structural advantage over plain biased MF: even if user u has no explicit rating for item i,
    the implicit term shifts p_u toward regions of the latent space popular among items the user *has* engaged with.

    :param config: Shared `PMFConfig`; `reg_item` is reused as λ_Y for Y.
    """

    def __init__(self, config: PMFConfig | None = None) -> None:
        super().__init__(config)
        self._Y: np.ndarray | None = None  # (n_items, n_factors)  implicit factors
        self._impl_matrix: sp.csr_matrix | None = (
            None  # (n_users, n_items)  1/√|N(u)| mask
        )

    def fit(self, train_data: np.ndarray) -> SVDPlusPlus:
        """Fit SVD++ on training triples.

        Extends `PMF.fit` by additionally initialising Y and the pre-normalised implicit-feedback sparse matrix.

        :param train_data: Array of shape (n, 3) with columns (user_id, item_id, rating).
        :return: `self` for method chaining.
        """
        conf = self.config
        rng = np.random.default_rng(conf.seed)

        users = train_data[:, 0].astype(int)
        items = train_data[:, 1].astype(int)
        ratings = train_data[:, 2].astype(np.float64)

        unique_users = np.unique(users)
        unique_items = np.unique(items)
        self._user_idx = {int(uid): i for i, uid in enumerate(unique_users)}
        self._item_idx = {int(iid): j for j, iid in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)
        n_obs = len(ratings)

        u_idx = np.array([self._user_idx[u] for u in users], dtype=np.int32)
        i_idx = np.array([self._item_idx[i] for i in items], dtype=np.int32)

        self._mu = float(ratings.mean())

        self._U = (
            rng.standard_normal((n_users, conf.n_factors)).astype(np.float64)
            * conf.init_std
        )
        self._V = (
            rng.standard_normal((n_items, conf.n_factors)).astype(np.float64)
            * conf.init_std
        )
        self._Y = (
            rng.standard_normal((n_items, conf.n_factors)).astype(np.float64)
            * conf.init_std
        )

        self._bu = np.zeros(n_users, dtype=np.float64)
        self._bi = np.zeros(n_items, dtype=np.float64)

        self._impl_matrix = _build_impl_matrix(u_idx, i_idx, n_users, n_items)

        self._fallback = MeanFallback()
        self._fallback.setup(train_data)

        self.loss_history = []
        prev_loss = float("inf")

        _print_header_svdpp(conf, n_users, n_items, n_obs)

        for epoch in range(1, conf.n_epochs + 1):
            t0 = time.perf_counter()

            if conf.batch_size is None:
                self._gradient_step(u_idx, i_idx, ratings)
            else:
                order = rng.permutation(n_obs)
                for start in range(0, n_obs, conf.batch_size):
                    sl = slice(start, start + conf.batch_size)
                    self._gradient_step(
                        u_idx[order[sl]], i_idx[order[sl]], ratings[order[sl]]
                    )

            loss = self._compute_loss(u_idx, i_idx, ratings)
            self.loss_history.append(loss)

            delta = abs(prev_loss - loss)
            elapsed = time.perf_counter() - t0
            print(
                f"  Epoch {epoch:>4}/{conf.n_epochs} | loss={loss:12.4f} | Δloss={delta:.6f} | {elapsed:.2f}s"
            )

            if epoch > 1 and delta < conf.tol:
                print(f"\n  Early stopping - Δloss={delta:.2e} < tol={conf.tol:.2e}")
                break

            prev_loss = loss

        print(f"{'='*64}\n")
        return self

    def _gradient_step(
        self,
        u_idx: np.ndarray,
        i_idx: np.ndarray,
        ratings: np.ndarray,
    ) -> None:
        """One vectorised simultaneous gradient step for SVD++.

        :param u_idx: Array of shape (batch,) with compact user indices.
        :param i_idx: Array of shape (batch,) with compact item indices.
        :param ratings: Array of shape (batch,) with true ratings.
        """
        conf = self.config
        lr = conf.lr

        unique_u, u_inv = np.unique(u_idx, return_inverse=True)
        user_impl = self._impl_matrix[unique_u] @ self._Y  # (n_unique_u, K)
        pu_prime_unique = self._U[unique_u] + user_impl  # (n_unique_u, K)
        pu_prime = pu_prime_unique[u_inv]  # (batch, K)

        # Errors
        scores = np.einsum("nd,nd->n", pu_prime, self._V[i_idx])
        if conf.use_biases:
            scores += self._mu + self._bu[u_idx] + self._bi[i_idx]
        errors = ratings - scores  # (batch,)
        err_col = errors[:, None]  # (batch, 1)

        # Snapshot factor rows before any in-place modification
        V_snap = self._V[i_idx].copy()  # (batch, K)  - old q_i per sample
        pu_snap = pu_prime.copy()  # (batch, K)  - old pu_prime per sample

        u_counts = np.bincount(u_idx, minlength=len(self._U)).astype(np.float64)
        i_counts = np.bincount(i_idx, minlength=len(self._V)).astype(np.float64)

        # Update U (p_u)
        delta_U = np.zeros_like(self._U)
        np.add.at(delta_U, u_idx, err_col * V_snap)
        delta_U[unique_u] /= u_counts[unique_u, np.newaxis]
        self._U[unique_u] += lr * (
            delta_U[unique_u] - conf.reg_user * self._U[unique_u]
        )

        # Update V (q_i)
        unique_i = np.unique(i_idx)
        delta_V = np.zeros_like(self._V)
        np.add.at(delta_V, i_idx, err_col * pu_snap)
        delta_V[unique_i] /= i_counts[unique_i, np.newaxis]
        self._V[unique_i] += lr * (
            delta_V[unique_i] - conf.reg_item * self._V[unique_i]
        )

        # Update Y (implicit factors y_j)
        agg_e_q = np.zeros((len(unique_u), conf.n_factors), dtype=np.float64)
        np.add.at(agg_e_q, u_inv, err_col * V_snap)  # (n_unique_u, K)

        u_local_counts = np.bincount(u_inv, minlength=len(unique_u)).astype(np.float64)
        agg_e_q /= u_local_counts[:, np.newaxis]

        # Compute implicit gradient for each item
        impl_sub = self._impl_matrix[unique_u]
        delta_Y = np.asarray(impl_sub.T @ agg_e_q)  # (n_items, K)

        # Only regularise items that carry a nonzero implicit gradient in this batch
        impl_items = np.unique(impl_sub.indices)
        self._Y[impl_items] += lr * (
            delta_Y[impl_items] - conf.reg_item * self._Y[impl_items]
        )

        # Update biases
        if conf.use_biases:
            delta_bu = np.zeros_like(self._bu)
            np.add.at(delta_bu, u_idx, errors)
            delta_bu[unique_u] /= u_counts[unique_u]
            self._bu[unique_u] += lr * (
                delta_bu[unique_u] - conf.reg_bias * self._bu[unique_u]
            )

            delta_bi = np.zeros_like(self._bi)
            np.add.at(delta_bi, i_idx, errors)
            delta_bi[unique_i] /= i_counts[unique_i]
            self._bi[unique_i] += lr * (
                delta_bi[unique_i] - conf.reg_bias * self._bi[unique_i]
            )

    def _compute_loss(
        self,
        u_idx: np.ndarray,
        i_idx: np.ndarray,
        ratings: np.ndarray,
    ) -> float:
        """Full regularised SVD++ loss (includes ‖Y‖² penalty).

        :param u_idx: Array of shape (batch,) with compact user indices.
        :param i_idx: Array of shape (batch,) with compact item indices.
        :param ratings: Array of shape (batch,) with true ratings.
        :return: Scalar loss value.
        """
        conf = self.config

        unique_u, u_inv = np.unique(u_idx, return_inverse=True)
        user_impl = self._impl_matrix[unique_u] @ self._Y
        pu_prime = self._U[unique_u][u_inv] + user_impl[u_inv]

        scores = np.einsum("nd,nd->n", pu_prime, self._V[i_idx])
        if conf.use_biases:
            scores += self._mu + self._bu[u_idx] + self._bi[i_idx]

        errors = ratings - scores
        loss = 0.5 * float(np.dot(errors, errors))
        loss += 0.5 * conf.reg_user * float(np.sum(self._U**2))
        loss += 0.5 * conf.reg_item * float(np.sum(self._V**2))
        loss += 0.5 * conf.reg_item * float(np.sum(self._Y**2))  # λ_Y = λ_V
        if conf.use_biases:
            loss += (
                0.5
                * conf.reg_bias
                * (
                    float(np.dot(self._bu, self._bu))
                    + float(np.dot(self._bi, self._bi))
                )
            )
        return loss

    def predict_test(
        self,
        test_data: np.ndarray,
        cold_start_handler: ColdStartHandler | None = None,
    ) -> np.ndarray:
        """Generate predictions for the full test set.

        For known users, augmented factors pu_prime are precomputed once for all training users in
        a single sparse matmul before the per-test loop.

        :param test_data: Array of shape (n_test, 3) - (id, user_id, item_id).
        :param cold_start_handler: Fallback for cold-start cases.
        :return: Array of shape (n_test, 2) - (id, predicted_rating).
        """
        self._check_fitted()
        handler = (
            cold_start_handler if cold_start_handler is not None else self._fallback
        )

        test_ids = test_data[:, 0].astype(int)
        test_users = test_data[:, 1].astype(int)
        test_items = test_data[:, 2].astype(int)

        u_compact = np.array(
            [self._user_idx.get(int(u), -1) for u in test_users], dtype=np.int32
        )
        i_compact = np.array(
            [self._item_idx.get(int(i), -1) for i in test_items], dtype=np.int32
        )

        ok_mask = (u_compact >= 0) & (i_compact >= 0)
        predictions = np.empty(len(test_data), dtype=np.float64)

        if ok_mask.any():
            unique_u_ok = np.unique(u_compact[ok_mask])
            user_impl = self._impl_matrix[unique_u_ok] @ self._Y  # (n_distinct, K)
            pu_prime_map = self._U[unique_u_ok] + user_impl  # (n_distinct, K)

            # Map compact user index -> row in pu_prime_map
            u_to_row = np.full(self._U.shape[0], -1, dtype=np.int32)
            u_to_row[unique_u_ok] = np.arange(len(unique_u_ok), dtype=np.int32)

            u_ok = u_compact[ok_mask]
            i_ok = i_compact[ok_mask]
            pu = pu_prime_map[u_to_row[u_ok]]  # (n_ok, K)
            qi = self._V[i_ok]  # (n_ok, K)

            scores = np.einsum("nd,nd->n", pu, qi)
            if self.config.use_biases:
                scores += self._mu + self._bu[u_ok] + self._bi[i_ok]
            predictions[ok_mask] = np.clip(
                scores, self.config.min_rating, self.config.max_rating
            )

        for idx in np.where(~ok_mask)[0]:
            uid, iid = int(test_users[idx]), int(test_items[idx])
            status = _cold_start_status(u_compact[idx], i_compact[idx])
            predictions[idx] = handler.predict(uid, iid, status)

        return np.column_stack([test_ids, predictions])

    def predict_one(self, uid: int, iid: int) -> float:
        self._check_fitted()
        u = self._user_idx.get(int(uid))
        i = self._item_idx.get(int(iid))
        if u is None or i is None:
            return self._mu
        impl = np.asarray(self._impl_matrix[u] @ self._Y).ravel()  # (K,)
        pu_prime = self._U[u] + impl
        s = float(pu_prime @ self._V[i])
        if self.config.use_biases:
            s += self._mu + self._bu[u] + self._bi[i]
        return float(np.clip(s, self.config.min_rating, self.config.max_rating))


def _build_impl_matrix(
    u_idx: np.ndarray,
    i_idx: np.ndarray,
    n_users: int,
    n_items: int,
) -> sp.csr_matrix:
    """Build the pre-normalised implicit-feedback sparse matrix.

    :param u_idx: Compact user indices of all training triples.
    :param i_idx: Compact item indices of all training triples.
    :param n_users: Total number of unique users.
    :param n_items: Total number of unique items.
    :return: CSR sparse matrix of shape (n_users, n_items).
    """
    # Binary CSR (1 per rated pair, duplicates collapsed by summing -> still 1 if no repeats)
    binary = sp.csr_matrix(
        (np.ones(len(u_idx), dtype=np.float64), (u_idx, i_idx)),
        shape=(n_users, n_items),
    )
    binary.sum_duplicates()
    binary.data[:] = 1.0  # guard against any accumulated duplicates

    counts = np.diff(binary.indptr).astype(np.float64)  # |N(u)| per user
    inv_sqrt = np.where(counts > 0, 1.0 / np.sqrt(counts), 0.0)
    return sp.diags(inv_sqrt, format="csr") @ binary  # (n_users, n_items) normalised


def _print_header_svdpp(
    conf: PMFConfig, n_users: int, n_items: int, n_obs: int
) -> None:
    """Print a training header for SVD++ summarising the configuration and dataset.

    :param conf: `PMFConfig` instance with hyperparameters to display.
    :param n_users: Total number of unique users.
    :param n_items: Total number of unique items.
    :param n_obs: Total number of ratings.
    """
    mode = (
        f"mini-batch SGD  (batch_size={conf.batch_size:,})"
        if conf.batch_size
        else "full-batch gradient descent"
    )
    print(f"\n{'='*64}")
    print(f"  SVD++  |  {n_users:,} users  ·  {n_items:,} items  ·  {n_obs:,} ratings")
    print(
        f"  K={conf.n_factors}  lr={conf.lr}  "
        f"λ_U={conf.reg_user}  λ_V={conf.reg_item}  λ_Y={conf.reg_item}  biases={conf.use_biases}"
    )
    print(f"  {mode}  |  tol={conf.tol}  max_epochs={conf.n_epochs}")
    print(f"{'='*64}")


SVDPP_CONFIGS: dict[str, PMFConfig] = {
    # Good starting point: the implicit term benefits from a slightly higher lr
    "default": PMFConfig(
        n_factors=10, n_epochs=100, lr=0.007, reg_user=0.02, reg_item=0.02
    ),
    # Larger latent space - more expressive implicit signal
    "deep": PMFConfig(
        n_factors=50, n_epochs=150, lr=0.003, reg_user=0.05, reg_item=0.05
    ),
    # Mini-batch - recommended for large datasets; implicit term computed per-batch
    "sgd": PMFConfig(
        n_factors=20,
        n_epochs=50,
        lr=0.015,
        reg_user=0.02,
        reg_item=0.02,
        batch_size=2_048,
    ),
}
