from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .cold_start import ColdStartHandler, ColdStartStatus, MeanFallback, StaticFallback
from .similarity import SIMILARITY_REGISTRY
from .strategy import STRATEGY_REGISTRY


def diagnose_cold_start(
    train_data: np.ndarray,
    test_data: np.ndarray,
) -> tuple[dict[int, ColdStartStatus], dict[str, int]]:
    """Classify every test case by its cold-start status.

    :param train_data: (n_train, 3) array with columns (user_id, item_id, rating).
    :param test_data:  (n_test,  3) array with columns (id, user_id, item_id).
    :return: Tuple of (status_map, summary) where status_map maps test-row index to `ColdStartStatus` and summary contains counts per category.
    """
    train_users = set(train_data[:, 0].astype(int).tolist())
    train_items = set(train_data[:, 1].astype(int).tolist())

    status_map: dict[int, ColdStartStatus] = {}
    summary = {"ok": 0, "cold_user": 0, "unknown_item": 0, "both": 0}

    for idx in range(len(test_data)):
        uid = int(test_data[idx, 1])
        iid = int(test_data[idx, 2])
        has_user = uid in train_users
        has_item = iid in train_items

        if has_user and has_item:
            status = ColdStartStatus.OK
            summary["ok"] += 1
        elif not has_user and has_item:
            status = ColdStartStatus.COLD_USER
            summary["cold_user"] += 1
        elif has_user and not has_item:
            status = ColdStartStatus.UNKNOWN_ITEM
            summary["unknown_item"] += 1
        else:
            status = ColdStartStatus.BOTH
            summary["both"] += 1

        status_map[idx] = status

    return status_map, summary


def print_cold_start_report(summary: dict[str, int]) -> None:
    """Print a human-readable cold-start summary.

    :param summary: Dict with counts per category, as returned by `diagnose_cold_start`.
    """
    total = sum(summary.values())
    print(" Cold-start diagnosis ".center(56, "="))
    for key, label in [
        ("ok", "OK (user + item in train)"),
        ("cold_user", "Cold user  (user not in train)"),
        ("unknown_item", "Unknown item (item not in train)"),
        ("both", "Both missing"),
    ]:
        n = summary[key]
        pct = 100.0 * n / total if total else 0.0
        print(f"  {label:<35} {n:>7,}  ({pct:5.1f} %)")
    print(f"  {'Total':<35} {total:>7,}")
    print("=" * 56)


def build_user_item_matrix(
    data: np.ndarray,
) -> tuple[sp.csr_matrix, dict[int, int], dict[int, int]]:
    """Build a sparse CSR user-item rating matrix from raw (user, item, rating) triples.

    :param data: (n_samples, 3) array with columns (user_id, item_id, rating).
    :return: Tuple of (matrix, user_idx, item_idx) where user_idx / item_idx map
             original IDs to matrix row / column indices.
    """
    users = data[:, 0].astype(int)
    items = data[:, 1].astype(int)

    user_ids = np.unique(users)
    item_ids = np.unique(items)
    user_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_idx = {iid: j for j, iid in enumerate(item_ids)}

    rows = np.array([user_idx[u] for u in users], dtype=np.int32)
    cols = np.array([item_idx[i] for i in items], dtype=np.int32)
    ratings = data[:, 2].astype(np.float64)

    matrix = sp.csr_matrix(
        (ratings, (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float64,
    )
    return matrix, user_idx, item_idx


def compute_user_stats(
    train_matrix: sp.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-user mean and standard deviation over rated items only.

    :param train_matrix: Sparse (N_users, N_items) user-item matrix.
    :return: Tuple (means, stds) of shape (N_users,) each.
    """
    counts = np.diff(train_matrix.indptr).astype(np.float64)  # (N,)
    sums = np.asarray(train_matrix.sum(axis=1)).ravel()  # (N,)
    means = np.where(counts > 0, sums / counts, 0.0)

    sq_sums = np.asarray(train_matrix.power(2).sum(axis=1)).ravel()
    mean_sq = np.where(counts > 0, sq_sums / counts, 0.0)
    stds = np.sqrt(np.clip(mean_sq - means**2, 0.0, None))

    return means, stds


@dataclass
class NeighbourRecord:
    """Pre-computed neighbour data for a single *valid* (non-cold-start) case.

    Used by `predict_from_records` to sweep different K values and strategies
    without repeating the expensive similarity computation.
    """

    val_idx: int  # index in val_data
    true_rating: float  # ground-truth label (available in CV splits)
    rater_sims: np.ndarray  # shape (≤ k_max,) - similarities of item raters
    rater_ratings: np.ndarray  # shape (≤ k_max,) - their ratings for the item
    query_mean: float  # mean rating of the query user
    query_std: float  # std  of ratings of the query user
    neighbour_means: np.ndarray  # shape (≤ k_max,) - means of rater users
    neighbour_stds: np.ndarray  # shape (≤ k_max,) - stds  of rater users


def precompute_neighbours(
    train_data: np.ndarray,
    val_data: np.ndarray,
    *,
    similarity: str = "euclidean",
    k_max: int = 50,
    batch_size: int = 4_096,
    cold_start_handler: ColdStartHandler | None = None,
) -> tuple[list[NeighbourRecord], np.ndarray, np.ndarray]:
    """Pre-compute top-`k_max` neighbour data for each non-cold-start val case.

    This separates the costly similarity computation from the final aggregation,
    letting the cross-validation script sweep over different K values and
    strategies without repeating matrix operations.

    :param train_data: (n, 3) array - (user_id, item_id, rating).
    :param val_data: (m, 3) array - (user_id, item_id, rating). Rating column is used as ground truth; it is *not* used during prediction.
    :param similarity: Name of the similarity function (see SIMILARITY_REGISTRY).
    :param k_max: Maximum number of neighbours to store per case.
    :param batch_size: Number of unique users processed per similarity batch.
    :param cold_start_handler: Handles fallback predictions for cold-start cases. Defaults to `MeanFallback`.
    :return: Tuple of:

        - `records`: list of `NeighbourRecord` for OK cases
        - `cold_preds`: fallback predictions for cold-start cases (float array)
        - `cold_true`: ground-truth ratings for those same cases
    """
    if cold_start_handler is None:
        cold_start_handler = MeanFallback()
        cold_start_handler.setup(train_data)

    sim_fn = SIMILARITY_REGISTRY[similarity]

    train_matrix, user_idx, item_idx = build_user_item_matrix(train_data)
    train_csc = train_matrix.tocsc()
    user_means, user_stds = compute_user_stats(train_matrix)

    val_users = val_data[:, 0].astype(int)
    val_items = val_data[:, 1].astype(int)
    val_ratings = val_data[:, 2].astype(np.float64)

    # Classify cold-start cases
    cold_map, _ = diagnose_cold_start(train_data, val_data)

    records: list[NeighbourRecord] = []
    cold_preds: list[float] = []
    cold_true: list[float] = []

    unique_val_users = np.unique(val_users)
    n_batches = int(np.ceil(len(unique_val_users) / batch_size))

    print(
        f"Pre-computing neighbours for {len(val_data):,} val cases  |  "
        f"similarity={similarity}  |  k_max={k_max}  |  {n_batches} batch(es)"
    )

    for batch_num, batch_start in enumerate(
        range(0, len(unique_val_users), batch_size), start=1
    ):
        batch_users = unique_val_users[batch_start : batch_start + batch_size]
        known_users = [u for u in batch_users if u in user_idx]

        if not known_users:
            print(f"  [{batch_num}/{n_batches}] skipped - no known users in batch")
            continue

        batch_row_idx = [user_idx[u] for u in known_users]
        query_matrix = train_matrix[batch_row_idx, :]
        sim_matrix = sim_fn(query_matrix, train_matrix)  # (B, N)
        user_to_row = {u: i for i, u in enumerate(known_users)}

        batch_indices = np.where(np.isin(val_users, batch_users))[0]

        for idx in batch_indices:
            uid = int(val_users[idx])
            iid = int(val_items[idx])
            status = cold_map[idx]

            if status is not ColdStartStatus.OK:
                cold_preds.append(cold_start_handler.predict(uid, iid, status))
                cold_true.append(float(val_ratings[idx]))
                continue

            if uid not in user_to_row or iid not in item_idx:
                cold_preds.append(
                    cold_start_handler.predict(uid, iid, ColdStartStatus.BOTH)
                )
                cold_true.append(float(val_ratings[idx]))
                continue

            sim_row = sim_matrix[user_to_row[uid]]
            own_idx = user_idx[uid]
            col_idx = item_idx[iid]

            item_col = train_csc.getcol(col_idx)
            rater_rows = item_col.indices.copy()
            rater_ratings = item_col.data.copy()

            # Exclude the query user itself
            mask = rater_rows != own_idx
            rater_rows = rater_rows[mask]
            rater_ratings = rater_ratings[mask]

            if len(rater_rows) == 0:
                cold_preds.append(
                    cold_start_handler.predict(uid, iid, ColdStartStatus.UNKNOWN_ITEM)
                )
                cold_true.append(float(val_ratings[idx]))
                continue

            rater_sims = sim_row[rater_rows]
            actual_k = min(k_max, len(rater_rows))

            if actual_k < len(rater_rows):
                top_pos = np.argpartition(rater_sims, -actual_k)[-actual_k:]
            else:
                top_pos = np.arange(len(rater_rows))

            # Sort top neighbours by descending similarity for consistent ordering
            top_pos = top_pos[np.argsort(-rater_sims[top_pos])]

            records.append(
                NeighbourRecord(
                    val_idx=int(idx),
                    true_rating=float(val_ratings[idx]),
                    rater_sims=rater_sims[top_pos],
                    rater_ratings=rater_ratings[top_pos],
                    query_mean=float(user_means[own_idx]),
                    query_std=float(user_stds[own_idx]),
                    neighbour_means=user_means[rater_rows[top_pos]],
                    neighbour_stds=user_stds[rater_rows[top_pos]],
                )
            )

        print(f"  [{batch_num}/{n_batches}] {len(batch_users):,} users processed")

    return records, np.array(cold_preds), np.array(cold_true)


def predict_from_records(
    records: list[NeighbourRecord],
    cold_preds: np.ndarray,
    *,
    k: int,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Make predictions from pre-computed neighbour records (used in cross-validation).

    :param records:    List of `NeighbourRecord` from `precompute_neighbours`.
    :param cold_preds: Fallback predictions for cold-start cases (already computed).
    :param k:          Number of neighbours to use (must be ≤ k_max used during pre-computation).
    :param strategy:   Name of the aggregation strategy (see STRATEGY_REGISTRY).
    :return: Tuple of (y_pred, y_true) arrays covering all (OK + cold-start) cases.
    """
    pred_fn = STRATEGY_REGISTRY[strategy]

    ok_preds = np.empty(len(records), dtype=np.float64)
    ok_true = np.empty(len(records), dtype=np.float64)

    for i, rec in enumerate(records):
        actual_k = min(k, len(rec.rater_sims))
        ok_preds[i] = pred_fn.predict(
            rec.rater_sims[:actual_k],
            rec.rater_ratings[:actual_k],
            query_user_mean=rec.query_mean,
            query_user_std=rec.query_std,
            neighbour_means=rec.neighbour_means[:actual_k],
            neighbour_stds=rec.neighbour_stds[:actual_k],
        )
        ok_true[i] = rec.true_rating

    y_pred = np.concatenate([ok_preds, cold_preds])
    y_true = np.concatenate([ok_true, np.array([])])  # cold_true lives in caller

    return ok_preds, ok_true


def predict_knn(
    train_data: np.ndarray,
    test_data: np.ndarray,
    *,
    k: int = 5,
    similarity: str = "euclidean",
    strategy: str = "w_mean",
    batch_size: int = 4_096,
    cold_start_handler: ColdStartHandler | None = None,
    cold_start_map: dict[int, ColdStartStatus] | None = None,
) -> np.ndarray:
    """Run KNN collaborative filtering and return predictions for all test cases.

    :param train_data: (n_train, 3) array - (user_id, item_id, rating).
    :param test_data: (n_test,  3) array - (id, user_id, item_id).
    :param k: Number of nearest neighbours.
    :param similarity: Similarity function key (see `SIMILARITY_REGISTRY`).
    :param strategy: Aggregation strategy key (see `STRATEGY_REGISTRY`).
    :param batch_size: Unique users per similarity-computation batch.
    :param cold_start_handler: Fallback strategy for cold-start cases. Defaults to `MeanFallback`.
    :param cold_start_map: Pre-computed cold-start status map from `diagnose_cold_start`; computed internally if None.
    :return: (n_test, 2) array with columns (test_id, predicted_rating).
    """
    if cold_start_handler is None:
        cold_start_handler = MeanFallback()
        cold_start_handler.setup(train_data)

    sim_fn = SIMILARITY_REGISTRY[similarity]
    pred_fn = STRATEGY_REGISTRY[strategy]

    train_matrix, user_idx, item_idx = build_user_item_matrix(train_data)
    train_csc = train_matrix.tocsc()
    user_means, user_stds = compute_user_stats(train_matrix)

    test_ids = test_data[:, 0].astype(int)
    test_user_ids = test_data[:, 1].astype(int)
    test_item_ids = test_data[:, 2].astype(int)

    # Initialise all predictions to the cold-start handler's global mean
    predictions = np.full(
        len(test_data), cold_start_handler.predict(0, 0, ColdStartStatus.BOTH)
    )

    if cold_start_map is None:
        cold_start_map, _ = diagnose_cold_start(train_data, test_data)

    # Apply cold-start fallbacks immediately
    for idx, status in cold_start_map.items():
        if status is not ColdStartStatus.OK:
            uid = int(test_user_ids[idx])
            iid = int(test_item_ids[idx])
            predictions[idx] = cold_start_handler.predict(uid, iid, status)

    unique_test_users = np.unique(test_user_ids)
    n_batches = int(np.ceil(len(unique_test_users) / batch_size))

    print(
        f"Predicting {len(test_data):,} ratings  |  "
        f"K={k}  |  similarity={similarity}  |  strategy={strategy}  |  {n_batches} batch(es)"
    )

    for batch_num, batch_start in enumerate(
        range(0, len(unique_test_users), batch_size), start=1
    ):
        batch_users = unique_test_users[batch_start : batch_start + batch_size]
        known_users = [u for u in batch_users if u in user_idx]
        unknown_count = len(batch_users) - len(known_users)

        if not known_users:
            print(f"  [{batch_num}/{n_batches}] skipped - no known users in batch")
            continue

        batch_row_idx = [user_idx[u] for u in known_users]
        query_matrix = train_matrix[batch_row_idx, :]
        sim_matrix = sim_fn(query_matrix, train_matrix)  # (B, N)
        user_to_row = {u: i for i, u in enumerate(known_users)}

        batch_indices = np.where(np.isin(test_user_ids, batch_users))[0]

        for idx in batch_indices:
            if cold_start_map[idx] is not ColdStartStatus.OK:
                continue  # already handled above

            uid = int(test_user_ids[idx])
            iid = int(test_item_ids[idx])

            if uid not in user_to_row or iid not in item_idx:
                continue

            sim_row = sim_matrix[user_to_row[uid]]  # (N,) - do NOT modify in-place
            own_idx = user_idx[uid]
            col_idx = item_idx[iid]

            item_col = train_csc.getcol(col_idx)
            rater_rows = item_col.indices
            rater_ratings = item_col.data

            # Exclude the query user (they are in train by construction)
            mask = rater_rows != own_idx
            rater_rows = rater_rows[mask]
            rater_ratings = rater_ratings[mask]

            if len(rater_rows) == 0:
                continue  # no neighbours -> keep fallback

            rater_sims = sim_row[rater_rows]
            actual_k = min(k, len(rater_rows))

            if actual_k < len(rater_rows):
                top_pos = np.argpartition(rater_sims, -actual_k)[-actual_k:]
            else:
                top_pos = np.arange(len(rater_rows))

            predictions[idx] = pred_fn(
                rater_sims[top_pos],
                rater_ratings[top_pos],
                query_user_mean=float(user_means[own_idx]),
                query_user_std=float(user_stds[own_idx]),
                neighbour_means=user_means[rater_rows[top_pos]],
                neighbour_stds=user_stds[rater_rows[top_pos]],
            )

        print(
            f"  [{batch_num}/{n_batches}] {len(batch_users):,} users  ({len(known_users):,} known, {unknown_count:,} unknown)"
        )

    return np.column_stack([test_ids, predictions])
