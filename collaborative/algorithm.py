from __future__ import annotations

from enum import Enum, auto

import numpy as np
import scipy.sparse as sp

from .similarity import SIMILARITY_REGISTRY
from .strategy import STRATEGY_REGISTRY


class ColdStartStatus(Enum):
    OK = auto()  # user and item both present in train
    COLD_USER = auto()  # user not seen during training
    UNKNOWN_ITEM = auto()  # item not seen during training
    BOTH = auto()  # neither user nor item is in train


def diagnose_cold_start(
    train_data: np.ndarray,
    test_data: np.ndarray,
) -> tuple[dict[int, ColdStartStatus], dict[str, int]]:
    """Check every test case against the training data and classify it.

    :param train_data: Training data as a numpy array of shape (n_train, 3) with columns (user_id, item_id, rating).
    :param test_data: Test data as a numpy array of shape (n_test, 3) with columns (id, user_id, item_id).
    :return: A tuple containing a mapping of test case indices to their cold-start status, and a summary count of each status category.
    """
    train_users = set(train_data[:, 0].astype(int))
    train_items = set(train_data[:, 1].astype(int))

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

    :param summary: A dictionary with counts of each cold-start category, typically returned by `diagnose_cold_start`.
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
    """Build a CSR sparse user-item rating matrix from raw triples.

    :param data: A numpy array of shape (n_samples, 3) where each row is (user_id, item_id, rating).
    :return: A tuple containing the sparse user-item matrix, a mapping of user IDs to row indices, and a mapping of item IDs to column indices.
    """
    user_ids = np.unique(data[:, 0].astype(int))
    item_ids = np.unique(data[:, 1].astype(int))
    user_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_idx = {iid: j for j, iid in enumerate(item_ids)}

    rows = np.fromiter(
        (user_idx[int(u)] for u in data[:, 0]), dtype=np.int32, count=len(data)
    )
    cols = np.fromiter(
        (item_idx[int(i)] for i in data[:, 1]), dtype=np.int32, count=len(data)
    )
    ratings = data[:, 2].astype(np.float64)

    matrix = sp.csr_matrix(
        (ratings, (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float64,
    )
    return matrix, user_idx, item_idx


def predict_knn(
    train_data: np.ndarray,
    test_data: np.ndarray,
    *,
    k: int = 5,
    similarity: str = "euclidean",
    strategy: str = "w_mean",
    batch_size: int = 4_096,
    fallback: float = 3.0,
    cold_start_map: dict[int, ColdStartStatus] | None = None,
) -> np.ndarray:
    """_summary_

    :param train_data: _description_
    :param test_data: _description_
    :param k: _description_, defaults to 5
    :param similarity: _description_, defaults to "euclidean"
    :param strategy: _description_, defaults to "w_mean"
    :param batch_size: _description_, defaults to 4_096
    :param fallback: _description_, defaults to 3.0
    :param cold_start_map: _description_, defaults to None
    :return: _description_
    """
    sim_fn = SIMILARITY_REGISTRY[similarity]
    pred_fn = STRATEGY_REGISTRY[strategy]

    # Training structures
    train_matrix, user_idx, item_idx = build_user_item_matrix(train_data)
    train_csc = train_matrix.tocsc()

    test_ids = test_data[:, 0].astype(int)
    test_user_ids = test_data[:, 1].astype(int)
    test_item_ids = test_data[:, 2].astype(int)

    predictions = np.full(len(test_data), fallback, dtype=np.float64)

    # Batch over unique test users
    unique_test_users = np.unique(test_user_ids)
    n_batches = int(np.ceil(len(unique_test_users) / batch_size))

    print(
        f"Predicting {len(test_data):,} ratings  |  "
        f"K={k}  |  similarity={similarity}  |  {n_batches} batch(es)"
    )

    for batch_num, batch_start in enumerate(
        range(0, len(unique_test_users), batch_size), start=1
    ):
        batch_users = unique_test_users[batch_start : batch_start + batch_size]
        known_users = [u for u in batch_users if u in user_idx]
        unknown_count = len(batch_users) - len(known_users)

        if not known_users:
            print(f"  [{batch_num}/{n_batches}] skipped — no known users in batch")
            continue

        # Similarity block: (B, n_train_users)
        batch_row_idx = [user_idx[u] for u in known_users]
        query_matrix = train_matrix[batch_row_idx, :]
        sim_matrix = sim_fn.compute(query_matrix, train_matrix)  # (B, N)
        user_to_row = {u: i for i, u in enumerate(known_users)}

        # Predict each test case in this batch
        batch_indices = np.where(np.isin(test_user_ids, batch_users))[0]

        for idx in batch_indices:
            uid = test_user_ids[idx]
            iid = test_item_ids[idx]

            # Skip cases identified as cold-start (fallback already set)
            if cold_start_map is not None:
                if cold_start_map[idx] is not ColdStartStatus.OK:
                    continue
            elif uid not in user_to_row or iid not in item_idx:
                # Fallback when no pre-computed map is available
                continue

            sim_row = sim_matrix[user_to_row[uid]]  # (N,) — do NOT modify in-place
            own_idx = user_idx[uid]
            col_idx = item_idx[iid]

            # Neighbours = training users who rated this item, minus self
            item_col = train_csc.getcol(col_idx)  # sparse (N, 1)
            rater_rows = item_col.indices  # row indices of raters
            rater_ratings = item_col.data  # corresponding ratings

            # Exclude the query user (present in train by construction)
            mask = rater_rows != own_idx
            rater_rows = rater_rows[mask]
            rater_ratings = rater_ratings[mask]

            if len(rater_rows) == 0:
                continue  # no neighbours for this item → fallback

            rater_sims = sim_row[rater_rows]  # (n_raters,)
            actual_k = min(k, len(rater_rows))

            # argpartition is O(n) vs O(n log n) for full sort
            if actual_k < len(rater_rows):
                top_pos = np.argpartition(rater_sims, -actual_k)[-actual_k:]
            else:
                top_pos = np.arange(len(rater_rows))

            predictions[idx] = pred_fn.predict(
                rater_sims[top_pos],
                rater_ratings[top_pos],
            )

        print(
            f"  [{batch_num}/{n_batches}] {len(batch_users):,} users  "
            f"({len(known_users):,} known, {unknown_count:,} unknown)"
        )

    return np.column_stack([test_ids, predictions])
