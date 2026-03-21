from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    :param y_true: Ground-truth ratings.
    :param y_pred: Predicted ratings (same shape as y_true).
    :return: MAE scalar.
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    :param y_true: Ground-truth ratings.
    :param y_pred: Predicted ratings (same shape as y_true).
    :return: RMSE scalar.
    """
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def coverage(y_pred: np.ndarray, fallback: float, tol: float = 1e-6) -> float:
    """Fraction of predictions that differ from the fallback value.

    Measures how often the model could produce a data-driven estimate vs.
    resorting to the cold-start fallback.

    :param y_pred: Predicted rating array.
    :param fallback: The cold-start fallback value used during prediction.
    :param tol: Absolute tolerance for equality comparison.
    :return: Value in [0, 1].
    """
    return float(np.mean(np.abs(np.asarray(y_pred) - fallback) > tol))


def _group_by_user(
    user_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group true and predicted ratings by user.

    :param user_ids: Array of user IDs.
    :param y_true: Array of ground-truth ratings.
    :param y_pred: Array of predicted ratings.
    :return: List of tuples containing grouped true and predicted ratings.
    """
    user_ids = np.asarray(user_ids)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    order = np.argsort(user_ids, kind="stable")
    uid_s = user_ids[order]
    true_s = y_true[order]
    pred_s = y_pred[order]

    groups: list[tuple[np.ndarray, np.ndarray]] = []
    starts = np.concatenate([[0], np.where(np.diff(uid_s))[0] + 1, [len(uid_s)]])
    for a, b in zip(starts[:-1], starts[1:]):
        groups.append((true_s[a:b], pred_s[a:b]))
    return groups


def mean_average_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    user_ids: np.ndarray,
    *,
    threshold: float = 7.0,
) -> float:
    """Mean Average Precision (MAP) for rating-prediction tasks.

    Items with `y_true >= threshold` are considered *relevant*.
    For each user the predicted scores rank the items; Average Precision is computed over that ranking and then averaged across users.

    Users with no relevant items in their test set are excluded from the mean.

    :param y_true: Ground-truth ratings.
    :param y_pred: Predicted ratings used for ranking.
    :param user_ids: User ID for each prediction (same length as y_true).
    :param threshold: Minimum true rating to be considered relevant.
    :return: MAP scalar.
    """
    groups = _group_by_user(user_ids, y_true, y_pred)
    aps: list[float] = []

    for true_r, pred_r in groups:
        relevant = (true_r >= threshold).astype(np.float64)
        n_rel = relevant.sum()
        if n_rel == 0:
            continue

        order = np.argsort(-pred_r)
        ranked = relevant[order]
        positions = np.arange(1, len(ranked) + 1, dtype=np.float64)
        cum_prec = np.cumsum(ranked) / positions
        ap = float(np.dot(cum_prec, ranked) / n_rel)
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def ndcg_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    user_ids: np.ndarray,
    *,
    k: int = 10,
) -> float:
    """Normalised Discounted Cumulative Gain at K (nDCG@K).

    Uses true ratings as *graded* relevance scores (not binary), so fractional rating differences are taken into account.
    Items are ranked by predicted score, the ideal ranking uses true scores.

    :param y_true: Ground-truth ratings (used as relevance).
    :param y_pred: Predicted ratings (used for ranking).
    :param user_ids: User ID for each prediction.
    :param k: Cut-off rank.
    :return: nDCG@K scalar averaged over users.
    """
    groups = _group_by_user(user_ids, y_true, y_pred)
    ndcgs: list[float] = []

    log_denom = np.log2(np.arange(2, k + 2, dtype=np.float64))  # log₂(2..k+1)

    for true_r, pred_r in groups:
        n = min(k, len(true_r))
        if n == 0:
            continue

        pred_order = np.argsort(-pred_r)[:n]
        ideal_order = np.argsort(-true_r)[:n]

        dcg = float(np.sum(true_r[pred_order] / log_denom[:n]))
        idcg = float(np.sum(true_r[ideal_order] / log_denom[:n]))

        if idcg > 0:
            ndcgs.append(dcg / idcg)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    user_ids: np.ndarray | None = None,
    *,
    fallback: float | None = None,
    map_threshold: float = 7.0,
    ndcg_k: int = 10,
) -> dict[str, float]:
    """Compute all available metrics and return them as a dict.

    Rating metrics are always computed.
    Ranking metrics require `user_ids`.
    Coverage requires `fallback`.

    :param y_true: Ground-truth ratings.
    :param y_pred: Predicted ratings.
    :param user_ids: Optional user ID array for ranking metrics.
    :param fallback: Optional fallback value for coverage computation.
    :param map_threshold: Relevance threshold for MAP.
    :param ndcg_k: Cut-off for nDCG.
    :return: Dict with metric names as keys and scalar values.
    """
    results: dict[str, float] = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }
    if fallback is not None:
        results["coverage"] = coverage(y_pred, fallback)
    if user_ids is not None:
        results["map"] = mean_average_precision(
            y_true, y_pred, user_ids, threshold=map_threshold
        )
        results[f"ndcg@{ndcg_k}"] = ndcg_at_k(y_true, y_pred, user_ids, k=ndcg_k)
    return results
