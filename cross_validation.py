from __future__ import annotations

import itertools
import os
import time
from typing import Any

import numpy as np

from collaborative import (
    diagnose_cold_start,
    precompute_neighbours,
    predict_from_records,
    print_cold_start_report,
)
from collaborative.cold_start import COLD_START_REGISTRY, MeanFallback
from collaborative.metrics import evaluate
from i_o import load_data, save_cv_results


def kfold_indices(
    n: int, n_folds: int, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return `n_folds` (train_idx, val_idx) pairs for a dataset of size `n`.

    :param n: Total number of samples.
    :param n_folds: Number of folds.
    :param seed: Random seed for shuffling.
    :return: List of (train_indices, val_indices) tuples.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    splits = np.array_split(order, n_folds)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(n_folds):
        val_idx = splits[fold_idx]
        train_idx = np.concatenate([splits[i] for i in range(n_folds) if i != fold_idx])
        folds.append((train_idx, val_idx))
    return folds


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def cross_validate(
    data: np.ndarray,
    *,
    n_folds: int = 5,
    k_values: list[int],
    similarities: list[str],
    strategies: list[str],
    k_max: int | None = None,
    batch_size: int = 4_096,
    cold_start_handler_name: str = "mean",
    map_threshold: float = 7.0,
    ndcg_k: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """K-fold cross-validation sweeping multiple (similarity, K, strategy) combos.

    Similarities are pre-computed once per (fold × similarity function).
    K and strategy sweeps re-use those pre-computed neighbour records at negligible cost.

    :param data: Full training dataset (n, 3) - (user, item, rating).
    :param n_folds: Number of CV folds.
    :param k_values: List of K values to evaluate.
    :param similarities: List of similarity function names to evaluate.
    :param strategies: List of strategy names to evaluate.
    :param k_max: Maximum K to pre-compute; defaults to max(k_values).
    :param batch_size: Users per similarity batch.
    :param cold_start_handler_name: Key in `COLD_START_REGISTRY`.
    :param map_threshold: True-rating threshold for MAP relevance.
    :param ndcg_k: Cut-off rank for nDCG.
    :param seed: Random seed for fold splitting.
    :param verbose: Print progress and interim results.
    :return: List of result dicts, one per (fold, similarity, K, strategy) combination.
    """
    if k_max is None:
        k_max = max(k_values)

    folds = kfold_indices(len(data), n_folds, seed=seed)
    results: list[dict[str, Any]] = []

    total_combos = n_folds * len(similarities) * len(k_values) * len(strategies)
    if verbose:
        print(f"\n{'='*60}")
        print(
            f"  Cross-validation: {n_folds} folds × {len(similarities)} sims × "
            f"{len(k_values)} K values × {len(strategies)} strategies"
        )
        print(f"  = {total_combos} total evaluations")
        print(f"{'='*60}\n")

    for fold_num, (train_idx, val_idx) in enumerate(folds, start=1):
        train_fold = data[train_idx]
        val_fold = data[val_idx]

        if verbose:
            print(f"\n{'─'*60}")
            print(
                f"  Fold {fold_num}/{n_folds}  "
                f"(train: {len(train_fold):,}, val: {len(val_fold):,})"
            )
            print(f"{'─'*60}")

            # Show cold-start stats for this fold
            _, cs_summary = diagnose_cold_start(
                train_fold,
                # val_fold has (user, item, rating); diagnose_cold_start expects (id, user, item) so we synthesise a fake id column
                np.column_stack([np.arange(len(val_fold)), val_fold[:, :2]]),
            )
            print_cold_start_report(cs_summary)

        for sim_name in similarities:
            t0 = time.perf_counter()

            # Build cold-start handler fresh per fold so stats reflect this fold's training data
            handler = COLD_START_REGISTRY[cold_start_handler_name]()
            handler.setup(train_fold)

            # Convert val_fold (user, item, rating) → (id, user, item) for diagnose
            val_as_test = np.column_stack(
                [
                    np.arange(len(val_fold), dtype=int),
                    val_fold[:, 0].astype(int),
                    val_fold[:, 1].astype(int),
                ]
            )

            # Pre-compute neighbours once per fold × similarity
            records, cold_preds, cold_true = precompute_neighbours(
                train_fold,
                val_fold,
                similarity=sim_name,
                k_max=k_max,
                batch_size=batch_size,
                cold_start_handler=handler,
            )

            elapsed_precompute = time.perf_counter() - t0
            if verbose:
                print(
                    f"  Precompute done  ({sim_name})  -  "
                    f"{len(records):,} OK cases, {len(cold_preds):,} cold-start  "
                    f"[{_fmt(elapsed_precompute)}]"
                )

            # Sweep K × strategy without recomputing similarities
            for k, strat_name in itertools.product(k_values, strategies):
                t1 = time.perf_counter()

                ok_preds, ok_true = predict_from_records(
                    records, cold_preds, k=k, strategy=strat_name
                )

                # Combine OK predictions and cold-start predictions for overall metrics
                y_pred_all = np.concatenate([ok_preds, cold_preds])
                y_true_all = np.concatenate([ok_true, cold_true])

                metrics_all = evaluate(y_pred_all, y_true_all)  # MAE, RMSE, no ranking

                # Ranking metrics on OK cases only (cold-start has no real neighbours)
                # Reconstruct user IDs for OK records
                ok_user_ids = (
                    np.array([int(val_fold[r.val_idx, 0]) for r in records])
                    if records
                    else np.array([], dtype=int)
                )

                metrics_ok = evaluate(
                    ok_preds,
                    ok_true,
                    user_ids=ok_user_ids if len(ok_user_ids) > 0 else None,
                    map_threshold=map_threshold,
                    ndcg_k=ndcg_k,
                )

                row: dict[str, Any] = {
                    "fold": fold_num,
                    "similarity": sim_name,
                    "k": k,
                    "strategy": strat_name,
                    # Overall (includes cold-start fallbacks)
                    "mae_all": metrics_all["mae"],
                    "rmse_all": metrics_all["rmse"],
                    # OK cases only
                    "mae_ok": metrics_ok["mae"],
                    "rmse_ok": metrics_ok["rmse"],
                    "map_ok": metrics_ok.get("map", float("nan")),
                    f"ndcg{ndcg_k}_ok": metrics_ok.get(f"ndcg@{ndcg_k}", float("nan")),
                    "n_ok": len(ok_preds),
                    "n_cold": len(cold_preds),
                    "elapsed_s": round(time.perf_counter() - t1, 3),
                }
                results.append(row)

                if verbose:
                    print(
                        f"    K={k:<4}  strat={strat_name:<15} | "
                        f"MAE(all)={row['mae_all']:.4f} | "
                        f"MAE(ok)={row['mae_ok']:.4f} | "
                        f"RMSE(ok)={row['rmse_ok']:.4f} | "
                        f"MAP={row['map_ok']:.4f}"
                    )

    return results


def aggregate_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Average metrics across folds for each (similarity, K, strategy) combination.

    :param results: Raw per-fold results from `cross_validate`.
    :return: Aggregated results, one dict per unique (similarity, K, strategy).
    """
    from collections import defaultdict

    metric_keys = ["mae_all", "rmse_all", "mae_ok", "rmse_ok", "map_ok"]
    for r in results:
        metric_keys += [k for k in r if k.startswith("ndcg") and k not in metric_keys]
    metric_keys = list(dict.fromkeys(metric_keys))  # deduplicate preserving order

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        key = (r["similarity"], r["k"], r["strategy"])
        groups[key].append(r)

    aggregated: list[dict[str, Any]] = []
    for (sim, k, strat), rows in sorted(groups.items()):
        agg: dict[str, Any] = {
            "similarity": sim,
            "k": k,
            "strategy": strat,
            "n_folds": len(rows),
        }
        for m in metric_keys:
            vals = [r[m] for r in rows if m in r and not np.isnan(r[m])]
            agg[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
            agg[f"{m}_std"] = float(np.std(vals)) if vals else float("nan")
        aggregated.append(agg)

    return aggregated


def print_summary(aggregated: list[dict[str, Any]]) -> None:
    """Print a formatted summary table of aggregated CV results.

    :param aggregated: Output of `aggregate_results`.
    """
    print(f"\n{'='*90}")
    print(
        f"  {'SIM':<12} {'K':<6} {'STRATEGY':<16} {'MAE(all)':<12} {'MAE(ok)':<12} {'RMSE(ok)':<12} {'MAP(ok)'}"
    )
    print(f"{'─'*90}")
    for r in aggregated:
        print(
            f"  {r['similarity']:<12} {r['k']:<6} {r['strategy']:<16} "
            f"{r.get('mae_all_mean', float('nan')):<12.4f} "
            f"{r.get('mae_ok_mean', float('nan')):<12.4f} "
            f"{r.get('rmse_ok_mean', float('nan')):<12.4f} "
            f"{r.get('map_ok_mean', float('nan')):.4f}"
        )
    print(f"{'='*90}\n")


if __name__ == "__main__":
    DATA_DIR = "./data/collaborative_filtering"

    N_FOLDS = 5
    SIMILARITIES = ["euclidean", "pearson", "jmsd"]
    K_VALUES = [10, 20, 30, 50]
    STRATEGIES = ["w_mean", "mean_centered", "z_score"]
    K_MAX = max(K_VALUES)
    BATCH_SIZE = 4_096
    COLD_START = "mean"  # "mean" || "static"
    MAP_THRESHOLD = 7.0
    NDCG_K = 10
    SEED = 42

    OUTPUT_DIR = "./cv_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load only the training data - the test set must never be used in CV
    train, _test = load_data(DATA_DIR)
    print(
        f"Train  {train.shape[0]:>8,} ratings | "
        f"{len(np.unique(train[:, 0])):,} users | "
        f"{len(np.unique(train[:, 1])):,} items"
    )

    # Run CV
    raw_results = cross_validate(
        train,
        n_folds=N_FOLDS,
        k_values=K_VALUES,
        similarities=SIMILARITIES,
        strategies=STRATEGIES,
        k_max=K_MAX,
        batch_size=BATCH_SIZE,
        cold_start_handler_name=COLD_START,
        map_threshold=MAP_THRESHOLD,
        ndcg_k=NDCG_K,
        seed=SEED,
    )

    save_cv_results(raw_results, os.path.join(OUTPUT_DIR, "cv_raw.csv"))

    aggregated = aggregate_results(raw_results)
    save_cv_results(aggregated, os.path.join(OUTPUT_DIR, "cv_aggregated.csv"))

    print_summary(aggregated)

    # Find best configuration by MAE on OK cases
    best = min(aggregated, key=lambda r: r.get("mae_ok_mean", float("inf")))
    print(
        f"Best config by MAE(ok):  similarity={best['similarity']} | "
        f"K={best['k']}  strategy={best['strategy']} | "
        f"MAE={best['mae_ok_mean']:.4f} ± {best['mae_ok_std']:.4f}"
    )
