from __future__ import annotations

import os

import numpy as np

from collaborative import diagnose_cold_start, predict_knn, print_cold_start_report
from i_o import load_data, save_predictions

if __name__ == "__main__":
    os.makedirs("./preds", exist_ok=True)

    DATA_DIR = "./data/collaborative_filtering"
    SIMILARITY_FUNCTION = "euclidean"
    STRATEGY_FUNCTION = "w_mean"
    K = 3
    OUTPUT_FILE = f"./preds/{K}NN_{SIMILARITY_FUNCTION}_{STRATEGY_FUNCTION}.csv"

    BATCH_SIZE = 4_096

    train, test = load_data(DATA_DIR)

    print(
        f"Train  {train.shape[0]:>8,} ratings | {len(np.unique(train[:, 0])):,} users | {len(np.unique(train[:, 1])):,} items"
    )
    print(
        f"Test   {test.shape[0]:>8,} cases   | {len(np.unique(test[:, 1])):,} users | {len(np.unique(test[:, 2])):,} items"
    )

    cold_map, summary = diagnose_cold_start(train, test)
    print_cold_start_report(summary)

    results = predict_knn(
        train,
        test,
        k=K,
        similarity=SIMILARITY_FUNCTION,
        strategy=STRATEGY_FUNCTION,
        batch_size=BATCH_SIZE,
        cold_start_map=cold_map,
    )

    save_predictions(results, OUTPUT_FILE)
    print(f"\nDone — predictions saved to {OUTPUT_FILE}")
