from __future__ import annotations

import os

import numpy as np

from collaborative import diagnose_cold_start, predict_knn, print_cold_start_report
from collaborative.cold_start import MeanFallback
from i_o import load_data, save_predictions

if __name__ == "__main__":
    os.makedirs("./preds", exist_ok=True)

    DATA_DIR = "./data/collaborative_filtering"
    SIMILARITY_FUNCTION = "euclidean"  # euclidean | cosine | pearson | jmsd
    STRATEGY_FUNCTION = "z_score"  # mean | w_mean | mean_centered | z_score
    K = 100
    BATCH_SIZE = 4_096
    OUTPUT_FILE = f"./preds/{K}NN_{SIMILARITY_FUNCTION}_{STRATEGY_FUNCTION}.csv"

    train, test = load_data(DATA_DIR)

    print(
        f"Train  {train.shape[0]:>8,} ratings | "
        f"{len(np.unique(train[:, 0])):,} users | "
        f"{len(np.unique(train[:, 1])):,} items"
    )
    print(
        f"Test   {test.shape[0]:>8,} cases   | "
        f"{len(np.unique(test[:, 1])):,} users | "
        f"{len(np.unique(test[:, 2])):,} items"
    )

    cold_map, summary = diagnose_cold_start(train, test)
    print_cold_start_report(summary)

    # MeanFallback: item mean for cold users, user mean for unknown items, global mean otherwise
    cold_start_handler = MeanFallback()
    cold_start_handler.setup(train)

    results = predict_knn(
        train,
        test,
        k=K,
        similarity=SIMILARITY_FUNCTION,
        strategy=STRATEGY_FUNCTION,
        batch_size=BATCH_SIZE,
        cold_start_handler=cold_start_handler,
        cold_start_map=cold_map,
    )

    save_predictions(results, OUTPUT_FILE)
    print(f"\nDone - predictions saved to {OUTPUT_FILE}")
