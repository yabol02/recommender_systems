from __future__ import annotations

import os

import numpy as np

from collaborative import diagnose_cold_start, print_cold_start_report
from collaborative.cold_start import MeanFallback
from collaborative.pmf import PMF, PMF_CONFIGS, SVDPP_CONFIGS, PMFConfig, SVDPlusPlus
from i_o import load_data, save_predictions

if __name__ == "__main__":
    os.makedirs("./preds", exist_ok=True)

    DATA_DIR = "./data/collaborative_filtering"

    # ------------------------------------------------------------------ #
    # Choose algorithm and configuration:
    #   MODEL  : PMF | SVDPlusPlus
    #   PRESET : named preset (or None for the manual configuration)
    #
    #   PMF presets   -> "default" | "deep" | "sgd" | "pure_pmf"
    #   SVD++ presets -> "default" | "deep" | "sgd"
    # ------------------------------------------------------------------ #
    MODEL = PMF  # <- PMF | SVDPlusPlus
    PRESET = "pure_pmf"  # <- Previous preset name or None

    registry = SVDPP_CONFIGS if MODEL is SVDPlusPlus else PMF_CONFIGS

    if PRESET is not None:
        cfg = registry[PRESET]
    else:
        # PRESET to None and manually configure hyperparameters here
        cfg = PMFConfig(
            n_factors=None,
            n_epochs=None,
            lr=None,
            reg_user=None,
            reg_item=None,
            reg_bias=None,
            tol=None,
            min_rating=None,
            max_rating=None,
            use_biases=True,
            batch_size=None,
            seed=None,
        )

    algo_name = "svdpp" if MODEL is SVDPlusPlus else "pmf"
    config_tag = (
        PRESET
        if PRESET is not None
        else f"K{cfg.n_factors}_lr{cfg.lr}_reg{cfg.reg_user}"
    )
    OUTPUT_FILE = f"./preds/{algo_name}_{config_tag}.csv"

    train, test = load_data(DATA_DIR)

    print(
        f"Train {train.shape[0]:>8,} ratings | {len(np.unique(train[:, 0])):,} users | {len(np.unique(train[:, 1])):,} items"
    )
    print(
        f"Test   {test.shape[0]:>8,} cases   | {len(np.unique(test[:, 1])):,} users | {len(np.unique(test[:, 2])):,} items"
    )

    cold_map, summary = diagnose_cold_start(train, test)
    print_cold_start_report(summary)

    cold_start_handler = MeanFallback()
    cold_start_handler.setup(train)

    model = MODEL(cfg)
    model.fit(train)
    results = model.predict_test(test, cold_start_handler=cold_start_handler)
    save_predictions(results, OUTPUT_FILE)
    print(f"Done - predictions saved to {OUTPUT_FILE}")
