from __future__ import annotations

import os
import numpy as np
import joblib

from collaborative import diagnose_cold_start, print_cold_start_report, ColdStartStatus
from collaborative.cold_start import MedianDampingFallback
from i_o import load_data, save_predictions

from surprise import Reader, Dataset

if __name__ == "__main__":
    os.makedirs("./preds", exist_ok=True)
    DATA_DIR = "./data/cut_1_0"
    RESOURCES_PATH = "LAB_A_models"

    # Load data: train (user, item, rating), test (ID, user, item)
    train_data, test_data = load_data(DATA_DIR)

    print(
        f"Train {train_data.shape[0]:>8,} ratings | {len(np.unique(train_data[:, 0])):,} users | {len(np.unique(train_data[:, 1])):,} items"
    )
    print(
        f"Test   {test_data.shape[0]:>8,} cases   | {len(np.unique(test_data[:, 1])):,} users | {len(np.unique(test_data[:, 2])):,} items"
    )

    # Diagnose cold start
    cold_map, summary = diagnose_cold_start(train_data, test_data)
    print_cold_start_report(summary)

    # Create and setup cold start handler (similar to pmf_main.py)
    cold_start_handler = MedianDampingFallback(damping_factor=2.0)
    cold_start_handler.setup(train_data)

    # Load pre-trained models
    main_model = joblib.load(RESOURCES_PATH + "/top_1_ok_model.pkl")
    fallback_model = joblib.load(RESOURCES_PATH + "/unkn_items_model.pkl")

    # Convert to surprise format
    reader = Reader(rating_scale=(1, 10))
    train_data_df = __import__("pandas").DataFrame(
        train_data, columns=["user", "item", "rating"]
    )
    train_data_surprise = Dataset.load_from_df(train_data_df, reader=reader)
    train_data_surprise = train_data_surprise.build_full_trainset()

    # Train models
    main_model.fit(train_data_surprise)
    fallback_model.fit(train_data_surprise)

    # Make predictions with cold start handling
    predictions = []

    for idx in range(len(test_data)):
        row = test_data[idx]
        test_id = int(row[0])
        user_id = int(row[1])
        item_id = int(row[2])
        
        status = cold_map[idx]
        
        # Determine which model to use based on cold start status
        if status == ColdStartStatus.OK:
            pred = main_model.predict(user_id, item_id)
            rating_pred = pred.est
        else:
            # For cold cases, use the handler directly
            rating_pred = cold_start_handler.predict(user_id, item_id, status)

        # Round to integer [1, 10]
        rating_pred = int(np.clip(np.rint(rating_pred), 1, 10))
        predictions.append([test_id, rating_pred])

    # Convert to numpy array
    predictions = np.array(predictions)

    # Save predictions
    save_predictions(predictions, "./preds/test_with_handler.csv")
    print("Done - predictions saved to ./preds/test_with_handler.csv")
