from __future__ import annotations

# ------------------------------------------------------------------ #
# En este .py vamos a predecir el test set de kaggle con los modelos
# entrenados en el LAB_A.ipynb. La idea es la siguiente:
# MAIN MODEL: Entrenado y optimizado mediante optuna. Probablemente un SVD
# Fallback model: idem. Un BaseLineOnly model
# fallback para cold user e item: la media
# Luego redondeamos cada prediccion al entero más cercano

import os
import numpy as np
import polars as pl
import joblib

from collaborative import diagnose_cold_start, print_cold_start_report, ColdStartStatus
from i_o import load_data, save_predictions

from surprise import Reader, Dataset

def attach_status_polars(df: pl.DataFrame, status_map: dict[int, ColdStartStatus]) -> pl.DataFrame:
    # Convertimos el Enum a string (.name) para que Polars trabaje con tipo de dato String nativo
    status_list = [status_map[i].name for i in range(df.height)]
    
    return df.with_columns(
        pl.Series("status", status_list)
    )
def split_by_status(df: pl.DataFrame):
    return {
        "ok": df.filter(pl.col("status") == ColdStartStatus.OK.name),
        "cold_user": df.filter(pl.col("status") == ColdStartStatus.COLD_USER.name),
        "unknown_item": df.filter(pl.col("status") == ColdStartStatus.UNKNOWN_ITEM.name),
        "both": df.filter(pl.col("status") == ColdStartStatus.BOTH.name),
    }

if __name__ == "__main__":
    os.makedirs("./preds", exist_ok=True)
    DATA_DIR = "./data/collaborative_filtering"
    RESOURCES_PATH = "LAB_A_models"

    train_data = pl.read_csv(DATA_DIR + "/train.csv")
    test_data = pl.read_csv(DATA_DIR + "/test.csv")
    #train_data, test_data = load_data(DATA_DIR)

    # Cargamos modelos
    main_model = joblib.load(RESOURCES_PATH + "/top_1_ok_model.pkl")
    fallback_model = joblib.load(RESOURCES_PATH + "/unkn_items_model.pkl")

    # Dividimos por tipo de cold (user, item, both o ninguno)
    cold_map, summary = diagnose_cold_start(train_data.to_numpy(), test_data.to_numpy())
    print_cold_start_report(summary)
    test_data = attach_status_polars(test_data, cold_map)
    test_splits = split_by_status(test_data)

    test_ok = test_splits["ok"]
    test_cold_user = test_splits["cold_user"]
    test_unknown_item = test_splits["unknown_item"]
    test_both = test_splits["both"]

    # Pasamos tipo de datos a formato surprise:

    # Train set
    reader = Reader(rating_scale=(1, 10))
    train_data_surprise = Dataset.load_from_df(train_data.to_pandas(), reader=reader)
    train_data_surprise = train_data_surprise.build_full_trainset()
    # Casos enteros
    test_ok_surprise = [
    (row["user"], row["item"], 0)
        for row in test_ok.select(["user", "item"]).to_dicts()
    ]
    # Casos cold
    test_cold_users = [
    (row["user"], row["item"], 0)
        for row in test_cold_user.select(["user", "item"]).to_dicts()
    ]
    test_cold_items = [
    (row["user"], row["item"], 0)
        for row in test_unknown_item.select(["user", "item"]).to_dicts()
    ]
    # Casos cold cold
    test_cold_cold = [
    (row["user"], row["item"], 0)
        for row in test_both.select(["user", "item"]).to_dicts()
    ]

    # ML
    # Entrenamiento:
    main_model.fit(train_data_surprise)
    fallback_model.fit(train_data_surprise)
    global_mean = train_data["rating"].mean()

    # Casos ok
    preds_ok = main_model.test(test_ok_surprise) 
    # Casos cold user
    preds_cold_user = fallback_model.test(test_cold_users)
    # Casos cold item
    preds_cold_item = fallback_model.test(test_cold_items)


    # ------------------------------------------------------------------
    # Crear el submission final con la columna ID original del test

    # Casos OK
    ok_ids = test_ok.select("ID")
    ok_ratings = [float(p.est) for p in preds_ok]
    ok_ratings = [int(np.clip(np.rint(x), 1, 10)) for x in ok_ratings] # Redondeo y limite 1-10
    ok_pred_df = ok_ids.with_columns(
        pl.Series("rating", ok_ratings).cast(pl.Int64)
    )

    # Casos cold user
    cold_user_ids = test_cold_user.select("ID")
    cold_user_ratings = [float(p.est) for p in preds_cold_user]
    cold_user_ratings = [int(np.clip(np.rint(x), 1, 10)) for x in cold_user_ratings]
    cold_user_pred_df = cold_user_ids.with_columns(
        pl.Series("rating", cold_user_ratings).cast(pl.Int64)
    )

    # Casos unknown item
    cold_item_ids = test_unknown_item.select("ID")
    cold_item_ratings = [float(p.est) for p in preds_cold_item]
    cold_item_ratings = [int(np.clip(np.rint(x), 1, 10)) for x in cold_item_ratings]
    cold_item_pred_df = cold_item_ids.with_columns(
        pl.Series("rating", cold_item_ratings).cast(pl.Int64)
    )

    # Casos BOTH -> global_mean
    both_rating = int(np.clip(np.rint(global_mean), 1, 10))
    both_pred_df = test_both.select("ID").with_columns(
        pl.lit(both_rating).alias("rating").cast(pl.Int64)
    )

    # Unir todo y ordenar por ID
    submission = pl.concat([
        ok_pred_df,
        cold_user_pred_df,
        cold_item_pred_df,
        both_pred_df
    ]).sort("ID")

    # Guardar con el nombre y columnas correctas
    submission.write_csv("./preds/lab_a.csv")