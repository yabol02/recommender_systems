from __future__ import annotations

import numpy as np


def load_data(directory: str) -> tuple[np.ndarray, np.ndarray]:
    """Load `train.csv` (user, item, rating) and `test.csv` (ID, user, item).

    :param directory: The directory where the train.csv and test.csv files are located.
    :return: A tuple containing the training data and testing data as numpy arrays.
    """
    train = np.loadtxt(f"{directory}/train.csv", delimiter=",", skiprows=1)
    test = np.loadtxt(f"{directory}/test.csv", delimiter=",", skiprows=1)
    return train, test


def save_predictions(predictions: np.ndarray, output_file: str) -> None:
    """Write (ID, predicted_rating) pairs to a CSV file.

    :param predictions: A numpy array containing the predicted ratings, where each row is (ID, predicted_rating).
    :param output_file: The path to the output CSV file.
    """
    np.savetxt(
        output_file,
        predictions,
        delimiter=",",
        header="ID,rating",
        comments="",
        fmt=["%d", "%.4f"],
    )
