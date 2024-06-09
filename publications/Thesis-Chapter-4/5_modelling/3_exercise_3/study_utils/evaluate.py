from pandas import DataFrame
from math import sqrt
from tensorflow import keras


def objective_value(
        mse_tuning: float,
        mse_calibration: float
    ) -> float:
    return (
        0.5 * sqrt(mse_tuning) 
        + 0.5 * abs(mse_tuning - mse_calibration)
    )


def model_scores(model: keras.models.Sequential, cal: DataFrame, tune: DataFrame, y_col: list, x_cols: list) -> dict:
    partitions = [
        ("calibration", cal[x_cols], cal[y_col]),
        ("tuning", tune[x_cols], tune[y_col])
    ]
    scores = {}
    for partition, data_x, data_y in partitions:
        partition_scores = model.evaluate(x=data_x, y=data_y, verbose=0)
        scores[partition] = dict(zip(model.metrics_names, partition_scores))
        scores[partition]["rmse"] = sqrt(scores[partition]["mse"])
    return scores
