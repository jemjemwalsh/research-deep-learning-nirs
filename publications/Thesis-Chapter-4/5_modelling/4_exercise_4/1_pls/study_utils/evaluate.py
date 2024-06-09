from pandas import DataFrame
from math import sqrt
from sklearn.cross_decomposition import PLSRegression

from .utils import calculate_metrics


def objective_value(
        mse_tuning: float,
        mse_calibration: float
    ) -> float:
    return (
        0.5 * sqrt(mse_tuning) 
        + 0.5 * abs(mse_tuning - mse_calibration)
    )


def model_scores(model: PLSRegression, cal: DataFrame, tune: DataFrame, y_col: list, x_cols: list) -> dict:
    partitions = [
        ("calibration", cal[x_cols], cal[y_col]),
        ("tuning", tune[x_cols], tune[y_col])
    ]
    scores = {}
    for partition, data_x, data_y in partitions:
        scores[partition] = calculate_metrics(
            y_true=data_y,
            y_pred=model.predict(X=data_x) 
        )
    return scores
