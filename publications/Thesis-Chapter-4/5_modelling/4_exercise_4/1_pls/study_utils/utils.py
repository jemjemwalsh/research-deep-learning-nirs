import numpy as np
from sklearn import metrics


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate SEP, RMSE, Bias, and RPD of predictions

    """
    n = y_true.shape[0]
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    y_error = y_true - y_pred
    mean_error = np.mean(y_error)
    std_error = np.sqrt(np.square(y_error - mean_error).sum() / (n-1))
    std_true = np.sqrt(np.square(y_true - y_true.mean()).sum() / (n-1))
    return {
        # number of samples
        "n": len(y_true),
        
        # calculate r-squared (R2)
        "r2": metrics.r2_score(y_true, y_pred),
        
        # mean square error (MSE)
        "mse": mse,

        # calculate root mean square error (RMSE)
        "rmse": rmse,

        # calculate standard error of prediction (SEP)
        "sep": std_error,

        # calculate bias
        "bias": mean_error,

        # calculate ratio of performance to deviation (RPD)
        "rpd": std_true / std_error,
    }


def parse_trial_params(params_raw: dict) -> dict:
    params = {
        "preprocessing": {
            "wavelength_range": params_raw.get("wavelength_range"),
            "pretreatment": params_raw.get("pretreatment"),
            "savgol_window_size": params_raw.get("savgol_window_size")
        },
        "model": {
            "n_components": params_raw.get("n_components"),
        },
    }

    return params
