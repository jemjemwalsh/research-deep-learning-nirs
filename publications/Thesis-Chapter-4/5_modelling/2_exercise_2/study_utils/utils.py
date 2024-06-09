import numpy as np
from sklearn import metrics


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate SEP, RMSE, Bias, and RPD of predictions

    """
    n = y_true.shape[0]
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    y_error = y_true - y_pred
    mean_error = np.mean(y_error)
    std_error = np.sqrt(np.square(y_error - mean_error).sum() / (n-1))
    std_true = np.sqrt(np.square(y_true - y_true.mean()).sum() / (n-1))
    return {
        # number of samples
        "n": len(y_true),
        
        # calculate r-squared (R2)
        "r2": metrics.r2_score(y_true, y_pred),

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
            "conv_layers": params_raw.get("conv_layers", 1),
            "conv_kernels": {},
            "conv_kernel_sizes": {},
            "dense_layers": params_raw.get("dense_layers", 3),
            "dense_units": {},
            "dense_dropout": {},
            "reg_beta": params_raw.get("reg_beta"),
        },
        "training": {
            "lr_init": params_raw.get("lr_init"),
            "lr_min_ratio": params_raw.get("lr_min_ratio"),
            "lr_min": params_raw.get("lr_init") * params_raw.get("lr_min_ratio"),
            "batch_size": params_raw.get("batch_size")
        }
    }

    for i in range(1, params["model"]["conv_layers"] + 1):
        params["model"]["conv_kernels"][i] = params_raw.get(f"conv_{i}_kernels", 1)
        params["model"]["conv_kernel_sizes"][i] = params_raw.get(f"conv_{i}_kernel_size")

    for i in range(1, params["model"]["dense_layers"] + 1):
        default_units = {1: 36, 2: 18, 3: 12}
        params["model"]["dense_units"][i] = params_raw.get(f"dense_{i}_units", default_units[i])
        if i != params["model"]["dense_layers"]:  # for all but the last layer
            params["model"]["dense_dropout"][i] = params_raw.get(f"dense_{i}_dropout", 0)

    return params
