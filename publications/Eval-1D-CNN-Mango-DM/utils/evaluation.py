import datetime as dt
import numpy as np
import os
import pandas as pd
import scipy.stats as stats

from sklearn import metrics


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    # define custom function to calculate SEP, RMSE, Bias, and RPD for all sf

    """
    n = y_true.shape[0]
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    y_error = y_true - y_pred
    mean_error = np.mean(y_error)
    std_error = np.sqrt(np.square(y_error - mean_error).sum() / (n-1))
    std_true = np.sqrt(np.square(y_true - y_true.mean()).sum() / (n-1))
    return {
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


def significance_difference(e_1: np.ndarray, e_2: np.ndarray) -> dict:
    # see https://doi.org/10.1255/nirn.378 (Fearn, 1996) for explanation

    n = e_1.shape[0]
    m_1 = e_1.mean()
    m_2 = e_2.mean()

    # bias
    d = e_1 - e_2
    s_d = np.sqrt(np.square(d - d.mean()).sum() / (n*(n-1)))
    t = stats.t.ppf(q=(1 - 0.025), df=(n - 1))
    bias_ci = (round(m_1 - m_2 - t * s_d, 3), round(m_1 - m_2 + t * s_d, 3))  # 95% confidence interval
    bias_ss = False if bias_ci[0] <= 0 <= bias_ci[1] else True  # check for statistical significance

    # standard deviation
    r, _ = stats.pearsonr(e_1.flatten(), e_2.flatten())
    t = stats.t.ppf(q=(1 - 0.025), df=(n - 2))
    k = 1 + (2 * (1 - r**2) * t**2)/(n - 2)
    l = np.sqrt(k + np.sqrt(k**2 - 1))
    s_1 = np.sqrt(np.square(e_1 - m_1).sum() / (n - 1))
    s_2 = np.sqrt(np.square(e_2 - m_2).sum() / (n - 1))
    s_ci = (round(s_1/s_2 * 1/l, 3), round(s_1/s_2 * l, 3))  # 95% confidence interval
    s_ss = False if s_ci[0] <= 1 <= s_ci[1] else True  # check for statistical significance

    return {
        "bias_diff_ci": bias_ci,
        "bias_diff_ss": bias_ss,
        "true_std_ratio_ci": s_ci,
        "true_std_ratio_ss": s_ss,
    }


def print_metrics(m: dict):
    print("Error Metrics: \t\t Cal \t\t Tune \t\t Test")
    print(f'R2: \t\t\t\t {round(m["cal"]["r2"], 3)} \t\t {round(m["tune"]["r2"], 3)} \t\t {round(m["test"]["r2"], 3)}')
    print(f'RMSE: \t\t\t\t {round(m["cal"]["rmse"], 3)} \t\t {round(m["tune"]["rmse"], 3)} \t\t {round(m["test"]["rmse"], 3)}')
    print(f'SIG: \t\t\t\t  \t\t  \t\t {m["test"]["stat_tests"]["true_std_ratio_ss"]}')
    print("\n")


def save_results(model_specs: dict):
    results_file = "data/results/raw_results.csv"
    new_model_results = pd.json_normalize([model_specs])
    if os.path.exists(results_file):
        model_results = pd.read_csv(results_file)
        model_results = pd.concat([model_results, new_model_results])
    else:
        model_results = new_model_results
    model_results.to_csv(results_file, index=False)


def model_stats(model_object, model_specs: dict, data_dict: dict, save=True):
    for s in ("train", "cal", "tune", "test"):
        prediction_time = dt.datetime.now()
        y_predictions = model_object.predict(data_dict[f"x_{s}"])
        prediction_time = dt.datetime.now() - prediction_time

        # calculate model fit metrics
        model_specs["metrics"][s] = calculate_metrics(
            y_true=data_dict[f"y_{s}"],
            y_pred=y_predictions,
        )

        # add prediction time
        model_specs["metrics"][s]["prediction_time"] = prediction_time
        model_specs["metrics"][s]["prediction_time_avg"] = prediction_time/len(y_predictions)

        # statistical tests against best reported model (Mishra & Passos, 2021)
        model_name = "mishra&passos_2021_best_cnn"
        base_model_preds = pd.read_csv(f"data/predictions/{model_name}.csv")
        if s == "test":
            model_specs["metrics"][s]["stat_tests"] = significance_difference(
                e_1=base_model_preds.query(f"test_data == '{s}'")["y_error"].to_numpy(),
                e_2=(data_dict[f"y_{s}"] - y_predictions)
            )

    print_metrics(model_specs["metrics"])
    print(model_specs)
    if save:
        save_results(model_specs)


def save_model_predictions(model_object, data_dict: dict, model_name: str):
    model_preds = []
    for s in ("train", "cal", "tune", "test"):
        y_true = data_dict[f"y_{s}"]
        y_pred = model_object.predict(data_dict[f"x_{s}"])
        y_error = y_true - y_pred
        df = pd.DataFrame(
            data={
                "model_name": model_name,
                "test_data": s,
                "y_true": y_true.flatten(),
                "y_pred": y_pred.flatten(),
                "y_error": y_error.flatten()
            }
        )
        model_preds.append(df)
    model_preds = pd.concat(model_preds)
    model_preds.to_csv(f"data/predictions/{model_name}.csv")
