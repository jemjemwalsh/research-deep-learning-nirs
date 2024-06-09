import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, plot: bool = True) -> dict:
    """Calculate SEP, RMSE, Bias, and RPD of predictions

    """
    if plot:
        plot_predictions(y_true, y_pred)
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
    

def plot_model_history(history: dict):
    plt.figure(figsize=(8, 4))
    plt.plot(history["loss"], label="Calibration set loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Tunning set loss")
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    ax2 = plt.gca().twinx()
    ax2.plot(history["lr"], color="r", ls="--")
    ax2.set_ylabel("learning rate", color="r")
    plt.tight_layout()
    plt.show()
    
    
def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    plt.scatter(y_true, y_pred)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], linestyle='--', color='red')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.show()