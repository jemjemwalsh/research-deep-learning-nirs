from optuna import Trial

from . import evaluate
from .model import create_model
from .parameters import get_parameters
from .preprocess import preprocess_data


MAX_EPOCHS = 750  # define the maximum number of epochs each model in the study can be trained for
RANDOM_SEED = 42  # specify a random seed for reproducibility


def objective(trial: Trial) -> float:
    """Define the objective function to be minimized.
    
    """
    # define the model/study name
    study_name = trial.study.study_name
    model_name = f"trial_{trial.number}"

    # get parameters to test from the the search space
    params = get_parameters(trial)
    
    # preprocess data
    df_cal, df_tune, y_col, x_cols = preprocess_data(
        wavelength_range=params["preprocessing"]["wavelength_range"], 
        pretreatment=params["preprocessing"]["pretreatment"],
        savgol_window_size=params["preprocessing"]["savgol_window_size"]
    )
    
    # model initialization
    model = create_model(
        n_components=params["model"]["n_components"]
    )
    
    # train model 
    model.fit(
        X=df_cal[x_cols],
        Y=df_cal[y_col]
    )
    
    # evaluate the model
    model_scores = evaluate.model_scores(
        model=model,
        cal=df_cal,
        tune=df_tune,
        y_col=y_col,
        x_cols=x_cols
    )
    objective_value = evaluate.objective_value(
        mse_tuning=model_scores["tuning"]["mse"],
        mse_calibration=model_scores["calibration"]["mse"]
    )
    
    # store metrics with the trial object
    trial.set_user_attr("metric_calibration_rmse", model_scores["calibration"]["rmse"])
    trial.set_user_attr("metric_tuning_rmse", model_scores["tuning"]["rmse"])
    trial.set_user_attr("metric_calibration_mse", model_scores["calibration"]["mse"])
    trial.set_user_attr("metric_tuning_mse", model_scores["tuning"]["mse"])
        
    return objective_value
