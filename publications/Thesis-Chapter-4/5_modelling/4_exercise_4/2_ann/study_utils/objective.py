from optuna import Trial

from . import evaluate
from .model import create_model
from .parameters import get_parameters
from .preprocess import preprocess_data
from .training import create_callbacks


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
    
    # model initialization and compile
    model = create_model(
        input_dims=len(x_cols),
        dense_layers=params["model"]["dense_layers"],
        activation_function=params["model"]["activation_function"],
        dense_units=params["model"]["dense_units"],
        dense_dropout=params["model"]["dense_dropout"],
        reg_beta=params["model"]["reg_beta"],
        learning_rate=params["training"]["lr_init"],
        random_seed=RANDOM_SEED
    )
    
    # define callbacks
    model_directory = f"data/{study_name}/models"
    callbacks = create_callbacks(
        trial=trial,
        model_directory=model_directory,
        model_name=model_name,
        lr_min=params["training"]["lr_min"],
        model_checkpoint=True
    )
    
    # train model 
    history = model.fit(
        x=df_cal[x_cols],
        y=df_cal[y_col],
        batch_size=params["training"]["batch_size"],
        epochs=MAX_EPOCHS,
        validation_data=(df_tune[x_cols], df_tune[y_col]),
        callbacks=callbacks,
        verbose=0
    )
    
    # load the best model weights
    model.load_weights(f"{model_directory}/ckpt/{model_name}.model.keras")
    
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
    trial.set_user_attr("trained_epochs", len(history.history["loss"]))
        
    return objective_value
