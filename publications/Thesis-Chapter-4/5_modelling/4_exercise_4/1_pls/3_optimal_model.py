import os

import optuna

import study_utils


STUDY_NAME = "study_pls"  # define the optuna study name
TRIAL_NUMBER = None
# TRAINING_SETS = "train_validation"
TRAINING_SETS = "train"

def main():
    
    # change current working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # initialise data directory
    os.makedirs(name=f"data/{STUDY_NAME}/models/eval/predict", exist_ok=True)

    # load the study
    study = optuna.load_study(
        study_name=STUDY_NAME, 
        storage=f"sqlite:///data/{STUDY_NAME}/{STUDY_NAME}.db"
    )
    print("Best trial:", study.best_trial.number)
    trial_number = TRIAL_NUMBER if TRIAL_NUMBER else study.best_trial.number
    model_name = f"trial_{trial_number}"
    if TRAINING_SETS == "train_validation":
        model_name = f"{model_name}_retrain"

    # setup best model
    params = study_utils.utils.parse_trial_params(
        params_raw=study.get_trials()[trial_number].params
    )
    print("Hyperparameters:\n", params)

    # preprocess data
    df, y_col, x_cols = study_utils.preprocess_data(
        wavelength_range=params["preprocessing"]["wavelength_range"], 
        pretreatment=params["preprocessing"]["pretreatment"],
        savgol_window_size=params["preprocessing"]["savgol_window_size"],
        return_partitioned=False
    )
    
    # model initialization and training
    model = study_utils.create_model(
        n_components=params["model"]["n_components"]
    )
    if TRAINING_SETS == "train_validation":
        df_cal = df.query("partition in ('train', 'validation') and train_partition == 'calibration' and subsequent_flag_1 == 0")
        df_tune = df.query("partition in ('train', 'validation') and train_partition == 'tunning' and subsequent_flag_1 == 0")
    else:
        df_cal = df.query("partition == 'train' and train_partition == 'calibration'")
        df_tune = df.query("partition == 'train' and train_partition == 'tunning'")
    model.fit(
        X=df_cal[x_cols],
        Y=df_cal[y_col]
    )
    
    # make and save predictions
    df_pred = df.copy()
    df_pred["y_true"] = df_pred["dry_matter"]
    df_pred["y_pred"] = model.predict(df[x_cols])
    df_pred.to_pickle(f"data/{STUDY_NAME}/models/eval/predict/{model_name}.pkl")


if __name__ == "__main__":
    main()
