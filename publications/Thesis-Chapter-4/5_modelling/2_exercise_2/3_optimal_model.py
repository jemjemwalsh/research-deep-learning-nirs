import os

import optuna
import tensorflow as tf

import study_utils


STUDY_NAME = "study_preprocessing_final_2"  # define the optuna study name
TRIAL_NUMBER = None
TRAIN = False
TRAINING_SETS = "train_validation"


def main():

    # verify gpu
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
    
    # change current working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # initialise data directory
    os.makedirs(name=f"data/{STUDY_NAME}/models/eval/predict", exist_ok=True)
    os.makedirs(name=f"data/{STUDY_NAME}/models/eval/metrics", exist_ok=True)

    # load the study
    study = optuna.load_study(
        study_name=STUDY_NAME, 
        storage=f"sqlite:///data/{STUDY_NAME}/{STUDY_NAME}.db"
    )
    print("Best trial:", study.best_trial.number)
    trial_number = TRIAL_NUMBER if TRIAL_NUMBER else study.best_trial.number
    model_name = f"trial_{trial_number}"
    if TRAIN and TRAINING_SETS == "train_validation":
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
    
    # model initialization and compile
    model = study_utils.create_model(
        input_dims=len(x_cols),
        conv_layers=params["model"]["conv_layers"],
        conv_kernels=params["model"]["conv_kernels"],
        conv_kernel_sizes=params["model"]["conv_kernel_sizes"],
        dense_layers=params["model"]["dense_layers"],
        dense_units=params["model"]["dense_units"],
        dense_dropout=params["model"]["dense_dropout"],
        reg_beta=params["model"]["reg_beta"],
        learning_rate=params["training"]["lr_init"],
        random_seed=42
    )

    # define callbacks
    callbacks = study_utils.create_callbacks(
        model_directory=f"data/{STUDY_NAME}/models/eval",
        model_name=model_name,
        lr_min=params["training"]["lr_min"],
        model_checkpoint=True,
        save_weights_only=False
    )
    
    if TRAIN:
        
        if TRAINING_SETS == "train_validation":
            df_cal = df.query("partition in ('train', 'validation') and train_partition == 'calibration'")
            df_tune = df.query("partition in ('train', 'validation') and train_partition == 'tunning'")
        else:
            df_cal = df.query("partition == 'train' and train_partition == 'calibration'")
            df_tune = df.query("partition == 'train' and train_partition == 'tunning'")
        
        # train model
        history = model.fit(
            x=df_cal[x_cols],
            y=df_cal[y_col],
            batch_size=params["training"]["batch_size"],
            epochs=750,
            validation_data=(df_tune[x_cols], df_tune[y_col]),
            callbacks=callbacks,
            verbose=0
        )
        print("Trained Epochs:", len(history.history["loss"]))
    
        # load the best model weights
        model.load_weights(f"data/{STUDY_NAME}/models/eval/ckpt/{model_name}.model.keras")
    
    else:
        
        # load the model weights from the study optimization
        model.load_weights(f"data/{STUDY_NAME}/models/ckpt/{model_name}.model.keras")
        
        # save model
        model.save(f"data/{STUDY_NAME}/models/eval/{model_name}.model.keras")
    
    # make and save predictions
    df_pred = df.copy()
    df_pred["y_true"] = df_pred["dry_matter"]
    df_pred["y_pred"] = model.predict(df[x_cols])
    df_pred.to_pickle(f"data/{STUDY_NAME}/models/eval/predict/{model_name}.pkl")


if __name__ == "__main__":
    main()
