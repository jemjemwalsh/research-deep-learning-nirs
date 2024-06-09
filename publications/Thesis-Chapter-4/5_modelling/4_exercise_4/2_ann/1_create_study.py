import os

import study_utils

STUDY_NAME = "study_ann"  # define the optuna study name


def main():
    
    # change current working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # initialise data directory
    os.makedirs(name=f"data/{STUDY_NAME}/models", exist_ok=True)
    
    # create study
    study = study_utils.create_or_load_study(STUDY_NAME)
    
    # add known 'good' hyperparameters to start search from a reasonable solution
    study.enqueue_trial(
        params={
            # preprocessing
            "pretreatment": "pretreatment_2",
            "savgol_window_size": 13,
            "wavelength_range": "range_684to990",
            
            # model
            "activation_function": "sigmoid",
            "dense_layers": 1,
            "dense_1_units": 5,
            
            # training
            "reg_beta": 0.0055,
            "lr_init": 0.005,
            "lr_min_ratio": 0.05,
            "batch_size": 128
        }
    )


if __name__ == "__main__":
    main()
