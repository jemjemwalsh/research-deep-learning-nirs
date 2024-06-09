import os

import study_utils

STUDY_NAME = "study_pls"  # define the optuna study name


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
            "n_components": 11,
            
        }
    )


if __name__ == "__main__":
    main()
