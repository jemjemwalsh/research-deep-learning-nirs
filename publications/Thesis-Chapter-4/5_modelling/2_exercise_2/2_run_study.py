import os

import optuna
import tensorflow as tf

import study_utils

STUDY_NAME = "study_preprocessing"  # define the optuna study name
N_TRIALS = 5000  # define the number of trials to conduct in the study


def main():

    # verify gpu
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
    
    # change current working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # load the existing study
    study = study_utils.create_or_load_study(STUDY_NAME)
    
    study.optimize(
        func=study_utils.objective, 
        n_trials=N_TRIALS,
        n_jobs=3  # enables multithread parallelism to increase speed
    )


if __name__ == "__main__":
    main()
