import os

import optuna

import study_utils

STUDY_NAME = "study_pls"  # define the optuna study name
N_TRIALS = 5000  # define the number of trials to conduct in the study


def main():

    # change current working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # load the existing study
    study = study_utils.create_or_load_study(STUDY_NAME)
    
    study.optimize(
        func=study_utils.objective, 
        n_trials=N_TRIALS,
        n_jobs=5  # enables multithread parallelism to increase speed
    )


if __name__ == "__main__":
    main()
