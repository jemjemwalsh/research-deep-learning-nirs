import os

import study_utils

STUDY_NAME = "study_architecture"  # define the optuna study name


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
            
            # model
            "conv_layers": 1,
            "conv_1_kernels": 1,
            "conv_1_kernel_size": 1,
            "dense_layers": 3,
            "dense_1_units": 36,
            "dense_2_units": 18,
            "dense_3_units": 12,
            "dense_1_dropout": 0,
            "dense_2_dropout": 0,
            
            # training
            "reg_beta": 0.0105,
            "lr_init": 0.004742,
            "lr_min_ratio": 0.05,
            "batch_size": 320
        }
    )


if __name__ == "__main__":
    main()
