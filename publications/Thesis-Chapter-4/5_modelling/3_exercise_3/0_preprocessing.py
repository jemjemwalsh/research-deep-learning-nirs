import os

import pickle

import study_utils

STUDY_NAME = "study_architecture"  # define the optuna study name
PREPROCESSING_PARAMS = {
    "pretreatment": "pretreatment_2",
    "savgol_window_size": 13,
    "wavelength_range": "range_402to990"
}


def main():
    
    # change current working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # initialise data directory
    os.makedirs(name=f"data/{STUDY_NAME}/preprocessed_data", exist_ok=True)
    
    # preprocess data
    df, y_col, x_cols = study_utils.preprocess_data(
        wavelength_range=PREPROCESSING_PARAMS["wavelength_range"], 
        pretreatment=PREPROCESSING_PARAMS["pretreatment"],
        savgol_window_size=PREPROCESSING_PARAMS["savgol_window_size"],
        return_partitioned=False
    )
    
    # save preprocessed data and metadata
    df.to_pickle(f"data/{STUDY_NAME}/preprocessed_data/dataset.pkl")
    with open(f"data/{STUDY_NAME}/preprocessed_data/columns.pkl", "wb") as file:
        pickle.dump([y_col, x_cols], file)


if __name__ == "__main__":
    main()
