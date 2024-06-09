import os

import pickle

import utils

MODEL = "CNN"
PREPROCESSING_PARAMS = {
    "CNN": {
        "pretreatment": "pretreatment_2",
        "savgol_window_size": 13,
        "wavelength_range": "range_402to990"
    },
    "ANN": {
        "pretreatment": "pretreatment_2",
        "savgol_window_size": 9,
        "wavelength_range": "range_402to990"
    },
    "PLS": {
        "pretreatment": "pretreatment_2",
        "savgol_window_size": 9,
        "wavelength_range": "range_402to990"
    },
}


def main():
    
    # change current working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # initialise data directory
    os.makedirs(name=f"data/preprocessed_data/{MODEL}", exist_ok=True)
    
    # preprocess data
    df, y_col, x_cols = utils.preprocess_data(
        wavelength_range=PREPROCESSING_PARAMS[MODEL]["wavelength_range"], 
        pretreatment=PREPROCESSING_PARAMS[MODEL]["pretreatment"],
        savgol_window_size=PREPROCESSING_PARAMS[MODEL]["savgol_window_size"],
        return_partitioned=False
    )
    
    # save preprocessed data and metadata
    df.to_pickle(f"data/preprocessed_data/{MODEL}/dataset.pkl")
    with open(f"data/preprocessed_data/{MODEL}/columns.pkl", "wb") as file:
        pickle.dump([y_col, x_cols], file)


if __name__ == "__main__":
    main()
