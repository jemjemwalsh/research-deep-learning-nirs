from optuna import Trial


def get_parameters(trial: Trial) -> dict:
    params = {}
    params["preprocessing"] = get_preprocessing_parameters(trial)
    params["model"] = get_model_parameters(trial)
    return params


def get_preprocessing_parameters(trial: Trial) -> dict:
    
    params = {}
    
    params["pretreatment"] = trial.suggest_categorical(
        name="pretreatment", 
        choices=["pretreatment_0", "pretreatment_1", "pretreatment_2", "pretreatment_3"]
    )
    
    if params["pretreatment"] in ["pretreatment_1", "pretreatment_2", "pretreatment_3"]:
        params["savgol_window_size"] = trial.suggest_int(
            name="savgol_window_size",
            low=9,
            high=41, 
            step=4
        )
    else:
        params["savgol_window_size"] = None
    
    params["wavelength_range"] = trial.suggest_categorical(
        name="wavelength_range", 
        choices=["range_684to990", "range_720to990", "range_600to990", "range_402to990"]
    )
    
    return params


def get_model_parameters(trial: Trial) -> dict:
    
    params = {}
    
    params["n_components"] = trial.suggest_int(
        name="n_components",
        low=1,
        high=40,
        step=1
    )
    
    return params
