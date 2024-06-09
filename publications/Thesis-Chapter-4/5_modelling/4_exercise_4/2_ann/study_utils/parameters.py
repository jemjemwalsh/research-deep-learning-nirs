from optuna import Trial


def get_parameters(trial: Trial) -> dict:
    params = {}
    params["preprocessing"] = get_preprocessing_parameters(trial)
    params["model"] = get_architecture_parameters(trial)
    params["training"] = get_training_hyperparameters(trial)
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


def get_architecture_parameters(trial: Trial) -> dict:
    
    params = {}
    
    params["activation_function"] = trial.suggest_categorical(
        name="activation_function", 
        choices=["sigmoid", "elu", "relu"]
    )
    
    params["dense_layers"] = trial.suggest_int(
        name="dense_layers",
        low=1,
        high=5, 
        step=1
    )
    params["dense_units"] = {}
    params["dense_dropout"] = {}
    for layer in range(1, params["dense_layers"]+1):
        params["dense_units"][layer] = trial.suggest_int(
            name=f"dense_{layer}_units", 
            low=6, 
            high=250, 
            step=4
        )
        if layer != params["dense_layers"]:
            params["dense_dropout"][layer] = trial.suggest_float(
                name=f"dense_{layer}_dropout", 
                low=0, 
                high=0.6, 
                step=0.005
            )
        
    params["reg_beta"] = trial.suggest_float(
        name="reg_beta", 
        low=0., 
        high=0.1, 
        step=0.0005
    )
    
    return params


def get_training_hyperparameters(trial: Trial) -> dict:
    
    params = {}
    
    # initial LR to start model training with
    params["lr_init"] = trial.suggest_float(
        name="lr_init", 
        low=1e-7, 
        high=1e-2, 
        log=True
    )
    
    # minimum LR to set in the ReduceLROnPlateau callback
    params["lr_min"] = params["lr_init"] * trial.suggest_float(
        name="lr_min_ratio", 
        low=0.025, 
        high=0.3,
        step=0.025
    )
    
    # batch size    
    params["batch_size"] = trial.suggest_int(
        name="batch_size", 
        low=64, 
        high=512, 
        step=64
    )
    
    return params
