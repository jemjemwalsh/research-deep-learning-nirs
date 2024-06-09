from optuna import Trial


def get_parameters(trial: Trial) -> dict:
    params = {}
    params["model"] = get_architecture_parameters(trial)
    params["training"] = get_training_hyperparameters(trial)
    return params


def get_architecture_parameters(trial: Trial) -> dict:
    
    params = {}
    
    params["conv_layers"] = trial.suggest_int(
        name="conv_layers",
        low=1,
        high=3,
        step=1
    )

    params["conv_kernels"] = {}
    params["conv_kernel_sizes"] = {}
    for layer in range(1, params["conv_layers"]+1):
        params["conv_kernels"][layer] = trial.suggest_int(
            name=f"conv_{layer}_kernels", 
            low=1, 
            # high=8,
            high=31,
            step=2
        )
        params["conv_kernel_sizes"][layer] = trial.suggest_int(
            name=f"conv_{layer}_kernel_size", 
            low=1,
            high=125,
            step=4
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
