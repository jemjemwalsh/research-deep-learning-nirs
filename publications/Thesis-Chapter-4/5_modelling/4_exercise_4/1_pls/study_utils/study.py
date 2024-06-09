import optuna

from .objective import MAX_EPOCHS


N_STARTUP_TRIALS = 250  # specify the number of trials to use random sampling before using the TPE algorithm
RANDOM_SEED = 42  # specify a random seed for reproducibility


def create_or_load_study(study_name: str):
    
    # create sampler
    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_SEED, 
        consider_endpoints=True, 
        multivariate=True,
        n_startup_trials=N_STARTUP_TRIALS
    )
    
    # create study
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///data/{study_name}/{study_name}.db",
        direction="minimize",
        sampler=sampler,
        load_if_exists=True
    )
    
    return study