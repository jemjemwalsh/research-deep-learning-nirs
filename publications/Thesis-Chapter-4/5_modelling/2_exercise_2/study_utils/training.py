import tensorflow as tf

from optuna import Trial
from optuna.integration import TFKerasPruningCallback
from tensorflow_addons.callbacks import TQDMProgressBar

MIN_EPOCHS = 200

def create_callbacks(
    model_directory: str, 
    model_name:str, 
    lr_min: float, 
    trial: Trial = None,
    model_checkpoint: bool = False,
    save_weights_only: bool = True
) -> list:
    
    callbacks = []
    
    # reduce learning rate dynamically
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        patience=25,
        factor=0.5,
        min_lr=lr_min,
        monitor="val_loss",
        verbose=0
    )
    callbacks.append(reduce_lr)

    # early stopping criteria
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,  
        patience=50,
        mode="min",
        restore_best_weights=True,
        start_from_epoch=(MIN_EPOCHS-50) # subtract patience
    )
    callbacks.append(early_stopping)

    # progress bar during training
    progress_bar = TQDMProgressBar(show_epoch_progress=False)
    callbacks.append(progress_bar)
    
    # save model artifact
    if model_checkpoint:
        checkpoint = CustomModelCheckpoint(
            filepath=f"{model_directory}/ckpt/{model_name}.model.keras", 
            verbose=0, 
            save_best_only=True,
            save_weights_only=save_weights_only
        )
        callbacks.append(checkpoint)
    
    if trial is not None:
        # stop training unpromising models early
        pruning = TFKerasPruningCallback(
            trial=trial, 
            monitor="val_loss"
        )
        callbacks.append(pruning)
    
    return callbacks


class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """Custom ModelCheckpoint to save models only after a certain number of epochs (to save resources). 
    This is a custom version of the ModelCheckpoint() callback that only saves the best models if they are trained for more than 150 epochs.
    By using this, the less promising models that hyperband will discard, are not saved to disk. This leads to less saved models in the disk..
    """
    
    def __init__(self, *args, **kwargs):
        super(CustomModelCheckpoint, self).__init__(*args, **kwargs)

    ## redefine the save so it only activates after 200 epochs
    def on_epoch_end(self, epoch, logs=None):
        if epoch >= MIN_EPOCHS: 
            super(CustomModelCheckpoint, self).on_epoch_end(epoch, logs)