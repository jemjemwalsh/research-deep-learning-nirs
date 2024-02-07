import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

from livelossplot import PlotLossesKerasTF
from sklearn.cross_decomposition import PLSRegression


# set random seeds for Python, NumPy, and TensorFlow
SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)


def build_pls(x_train: np.ndarray, y_train: np.ndarray) -> tuple[PLSRegression, dict]:

    # latent variables
    lvs = 11

    # train the model
    training_time = dt.datetime.now()
    model = PLSRegression(n_components=lvs, scale=False)
    model.fit(X=x_train, Y=y_train)
    training_time = dt.datetime.now() - training_time
    meta_data = {"time": training_time}

    return model, meta_data


def create_callbacks(monitor_val_loss: bool = True) -> list:

    # visualise tracked metrics in real time during training
    plot_losses = PlotLossesKerasTF()

    # reduce learning rate dynamically
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        patience=25,
        factor=0.5,
        min_lr=10**-6,
        monitor="val_loss" if monitor_val_loss else "loss",
        verbose=0
    )

    # early stopping criteria
    if monitor_val_loss:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=10**-5,
            patience=50,
            mode="auto",
            restore_best_weights=True
        )
    else:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=10**(-2.5),
            patience=50,
            mode="auto",
            restore_best_weights=True
        )

    # return [plot_losses, reduce_lr, early_stopping]
    return [reduce_lr, early_stopping]


def plot_model_history(history: dict):
    plt.figure(figsize=(8, 4))
    plt.plot(history["loss"], label="Calibration set loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Tunning set loss")
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    ax2 = plt.gca().twinx()
    ax2.plot(history["lr"], color="r", ls="--")
    ax2.set_ylabel("learning rate", color="r")
    plt.tight_layout()
    plt.show()


def build_ann(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray = None,
    y_val: np.ndarray = None,
) -> tuple[tf.keras.models.Sequential, dict]:

    # clear keras session
    tf.keras.backend.clear_session()

    # number of sample to train for each pass into the ANN
    batch = 128

    # learning rate
    lr = 0.01*batch/256.

    # input layer dimensions
    input_dims = x_train.shape[1]

    # determine if validation data available
    validation_data = (x_val, y_val) if x_val is not None and y_val is not None else None
    monitor_val_loss = x_val is not None and y_val is not None

    # model architecture
    model = tf.keras.models.Sequential(
        layers=[
            # single hidden layer with 5 neurons
            tf.keras.layers.Dense(units=5, activation="sigmoid", input_dim=input_dims),

            # linear output layer
            tf.keras.layers.Dense(units=1, activation="linear"),
        ]
    )

    # compile model with adam optimiser
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mse"]
    )

    # train the model
    training_time = dt.datetime.now()
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch,
        epochs=10000,
        validation_data=validation_data,
        callbacks=create_callbacks(monitor_val_loss=monitor_val_loss),
        verbose=1
    )
    training_time = dt.datetime.now() - training_time
    meta_data = {
        "time": training_time,
        "epochs": len(history.history["loss"]),
        "history": history
    }
    plot_model_history(history.history)

    return model, meta_data


def build_cnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray = None,
    y_val: np.ndarray = None,
) -> tuple[tf.keras.models.Sequential, dict]:

    # clear keras session
    tf.keras.backend.clear_session()

    # number of sample to train for each pass into the ANN
    batch = 128

    # learning rate
    lr = 0.01*batch/256.

    # input layer dimensions
    input_dims = x_train.shape[1]

    # L2 regularisation parameter
    reg_beta = 0.011
    beta = reg_beta/2.

    # weights L2 regularization
    kernel_reg = tf.keras.regularizers.l2(beta)

    # weights initialisation (used for all layers for simplicity)
    kernel_init = tf.keras.initializers.he_normal(seed=SEED_VALUE)

    # determine if validation data available
    validation_data = (x_val, y_val) if x_val is not None and y_val is not None else None
    monitor_val_loss = x_val is not None and y_val is not None

    # model architecture
    model = tf.keras.Sequential(
        layers=[
            tf.keras.layers.Reshape(
                target_shape=(input_dims, 1),
                input_shape=(input_dims,)
            ),
            tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=21,
                strides=1,
                padding="same",
                kernel_initializer=kernel_init,
                kernel_regularizer=kernel_reg,
                activation="elu",
                input_shape=(input_dims, 1)
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=36,
                kernel_initializer=kernel_init,
                kernel_regularizer=kernel_reg,
                activation="elu"
            ),
            tf.keras.layers.Dense(
                units=18,
                kernel_initializer=kernel_init,
                kernel_regularizer=kernel_reg,
                activation="elu"
            ),
            tf.keras.layers.Dense(
                units=12,
                kernel_initializer=kernel_init,
                kernel_regularizer=kernel_reg,
                activation="elu"
            ),
            tf.keras.layers.Dense(
                units=1,
                kernel_initializer=kernel_init,
                kernel_regularizer=kernel_reg,
                activation="linear"
            ),
        ]
    )

    # compile model with adam optimiser
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mse"]
    )

    # train the model
    training_time = dt.datetime.now()
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch,
        epochs=750,
        validation_data=validation_data,
        callbacks=create_callbacks(monitor_val_loss=monitor_val_loss),
        verbose=1
    )
    training_time = dt.datetime.now() - training_time
    meta_data = {
        "time": training_time,
        "epochs": len(history.history["loss"]),
        "history": history
    }
    plot_model_history(history.history)

    return model, meta_data
