import tensorflow as tf


def create_model(
    input_dims: int,
    conv_layers: int,
    conv_kernels: dict,
    conv_kernel_sizes: dict,
    dense_layers: int,
    dense_units: dict,
    dense_dropout: dict,
    reg_beta: float,
    learning_rate: float,
    random_seed: int = None
) -> tf.keras.models.Sequential:

    # clear keras session
    tf.keras.backend.clear_session()

    # weights L2 regularization (all layers)
    kernel_reg = tf.keras.regularizers.l2(reg_beta)

    # weights initialisation (all layers)
    kernel_init = tf.keras.initializers.he_normal(seed=random_seed)

    # model architecture
    model = tf.keras.Sequential()
    
    # input layer
    if conv_layers > 0:
        model.add(
            tf.keras.layers.Reshape(
                name="input",
                target_shape=(input_dims, 1),
                input_shape=(input_dims,)
            )
        )
    else:
        model.add(
            tf.keras.layers.Input(
                name="input",
                shape=(input_dims,),
            )
        )
    
    # convolutional layers
    for layer in range(1, conv_layers+1):
        model.add(
            tf.keras.layers.Conv1D(
                name=f"conv1d_{layer}",
                filters=conv_kernels[layer],
                kernel_size=conv_kernel_sizes[layer],
                strides=1,
                padding="same",
                kernel_initializer=kernel_init,
                kernel_regularizer=kernel_reg,
                activation="elu",
            )
        )
        model.add(
            tf.keras.layers.BatchNormalization(
                name=f"batchnorm_{layer}"
            )
        )
    
    # flatter layer
    if conv_layers > 0:
        model.add(
            tf.keras.layers.Flatten(
                name="flatten"
            )
        )
    
    # dense layers
    for layer in range(1, dense_layers+1):
        model.add(
            tf.keras.layers.Dense(
                name=f"dense_{layer}",
                units=dense_units[layer],
                kernel_initializer=kernel_init,
                kernel_regularizer=kernel_reg,
                activation="elu"
            )
        )
        if layer != dense_layers:
            model.add(
                tf.keras.layers.Dropout(
                    name=f"dropout_{layer}",
                    rate=dense_dropout[layer]
                    
                )
            )
    
    # output layer
    model.add(     
        tf.keras.layers.Dense(
            name="output",
            units=1,
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_reg,
            activation="linear"
        )
    )
    
    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="mse",
        metrics=["mse"]
    )
    
    return model
