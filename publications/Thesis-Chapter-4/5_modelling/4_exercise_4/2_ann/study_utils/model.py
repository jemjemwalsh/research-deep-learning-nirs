import tensorflow as tf


def create_model(
    input_dims: int,
    dense_layers: int,
    activation_function: str,
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
    
    model.add(
        tf.keras.layers.Input(
            name="input",
            shape=(input_dims,),
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
                activation=activation_function
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
