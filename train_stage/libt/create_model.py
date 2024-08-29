import tensorflow as tf

def create_autoencoder(input_dim: int) -> tf.keras.Model:
    """
    Create an autoencoder model.

    Parameters:
    - input_dim: int, the dimensionality of the input data.

    Returns:
    - tf.keras.Model: The constructed autoencoder model.
    """
    autoencoder = tf.keras.models.Sequential([
        tf.keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim, )),
        tf.keras.layers.Dense(16, activation='elu'),
        tf.keras.layers.Dense(12, activation='elu'),
        tf.keras.layers.Dense(8, activation='elu'),
        tf.keras.layers.Dense(4, activation='elu'),
        tf.keras.layers.Dense(2, activation='elu'),
        tf.keras.layers.Dense(4, activation='elu'),
        tf.keras.layers.Dense(8, activation='elu'),
        tf.keras.layers.Dense(12, activation='elu'),
        tf.keras.layers.Dense(16, activation='elu'),
        tf.keras.layers.Dense(input_dim, activation='elu')
    ])
    autoencoder.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    return autoencoder
