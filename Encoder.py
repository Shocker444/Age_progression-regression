from tensorflow.keras import layers
from tensorflow import keras


def build_encoder(latent_dim):
    """
    Encoder Network
    """

    input_layer = keras.Input(shape=(128, 128, 3))

    # 1st Convolutional Block
    enc = layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(input_layer)  # (batch_size, 64, 64, 64)
    # enc = layers.BatchNormalization()(enc)
    enc = layers.ReLU()(enc)

    # 2nd Convolutional Block
    enc = layers.Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(enc)  # (batch_size, 32, 32, 128)
    # enc = layers.BatchNormalization()(enc)
    enc = layers.ReLU()(enc)

    # 3rd Convolutional Block
    enc = layers.Conv2D(filters=256, kernel_size=5, strides=2, padding='same')(enc)  # (batch_size, 16, 16, 256)
    # enc = layers.BatchNormalization()(enc)
    enc = layers.ReLU()(enc)

    # 4th Convolutional Block
    enc = layers.Conv2D(filters=512, kernel_size=5, strides=2, padding='same')(enc)  # (batch_size, 8, 8, 512)
    # enc = layers.BatchNormalization()(enc)
    enc = layers.ReLU()(enc)

    # Flatten layer
    enc = layers.Flatten()(enc)  # (batch_size, 8*8*512)

    # 1st Fully Connected Layer
    enc = layers.Dense(50)(enc)  # (batch_size, 50)
    # enc = layers.BatchNormalization()(enc)
    enc = layers.ReLU()(enc)

    # 2nd Fully Connected Layer
    enc = layers.Dense(latent_dim)(enc)

    # Create a model
    model = keras.Model(inputs=input_layer, outputs=enc)
    return model
