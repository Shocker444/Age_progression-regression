from keras import backend as K
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_addons as tfa



def expand_label_input(x):
    x = K.expand_dims(x, axis=1)
    x = K.expand_dims(x, axis=1)
    x = K.tile(x, [1, 64, 64, 1])
    return x


def build_discriminator(num_classes=7):

    """
    Discriminator Network
    """

    input_shape = (128, 128, 3)
    label_shape = (num_classes, )
    image_input = keras.Input(shape = input_shape)
    label_input = keras.Input(shape = label_shape)

    #init = keras.initializers.RandomNormal(stddev=0.02)
    #const = ClipConstraint(0.01)

    #li = layers.Embedding(num_classes, 50)(label_input)
    #li = layers.Dense(48*48*1, activation='relu')(li)
    #li = layers.Reshape((48, 48, 1))(li)


    # 1st Convolution Block
    #x = DiscResnetBlock(16, 5)(image_input)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(16, kernel_size = 5, strides = 2, padding = 'same'))(image_input) # (batch_size, 64, 64, 16)
    #x = Residual(16, 5, add_bn=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    li = layers.Lambda(expand_label_input)(label_input)
    x = layers.Concatenate()([x, li]) # (batch_size, 16, 16, n+16)

   # 2nd Convolution Block
    #x = DiscResnetBlock(32, 5)(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(32, kernel_size = 5, strides=(2, 2), padding='same'))(x) # (batch_size, 32, 32, 32)
    #x = Residual(32, 5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    # 3rd Convolution Block
    #x = DiscResnetBlock(64, 5)(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(64, kernel_size = 5, strides=(2, 2), padding='same'))(x) # (batch_size, 16, 16, 64)
    #x = Residual(64, 5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    # 4th Convolution Block
    #x = DiscResnetBlock(128, 5)(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(128, kernel_size = 5, strides=(2, 2), padding='same'))(x) # (batch_size, 8, 8, 128)
    #x = Residual(128, 5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(1, activation='sigmoid')(x)
    #x = layers.Conv2D(1, kernel_size = 5, strides = 1, activation = 'sigmoid')(x)
    model = keras.Model(inputs = [image_input, label_input], outputs = x)
    return model
