from tensorflow.keras import layers
from tensorflow import keras
from networks.blocks import GenResnetBlock


def build_generator(latent_dim, num_classes=7):

    '''
    Generator Network
    '''
    latent_dims = latent_dim
    input_z_noise = keras.Input(shape = (latent_dims, ))
    input_label = keras.Input(shape = (num_classes, ))

    #init = keras.initializers.RandomNormal(stddev=0.02)
    #li = layers.Embedding(num_classes, 50)(input_label)
    #li = layers.Dense(12*12*1)(li)
    #li = layers.Reshape((12, 12, 1))(li)
    merge = layers.Concatenate()([input_z_noise, input_label])

    nodes = 8*8*1024
    x = layers.Dense(nodes)(merge)
    #x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((8, 8, 1024))(x)
    x = layers.Dropout(0.2)(x)

    # 1st Deconvolution Block
    x = GenResnetBlock(512, 5)(x)
    #x = layers.Conv2DTranspose(filters = 512, kernel_size = 5, padding = 'same', strides=2)(x) # (batch_size, 16, 16, 512)
    #x = layers.BatchNormalization(momentum = 0.8)(x)
    x = layers.ReLU()(x)

    # 2nd Deconvolution Block
    x = GenResnetBlock(256, 5)(x)
    #x = layers.Conv2DTranspose(filters = 256, kernel_size = 5, padding = 'same', strides=2)(x) # (batch_size, 32, 32, 256)
    #x = layers.BatchNormalization(momentum = 0.8)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)

    # 3rd Deconvolution Block
    x = GenResnetBlock(128, 5)(x)
    #x = layers.Conv2DTranspose(filters = 128, kernel_size = 5, padding = 'same', strides=2)(x) # (batch_size, 32, 32, 128)
    #x = layers.BatchNormalization(momentum = 0.8)(x)
    x = layers.ReLU()(x)

    # 4th Deconvolution Block
    x = GenResnetBlock(64, 5)(x)
    #x = layers.Conv2DTranspose(filters = 64, kernel_size = 5, padding = 'same', strides=2)(x) # (batch_size, 64, 64, 64)
    #x = layers.BatchNormalization(momentum = 0.8)(x)
    x = layers.ReLU()(x)



    dec = layers.Conv2DTranspose(filters = 3, kernel_size = 5, padding = 'same', strides=1, activation='tanh')(x) # (batch_size, 128, 128, 3)

    model = keras.Model(inputs = [input_z_noise, input_label], outputs = dec)
    return model
