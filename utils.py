import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# loss function
def l2_norm(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true-y_pred)))

# Callback to monitor images
class Generate(keras.callbacks.Callback):
    def __init__(self, encoder, generator, seed, labels):
        super(Generate, self).__init__()
        self.encoder = encoder
        self.seed = seed
        self.labels = labels
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        encoded = self.encoder(self.seed)
        prediction = self.generator([encoded, self.labels])

        plt.figure(figsize=(16, 16))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(prediction[i])
            plt.axis('off')


        plt.savefig(f'image at epoch {epoch + 1}.png')

# save model weights after every epoch incase of interuption
class Checkpoint_callback(keras.callbacks.Callback):
    def __init__(self, encoder, generator, discriminator):
        super(Checkpoint_callback, self).__init__()
        self.encoder = encoder
        self.discriminator = discriminator
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        self.encoder.save('encoder.h5')
        self.generator.save('generator.h5')
        self.discriminator.save('discriminator.h5')