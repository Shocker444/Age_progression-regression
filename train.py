# imports
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
from tensorflow.keras.utils import to_categorical
from Generator import build_generator
from Discriminator import build_discriminator
from Encoder import build_encoder
from model import GAN
from utils import l2_norm
from utils import Checkpoint_callback
from utils import Generate

image_path = r'C:\Users\SHOCKER\tensorflow_projects\basic_machine_learning\Generative_model_stuff\Age Progression\UTKFace'

# age_group = 0-10, 11-20, 21-30, 31-40, 41-50, 51-60, 61+
g1, image1 = [], []
g2, image2 = [], []
g3, image3 = [], []
g4, image4 = [], []
g5, image5 = [], []
g6, image6 = [], []
g7, image7 = [], []


def get_agegroup(image_path):
    for image in os.listdir(image_path):
        age = image.split('_')[0]
        if 0 <= int(age) <= 10 and len(g1) < 3000:
            g1.append(0)
            image1.append(str(image_path) + '/' + image)
        elif 11 <= int(age) <= 20 and len(g2) < 3000:
            g2.append(1)
            image2.append(str(image_path) + '/' + image)
        elif 21 <= int(age) <= 30 and len(g3) < 3000:
            g3.append(2)
            image3.append(str(image_path) + '/' + image)
        elif 31 <= int(age) <= 40 and len(g4) < 3000:
            g4.append(3)
            image4.append(str(image_path) + '/' + image)
        elif 41 <= int(age) <= 50 and len(g5) < 3000:
            g5.append(4)
            image5.append(str(image_path) + '/' + image)
        elif 51 <= int(age) <= 60 and len(g6) < 3000:
            g6.append(5)
            image6.append(str(image_path) + '/' + image)
        elif 61 <= int(age) <= 90 and len(g7) < 3000:
            g7.append(6)
            image7.append(str(image_path) + '/' + image)


get_agegroup(image_path)
group = g1 + g2 + g3 + g4 + g5 + g6 + g7
images = image1 + image2 + image3 + image4 + image5 + image6 + image7
latent_dim = 100

group_cat = to_categorical(np.array(group))
dataset = tf.data.Dataset.from_tensor_slices((images, group_cat))


def load_func(image, label):
    img = tf.io.read_file(image)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize(img, (128, 128))
    img = (img - 127.5) / 127.5
    img = tf.cast(tf.reshape(img, (128, 128, 3)), dtype=tf.float32)
    return img, label


train_ds = dataset.map(load_func)

batch_size = 4
buffer_size = 4000
train_ds = train_ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

discriminator = build_discriminator()
generator = build_generator(latent_dim)
encoder = build_encoder(latent_dim)

cGAN = GAN(discriminator=discriminator, generator=generator, encoder=encoder)

cGAN.compile(g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8),
             d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8),
             loss_fn=keras.losses.BinaryCrossentropy(),
             loss_fn2=l2_norm)

epochs = 50
for image, label in train_ds.take(1):
    seed = image[0:4, :, :, :]
    labels = label[0:4, :]

callbacks = [Generate(encoder, generator, seed, labels), Checkpoint_callback(encoder, generator, discriminator)]

cGAN.fit(train_ds, epochs=epochs, callbacks=callbacks)
