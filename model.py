import tensorflow as tf
from tensorflow import keras


class GAN(keras.Model):
    def __init__(self, discriminator, generator, encoder):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.encoder = encoder

    def compile(self, d_optimizer, g_optimizer, loss_fn, loss_fn2):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.loss_fn2 = loss_fn2

    def discriminator_loss(self, real, fake):
        real_loss = self.loss_fn(tf.ones_like(real), real)
        fake_loss = self.loss_fn(tf.zeros_like(fake), fake)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake):
        fake_loss = self.loss_fn(tf.ones_like(fake), fake)
        return fake_loss

    def train_step(self, batch):
        tr_image, tr_label = batch

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            enc = self.encoder(tr_image)
            generated_image = self.generator([enc, tr_label], training=True)

            real = self.discriminator([tr_image, tr_label], training=True)
            fake = self.discriminator([generated_image, tr_label], training=True)

            gen_loss = self.generator_loss(fake)

            l2_norm = self.loss_fn2(generated_image, tr_image)
            disc_loss = self.discriminator_loss(real, fake)

            total_loss = 100 * l2_norm + gen_loss
        gen_grad = gen_tape.gradient(total_loss, self.generator.trainable_variables)
        disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))

        return {"disc_loss": disc_loss, "gen_loss": gen_loss, "l2_loss": l2_norm, "total_loss": total_loss}
