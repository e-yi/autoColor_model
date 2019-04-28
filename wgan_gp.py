# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np


MASK = 0
BATCH_SIZE = 32


def customLoss(y_true, y_pred):
    # y_true = K.print_tensor(y_true, message='y_true = ')
    # y_pred = K.print_tensor(y_pred, message='y_pred = ')
    # y_true_f = K.flatten(y_true)
    # y_pred_f = K.flatten(y_pred)
    mask = 1 - K.cast(K.equal(y_true, MASK), K.floatx())
    return K.mean(K.square(y_pred - y_true) * mask, axis=-1)


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""

    def _merge_function(self, inputs):
        alpha = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGANGP:
    def __init__(self):
        self.X_val = None

        self.img_rows = 1
        self.img_cols = 5
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # condition input
        c_img = Input(shape=self.img_shape)
        # Generate image based of noise (fake sample)
        fake_img = self.generator(c_img)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_img, c_img],
                                  outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=optimizer,
                                  loss_weights=[1, 1, 10])
        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=self.img_shape)
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(inputs=z_gen,
                                     outputs=[valid, img])
        self.generator_model.compile(loss=[self.wasserstein_loss, customLoss],
                                     optimizer=optimizer,
                                     loss_weights=[1, 10])

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(32 * 5 * 5))
        # model.add(Dropout(0.25))
        model.add(Reshape((5, 5, 32)))
        # model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        # model.add(UpSampling2D())
        model.add(Conv2D(16, kernel_size=3, padding="same"))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=(5, 1), padding="valid"))
        model.add(Activation("tanh"))

        model.summary()

        condition = Input(shape=self.img_shape)
        img = model(condition)

        return Model(condition, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=(1, 3), padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add((Dropout(0.25)))
        model.add(Conv2D(32, (1, 3), padding='valid'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add((Dropout(0.25)))
        model.add(Conv2D(32, (1, 3), padding='valid'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add((Dropout(0.25)))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size, sample_interval=50):

        # # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        #
        # # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)

        a = np.loadtxt('datasets/mturkData.csv', delimiter=',')
        b = a.reshape((-1, 3, 5))
        c = b.transpose((0, 2, 1))
        X_train = np.expand_dims(c, 1)
        del a, b, c

        X_train = X_train * 2 - 1  # [-1,1]
        np.random.shuffle(X_train)
        sp = X_train.shape[0] * 8 // 10
        X_train, self.X_val = X_train[:sp], X_train[sp:]

        def getRandomMaskedColor(colors):
            assert colors.shape[1:] == self.img_shape
            colors = np.copy(colors)
            for i, color in enumerate(colors):
                rm_choice = np.random.choice(5, 5, p=[0.1, 0.2, 0.3, 0.3, 0.1])  # 可能有0到5个颜色被删除
                color[0, rm_choice] = MASK
            return colors

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty
        # fake_img = np.zeros((batch_size, *self.img_shape))

        print("start training")

        for epoch in range(epochs):

            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                # noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                condition = getRandomMaskedColor(imgs)
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, condition],
                                                          [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            _, g_loss, c_loss = self.generator_model.train_on_batch(condition,
                                                                    [valid, condition])

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f, mse loss: %f]" %
                  (epoch, d_loss[0], g_loss, c_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        colorIdx = np.random.choice(self.X_val.shape[0], 4, replace=False)
        colors = self.X_val[colorIdx]
        colors = np.copy(colors)
        for i, color in enumerate(colors):
            rm = np.random.choice(5, i + 1, replace=False)
            color[0, rm] = MASK

        gen_imgs = self.generator.predict(colors)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        colors = 0.5 * colors + 0.5

        r, c = 2, 4
        fig, axs = plt.subplots(r, c)
        for j in range(c):
            axs[0, j].imshow(gen_imgs[j])
            axs[1, j].imshow(colors[j])
            axs[0, j].axis('off')
            axs[1, j].axis('off')
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()

    def save(self, folder='./models/', gname='generator.h5'):
        self.generator.save(folder+gname)

if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=20000, batch_size=BATCH_SIZE, sample_interval=500)
    wgan.save()
