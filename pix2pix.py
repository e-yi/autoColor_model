import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, GaussianNoise, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import keras.backend as K
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

def customLoss(y_true,y_pred):
    return K.mean(K.abs(y_pred - y_true)*(1-K.cast(K.equal(y_true,-1), K.floatx())), axis=-1)

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 1 ##  the input image is 256*1
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        #continue training
        c_flag = False
        
        # Configure data loader
        self.dataset_name = 'adobeColor_s' ##
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        #patch = int(self.img_rows / 2**4) ##
        self.disc_patch = (1, int(self.img_cols / 2**4), 1) ##

        # Number of filters in the first layer of G and D
        self.gf = 128
        self.df = 128

        optimizer = Adam(0.0002, 0.5)
        
        if c_flag:
            self.discriminator = load_model('discriminator.h5')
        else:
            # Build and compile the discriminator
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='mse',
                optimizer=optimizer,
                metrics=['accuracy'])
        #

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        if c_flag:
            self.generator = load_model('generator.h5')
        else:
            self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', customLoss],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=(1,4), bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=(1,4), dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=(1,2))(layer_input) ##
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u, training=True)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)

        # Upsampling
        u0 = deconv2d(d5, d4, self.gf*8, dropout_rate = 0.5)
        u1 = deconv2d(u0, d3, self.gf*8, dropout_rate = 0.5)
        u2 = deconv2d(u1, d2, self.gf*4)
        u3 = deconv2d(u2, d1, self.gf*2)

        u4 = UpSampling2D(size=(1,2))(u3) ##
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=(2,4), bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-2)([img_A, img_B])
        
        d0 = AveragePooling2D((1,8),2,'same')(combined_imgs)
        d1 = d_layer(d0 , self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=(1,4), strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50, save_interval=50):
        print('start training')
        start_time = datetime.datetime.now()
        
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        
        for epoch in range(epochs):
            #print('epoch' + str(epoch))
            
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                #print(batch_i)
                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)
                #print('img_A',np.shape(imgs_A))
                #print('img_B',np.shape(imgs_B))  #img_B (batch_size, 1, 256, 3)
                #print('fake_A',np.shape(fake_A))
                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # -----------------
                #  Train Generator
                # -----------------
                
                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
                
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))
                
                early_stop = {0.6:0,0.7:0,0.8:0}
                for p in early_stop:
                    if p < d_loss[1] and not early_stop[p]:
                        early_stop[p] = 1
                        self.save(gname='generator_%d.h5'%int(p*100),
                            dname='discriminator_%d.h5'%int(p*100))
                
                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                
                if batch_i % save_interval == 0:
                    self.save()

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A],axis=1)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = np.clip(gen_imgs,0,1)

        fig, axs = plt.subplots(3, 1)
        for i in range(3):
            axs[i].imshow(gen_imgs[i])
            axs[i].axis('off')
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()

    def save(self,dname='discriminator.h5',gname='generator.h5'):
        self.discriminator.save(dname)
        self.generator.save(gname)

if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=1, batch_size=8, sample_interval=200, save_interval=1000)
    gan.save()
