# %% [code]
### NOTE: THESE SCRIPTS WERE RUN IN THE KAGGLE KERNEL. DATA IS NOT AVAILABLE IN THE GITHUB PAGE
### tutorial help was from https://www.kaggle.com/amyjang/monet-cyclegan-tutorial

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kaggle_datasets import KaggleDatasets
import tensorflow_addons as tfa

strategy = tf.distribute.get_strategy()

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)

AUTOTUNE = tf.data.experimental.AUTOTUNE
    
print(tf.__version__)

# %% [code]

# Create a set for monet labels and monet paintings
monet_set = []
monet_labels = []
# Iterate through monet jpg folder and use PIL to extract the image. Append the label and the image
# to their respective arrays
for dirname, _, filenames in os.walk('/kaggle/input/gan-getting-started/monet_jpg'):
    for filename in filenames:
        monet_set.append(PIL.Image.open(os.path.join(dirname, filename)))
        monet_labels.append(1)

# Create a 4D stack of images (#images, 256 width, 256 height, 3 channels)
monet_set = np.stack(monet_set)
monet_set = np.float32(monet_set)
monet_labels = np.array(monet_labels)
print(monet_set.shape)
print(monet_labels.shape)
# Print shapes for verification on notebook

# Create a set for photo labels and photos
photo_set = []
photo_labels = []
# Iterate through monet jpg folder and use PIL to extract the image. Append the label and the image
# to their respective arrays
for dirname, _, filenames in os.walk('/kaggle/input/gan-getting-started/photo_jpg'):
    for filename in filenames:
        photo_set.append(PIL.Image.open(os.path.join(dirname, filename)))
        photo_labels.append(0)

# Create a 4D stack of images (#images, 256 width, 256 height, 3 channels)
photo_set = np.stack(photo_set)
photo_set = photo_set[0:3000]
photo_set = np.float32(photo_set)
photo_labels = np.array(photo_labels)
photo_labels = photo_labels[0:3000]
print(photo_set.shape)
print(photo_labels.shape)
# Print shapes for verification on notebook




# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
# Alternative using numpy
# Plot figures to verify that images are correctly oriented.
# Scale images from 0->255 to 0->1
photo_set = photo_set / 255.
monet_set = monet_set / 255.
plt.subplot(121)
plt.title('Photo')
plt.imshow(photo_set[0])

plt.subplot(122)
plt.title('Monet')
plt.imshow(monet_set[0])

# %% [code]
# Create helper functions that help with the generators upsampling and downsampling
def downsample(filters,size, apply_instancenorm=True):
    # create a sequential block
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    result = keras.Sequential()
    # add a conv2D block with stride=2 to downsample 
    result.add(layers.Conv2D(filters, size, strides=2, padding = 'same'))
    # add an activation function and batchnorm
    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))
    result.add(layers.LeakyReLU())
    return result

def upsample(filters,size,dropout = 0):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    # create a sequential block
    result = keras.Sequential()
    # add a conv2D block with stride=2 to downsample 
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding = 'same'))
    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))
    # add dropout layer
    result.add(layers.Dropout(dropout))
    # add an activation function
    result.add(layers.LeakyReLU())
    return result

# %% [code]
# Create the generator Class consistent with the Pix2Pix CycleGan network architecture
def Gen():
    inputs = layers.Input(shape=[256,256,3])

    # bs = batch size
    down_stack = [
        downsample(64, 4, apply_instancenorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, dropout=0.2), # (bs, 2, 2, 1024)
        upsample(512, 4, dropout=0.2), # (bs, 4, 4, 1024)
        upsample(512, 4, dropout=0.2), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    x_res = []
    # propagate x through the downsampling stage.
    # keep track of the residual x's for each stage
    # to concatenate during the upsampling
    for down in down_stack:
        x = down(x)
        x_res.append(x)
    # propagate x through the upsampling stage.
    # Add x to the corresponding residual
    for upsamp, res in zip(up_stack, reversed(x_res[:-1])):
        x = upsamp(x)
        x = layers.Concatenate()([x, res])
    # send x through the final layer
    x = last(x)
    return keras.Model(inputs=inputs, outputs=x)
    
print(Gen().summary())

# %% [code]
# Note: the pix2pix network doesn't have a single unit output instead
# it uses a receptive field. To stay nearly consistent with the paper,
# the best receptive field tested was 70x70 for a 256x256 image with 
# zero padding. 16x16 was slightly worse than 70x70, so a receptive
# field around 70x70 would be ideal. The receptive field used is 62x62
# src: https://arxiv.org/pdf/1611.07004.pdf - pix2pix

def Disc():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, 3], name='input_image')

    x = inp

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU()(norm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)

Disc().summary()

# %% [code]


# %% [code]
# Create the Pix2Pix class
class Pix2Pix(keras.Model):
    def __init__(
        self,
        # Initialize the models
        monet_generator,
        photo_generator,
        monet_discriminator,
        photo_discriminator,
        lambda_cycle=10,
    ):
        super(Pix2Pix, self).__init__()
        # Initialize the models
        self.m_gen = m_gen
        self.p_gen = p_gen
        self.m_disc = m_disc
        self.p_disc = p_disc
        self.lambda_cycle = lambda_cycle
    def compile(
        # Create the optimizers for the 
        # 4 networks and a loss function
        # for the generator, discriminator,
        # the cycle, and identity.
        self,
        m_gen_opt,
        p_gen_opt,
        m_disc_opt,
        p_disc_opt,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn
    ):
        # Create the optimizers for the 
        # 4 networks and a loss function
        # for the generator, discriminator,
        # the cycle, and identity.
        super(Pix2Pix, self).compile()
        self.m_gen_opt = m_gen_opt
        self.p_gen_opt = p_gen_opt
        self.m_disc_opt = m_disc_opt
        self.p_disc_opt = p_disc_opt
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
    def train_step(self, batch_data):
        with tf.GradientTape(persistent=True) as tape:
            # get the real photo and monet
            R_monet, R_photo = batch_data
        
            # Create a fake monet from a photo and a fake photo from monet
            F_monet = self.m_gen(R_photo, training=True)
            F_photo = self.p_gen(R_monet, training=True)
            # Create a photo from a real photo and a monet from a real monet (identity)
            I_monet = self.m_gen(R_monet, training=True)
            I_photo = self.p_gen(R_photo, training=True)
            # Cycle the fake monet and real monet back to itself
            C_monet = self.m_gen(F_photo, training=True)
            C_photo = self.p_gen(F_monet, training=True)
        
            # use discriminator on real photo and monet
            disc_R_monet = self.m_disc(R_monet, training=True)
            disc_R_photo = self.p_disc(R_photo, training=True)
        
            # use discriminator on fake photo and monet
            disc_F_monet = self.m_disc(F_monet, training=True)
            disc_F_photo = self.p_disc(F_photo, training=True)
        
        
            # discriminator loss functions based on real/fake photos/monet 
            monet_disc_loss = self.disc_loss_fn(disc_R_monet, disc_F_monet)
            photo_disc_loss = self.disc_loss_fn(disc_R_photo, disc_F_photo)
        
            # generator loss functions
            monet_gen_loss = self.gen_loss_fn(disc_F_monet)
            photo_gen_loss = self.gen_loss_fn(disc_F_photo)
        
            # cycle loss function
            total_cycle_loss = self.cycle_loss_fn(R_monet, C_monet, self.lambda_cycle) + self.cycle_loss_fn(R_photo, C_photo, self.lambda_cycle)
        
            # total generator loss
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + self.identity_loss_fn(R_monet, I_monet, self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(R_photo, I_photo, self.lambda_cycle)
        
        
        # Calculate the gradients for generator and discriminator
        monet_G_grad = tape.gradient(total_monet_gen_loss,
                                                  self.m_gen.trainable_variables)
        photo_G_grad = tape.gradient(total_photo_gen_loss,
                                                  self.p_gen.trainable_variables)

        monet_D_grad = tape.gradient(monet_disc_loss,
                                                      self.m_disc.trainable_variables)
        photo_D_grad = tape.gradient(photo_disc_loss,
                                                      self.p_disc.trainable_variables)

        # Apply the gradients to the trainable variables
        self.m_gen_opt.apply_gradients(zip(monet_G_grad,
                                                 self.m_gen.trainable_variables))

        self.p_gen_opt.apply_gradients(zip(photo_G_grad,
                                                 self.p_gen.trainable_variables))

        self.m_disc_opt.apply_gradients(zip(monet_D_grad,
                                                  self.m_disc.trainable_variables))

        self.p_disc_opt.apply_gradients(zip(photo_D_grad,
                                                  self.p_disc.trainable_variables))
        
        return {
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss,
            "identity_monet": self.identity_loss_fn(R_monet, I_monet, self.lambda_cycle),
            "identity_photo": self.identity_loss_fn(R_photo, I_photo, self.lambda_cycle),
            "cycle": total_cycle_loss
        }

# %% [code]
with strategy.scope():
    # define the discriminator loss function
    def discriminator_loss(real, generated):
        # find the discriminator loss for both the real and generated images
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)
        generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)
        # return the average loss
        avg_disc_loss = (real_loss + generated_loss)*0.5
        return avg_disc_loss
    # define the generator loss function
    def generator_loss(generated):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)
    # define the cycle loss function
    def calc_cycle_loss(R_image, C_image, LAMBDA):
        # get the mean of the difference between real and cycled image
        loss1 = tf.reduce_mean(tf.abs(R_image - C_image))
        return LAMBDA * loss1
    # define identity loss function
    def identity_loss(R_image, I_image, LAMBDA):
        # get the mean of the difference between the an image and the image going through
        # the generator that produces the same type of image
        loss = tf.reduce_mean(tf.abs(R_image - I_image))
        return LAMBDA * 0.5 * loss

# %% [code]
# create the models for the monet generator, photo generator
# monet discriminator, photo discriminator
with strategy.scope():
    m_gen = Gen()
    p_gen = Gen()
    m_disc = Disc()
    p_disc = Disc()

# Set the optimizers for the monet generator, photo generator
# monet discriminator, photo discriminator
with strategy.scope():
    monet_generator_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_generator_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    monet_discriminator_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_discriminator_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Create the cyclegan model and compile it with the correct parameters
with strategy.scope():
    cycle_gan_model = Pix2Pix(
        m_gen, p_gen, m_disc, p_disc
    )

    cycle_gan_model.compile(
        m_gen_opt = monet_generator_opt,
        p_gen_opt = photo_generator_opt,
        m_disc_opt = monet_discriminator_opt,
        p_disc_opt = photo_discriminator_opt,
        gen_loss_fn = generator_loss,
        disc_loss_fn = discriminator_loss,
        cycle_loss_fn = calc_cycle_loss,
        identity_loss_fn = identity_loss
    )

# %% [code]
# FOR NUMPY VERSION
photo_dset = tf.data.Dataset.from_tensor_slices((photo_set))
monet_dset = tf.data.Dataset.from_tensor_slices((monet_set))
photo_dset = photo_dset.batch(1)
monet_dset = monet_dset.batch(1)
cycle_gan_model.fit(
    tf.data.Dataset.zip((monet_dset, photo_dset)),
    epochs = 20
)

# %% [code]
_, ax = plt.subplots(10, 2, figsize=(12, 36))
for i, img in enumerate(photo_dset.take(10)):
    prediction = m_gen(img, training=False)[0].numpy()
    prediction = (prediction * 255.).astype(np.uint8)
    img = (img[0] * 255.).numpy().astype(np.uint8)

    ax[i, 0].imshow(img)
    ax[i, 1].imshow(prediction)
    ax[i, 0].set_title("Input Photo")
    ax[i, 1].set_title("Monet-esque")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")
plt.show()