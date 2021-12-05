import tensorflow as tf
import numpy as np

OUTPUT_CHANNELS = 68

# Encoder
def conv2d_block(input_tensor, n_filters, kernel_size=3):
    '''
    Add 2 convolutional layers with the parameters
    '''
    x = input_tensor
    for i in range(2):
        x = tf.keras.layers.Conv2D(
            filters=n_filters, kernel_size=kernel_size, kernel_initializer='he_normal', 
            activation='relu', padding='same')(x)
    return x

def encoder_block(inputs, n_filters=64, pool_size=(2,2), dropout=0.3):
    '''
    Add 2 convolutional blocks and then perform down sampling on output of convolutions
    '''
    f = conv2d_block(inputs, n_filters)
    p = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(f)
    p = tf.keras.layers.Dropout(dropout)(p)
    return f, p

def encoder(inputs):
    '''
    defines the encoder or downsampling path.
    '''
    f1, p1 = encoder_block(inputs, n_filters=64)
    f2, p2 = encoder_block(p1, n_filters=128)
    f3, p3 = encoder_block(p2, n_filters=256)
    f4, p4 = encoder_block(p3, n_filters=512)
    return p4, (f1, f2, f3, f4)

# Bottlenect
def bottleneck(inputs):
    bottle_neck = conv2d_block(inputs, n_filters=1024)
    return bottle_neck

# Decoder
def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3):
    '''
    defines the one decoder block of the UNet
    '''
    u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides, padding='same')(inputs)
    c = tf.keras.layers.concatenate([u, conv_output])
    c = tf.keras.layers.Dropout(dropout)(c)
    c = conv2d_block(c, n_filters)
    return c

def decoder(inputs, convs, output_channels):
    '''
    Defines the decoder of the UNet chaining together 4 decoder blocks.
    '''
    f1, f2, f3, f4 = convs
    c6 = decoder_block(inputs, f4, n_filters=512, kernel_size=3, strides=2)
    c7 = decoder_block(c6, f3, n_filters=256, kernel_size=3, strides=2)
    c8 = decoder_block(c7, f2, n_filters=128, kernel_size=3, strides=2)
    c9 = decoder_block(c8, f1, n_filters=64, kernel_size=3, strides=2)
    outputs = tf.keras.layers.Conv2D(output_channels, 1)(c9)
    return outputs

# putting it all together
# UNet
def FacialLandmarkDetector():
    '''
    Defines the UNet by connecting the encoder, bottleneck and decoder
    '''
    inputs = tf.keras.layers.Input(shape=(256, 256 ,3))
    encoder_output, convs = encoder(inputs)
    bottle_neck = bottleneck(encoder_output)
    outputs = decoder(bottle_neck, convs, OUTPUT_CHANNELS)
    model = tf.keras.Model(inputs, outputs)

    initial_learning_rate = 1e-03
    decay_steps = 15.0
    decay_rate = 0.5
    learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate, decay_steps, decay_rate)
    opt = tf.keras.optimizers.RMSprop(learning_rate_fn)
    # opt = tf.keras.optimizers.RMSprop(
    #     learning_rate=1e-04, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
    #     name='RMSprop')
    model.compile(optimizer=opt, loss='mse')
    model.build((None, 256, 256, 3))
    # model.summary()
    return model