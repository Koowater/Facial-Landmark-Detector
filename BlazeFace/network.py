import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, MaxPooling2D, ReLU, BatchNormalization, Reshape, Concatenate
from tensorflow.python.keras.activations import relu

class BlazeBlock(tf.keras.Model):
    def __init__(self, channel, stride=1, name='BlazeBlock'):
        super(BlazeBlock, self).__init__(name=name)

        self.stride = stride
        if self.stride == 2:
            self.pooling = MaxPooling2D(strides=self.stride)

        self.dwconv = DepthwiseConv2D(kernel_size=5, strides=(self.stride, self.stride), padding='same')
        self.pwconv = Conv2D(filters=channel, kernel_size=(1, 1))

        self.bn = BatchNormalization()

    def call(self, input_tensor):
        residual = input_tensor
        if self.stride == 2:
            residual = self.pooling(residual)

        x = self.dwconv(input_tensor)
        x = self.pwconv(x)
        x = self.bn(x)

        if x.shape[-1] != residual.shape[-1]:
            residual = tf.concat([residual, tf.zeros_like(residual)], axis=-1, name='channel_padding')
            
        x = relu(x + residual)
        return x

class DoubleBlazeBlock(tf.keras.Model):
    def __init__(self, channel, stride=1, name='DoubleBlazeBlock'):
        super(DoubleBlazeBlock, self).__init__(name=name)

        self.stride = stride
        if self.stride == 2:
            self.pooling = MaxPooling2D(strides=self.stride)

        self.dwconv_0 = DepthwiseConv2D(kernel_size=5, strides=(stride, stride), padding='same')
        self.pwconv_0 = Conv2D(filters=24, kernel_size=(1, 1))
        self.dwconv_1 = DepthwiseConv2D(kernel_size=5, strides=(1, 1), padding='same')
        self.pwconv_1 = Conv2D(filters=channel, kernel_size=(1, 1))

        self.bn_0 = BatchNormalization()
        self.bn_1 = BatchNormalization()

    def call(self, input_tensor):
        residual = input_tensor
        if self.stride == 2:
            residual = self.pooling(residual)

        x = self.dwconv_0(input_tensor)
        x = self.pwconv_0(x)
        x = self.bn_0(x)
        x = relu(x)
        x = self.dwconv_1(x)
        x = self.pwconv_1(x)
        x = self.bn_1(x)

        if x.shape[-1] != residual.shape[-1]:
            residual = tf.concat([residual, tf.zeros_like(residual)], axis=-1, name='channel_padding')

        x = relu(x + residual)
        return x


def FeatureExtractor():
    inputs = Input(shape=(128, 128, 3), name='Input')
    conv = Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='same', name='Conv')(inputs)
    x = ReLU(name='ReLU')(conv)
    SBB_0 = BlazeBlock(24,    name='BlazeBlock_0')(x)
    SBB_1 = BlazeBlock(24,    name='BlazeBlock_1')(SBB_0)
    SBB_2 = BlazeBlock(48, 2, name='BlazeBlock_2')(SBB_1)
    SBB_3 = BlazeBlock(48,    name='BlazeBlock_3')(SBB_2)
    SBB_4 = BlazeBlock(48,    name='BlazeBlock_4')(SBB_3)
    DBB_0 = DoubleBlazeBlock(96, 2, name='DoubleBlazeBlock_0')(SBB_4)
    DBB_1 = DoubleBlazeBlock(96,    name='DoubleBlazeBlock_1')(DBB_0)
    DBB_2 = DoubleBlazeBlock(96,    name='DoubleBlazeBlock_2')(DBB_1)
    DBB_3 = DoubleBlazeBlock(96, 2, name='DoubleBlazeBlock_3')(DBB_2)
    DBB_4 = DoubleBlazeBlock(96,    name='DoubleBlazeBlock_4')(DBB_3)
    DBB_5 = DoubleBlazeBlock(96,    name='DoubleBlazeBlock_5')(DBB_4)

    model = tf.keras.Model(inputs=inputs, outputs=[DBB_2, DBB_5], name='FeatureExtractor')
    
    return model

if __name__ == '__main__':
    model = FeatureExtractor()
    model.summary()














