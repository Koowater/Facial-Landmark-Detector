from re import U
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.keras import Input

'''
    HourGlass module에서 사용되는 convolution layer block입니다.
'''
class ConvBlock(Layer):
    def __init__(self, in_planes, out_planes, name=None):
        if isinstance(name, type(None)):
            super(ConvBlock, self).__init__()
        else:
            super(ConvBlock, self).__init__(name=name)
        
        if in_planes != out_planes:
            self.downsample = 1
            # Pytorch : 
            self.dsbn = layers.BatchNormalization()
            self.dsrl = layers.Activation('relu')
            self.dsconv = layers.Conv2D(out_planes, (1, 1), padding='same', strides=(1, 1), use_bias=False, kernel_initializer='he_normal')
        else:
            self.downsample = 0
            
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(int(out_planes / 2), (3, 3), padding='same', strides=(1, 1), use_bias=False, kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(int(out_planes / 4), (3, 3), padding='same', strides=(1, 1), use_bias=False, kernel_initializer='he_normal')
        self.bn3 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(int(out_planes / 4), (3, 3), padding='same', strides=(1, 1), use_bias=False, kernel_initializer='he_normal')
        
    def call(self, input_tensor):
        residual = input_tensor
        
        out1 = self.bn1(input_tensor)
        out1 = tf.nn.relu(out1)
        out1 = self.conv1(out1)
        
        out2 = self.bn2(out1)
        out2 = tf.nn.relu(out2)
        out2 = self.conv2(out2)
        
        out3 = self.bn3(out2)
        out3 = tf.nn.relu(out3)
        out3 = self.conv3(out3)
        
        # residual
        out3 = tf.concat([out1, out2, out3], 3)        

        if self.downsample == 1:
            residual = self.dsbn(residual)
            residual = self.dsrl(residual)
            residual = self.dsconv(residual)

        #out3 += residual
        
        return tf.math.add(out3, residual)    


'''
    ConvBlock을 사용해 FAN을 구성하는 HourGlass module을 구현합니다.
'''
class HourGlass(Layer):
    def __init__(self, num_modules, depth, num_features, name=None):
        if isinstance(name, type(None)):
            super(HourGlass, self).__init__()
        else:
            super(HourGlass, self).__init__(name=name)
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.layers_dict = dict()
        
        self._generate_network(self.num_modules, self.depth)
        
    def _generate_network(self, num_modules, level):
        for conv_module in range(num_modules):
            self.layers_dict['b1_' + str(level) + '_' + str(conv_module)] = ConvBlock(self.features, self.features)
            self.layers_dict['b2_' + str(level) + '_' + str(conv_module)] = ConvBlock(self.features, self.features)
            
        if level > 1:
            self._generate_network(self.num_modules, level - 1) 
        else:
            for conv_module in range(num_modules):
                self.layers_dict['b2_plus_' + str(level) + '_' + str(conv_module)] = ConvBlock(self.features, self.features)
        
        for conv_module in range(num_modules):
            self.layers_dict['b3_' + str(level) + '_' + str(conv_module)] = ConvBlock(self.features, self.features)
        self.layers_dict['bn'] = layers.BatchNormalization()
        self.layers_dict['conv2dtranspose'] = layers.Conv2DTranspose(256, (4, 4), strides=2, padding='same', kernel_initializer='he_normal')
    
    def _call(self, num_modules, level, inp):
        # Upper branch
        up1 = inp
        for conv_module in range(num_modules):
            up1 = self.layers_dict['b1_' + str(level) + '_' + str(conv_module)](up1)
        
        # Lower branch
        low1 = layers.AveragePooling2D((2, 2), strides=(2, 2))(inp)
        
        for conv_module in range(num_modules):
            low1 = self.layers_dict['b2_' + str(level) + '_' + str(conv_module)](low1)
        
        if level > 1:
            low2 = self._call(self.num_modules, level - 1, low1)
        else:
            low2 = low1
            for conv_module in range(num_modules):
                low2 = self.layers_dict['b2_plus_' + str(level) + '_' + str(conv_module)](low2)
            
        low3 = low2
        for conv_module in range(num_modules):
            low3 = self.layers_dict['b3_' + str(level) + '_' + str(conv_module)](low3)

        # up2 = layers.UpSampling2D((2, 2), interpolation='nearest')(low3)
        up2 = self.layers_dict['conv2dtranspose'](low3)

        if up2.shape[1] != up1.shape[1]:
            up2 = layers.ZeroPadding2D(((1, 0), (1, 0)))(up2)

        up2 = self.layers_dict['bn'](up2)
        up2 = tf.nn.relu(up2)

        return up1 + up2
        
    def call(self, x):
        return self._call(self.num_modules, self.depth, x)


# Loss function
def HeatmapLoss(y_true, y_pred):
    l = ((y_pred - y_true)**2)
    l = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(l, 3), 2), 1)
    return l ## l of dim bsize

def NME(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))

def lr_scheduler(epoch, lr):
    if epoch == 20:
        return 1e-04
    if epoch == 35:
        return 1e-05
    if epoch == 50:
        return 1e-06
    return lr

# def FacialLandmarkDetector(num_modules=4):
#     num_parts = 68
#     layers_dict = dict()
    
#     inputs = Input(shape=(256, 256, 3), batch_size=None, sparse=False, ragged=False)
    
#     #base part
#     conv1 = layers.Conv2D(64, (7, 7), padding='same', strides=2, kernel_initializer='he_normal')
#     bn1 = layers.BatchNormalization()
    
#     conv2 = ConvBlock(64, 128)
#     conv3 = ConvBlock(128, 128)
#     conv4 = ConvBlock(128, 256)
    
#     # Stacking part
#     for hg_module in range(num_modules):
#         layers_dict['m' + str(hg_module)] = HourGlass(1, 4, 256)
#         layers_dict['top_m_' + str(hg_module)] = ConvBlock(256, 256)
#         layers_dict['conv_last' + str(hg_module)] = layers.Conv2D(256, (1, 1), padding='valid', strides=1, kernel_initializer='he_normal')
#         layers_dict['bn_end' + str(hg_module)] = layers.BatchNormalization()
#         layers_dict['l' + str(hg_module)] = layers.Conv2D(num_parts, (1, 1), padding='valid', strides=1, kernel_initializer='he_normal') # 1 heatmaps
        
#         if hg_module < num_modules - 1:
#             layers_dict['bl' + str(hg_module)] = layers.Conv2D(256, (1, 1), padding='valid', strides=1, kernel_initializer='he_normal')
#             layers_dict['al' + str(hg_module)] = layers.Conv2D(256, (1, 1), padding='valid', strides=1, kernel_initializer='he_normal')

#     # Call
#     x = tf.nn.relu(bn1(conv1(inputs)))
#     x = conv2(x)
#     x = layers.AveragePooling2D((2, 2), strides=2)(x)
#     x = conv3(x)
#     x = conv4(x)
    
#     previous = x
#     outputs = []
    
#     for i in range(num_modules):
#         hg = layers_dict['m' + str(i)](previous)
        
#         ll = hg
#         ll = layers_dict['top_m_' + str(i)](ll)
        
#         ll = tf.nn.relu(layers_dict['bn_end' + str(i)]
#                         (layers_dict['conv_last' + str(i)](ll)))
        
#         # Predict heatmaps
#         tmp_out = layers_dict['l' + str(i)](ll)
#         outputs.append(tmp_out)
        
#         if i < num_modules - 1:
#             ll = layers_dict['bl' + str(i)](ll)
#             tmp_out_ = layers_dict['al' + str(i)](tmp_out)
#             previous = previous + ll + tmp_out_
    
#     initial_learning_rate = 1e-03
#     decay_steps = 15.0
#     decay_rate = 0.1
#     learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
#         initial_learning_rate, decay_steps, decay_rate)
#     # opt = tf.keras.optimizers.Adam(learning_rate_fn)
#     opt = tf.keras.optimizers.RMSprop(learning_rate_fn)
#     # opt = tf.keras.optimizers.RMSprop(
#     #     learning_rate=1e-04, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
#     #     name='RMSprop')
#     model = models.Model(inputs=inputs, outputs=outputs[-1], name='FacialLandmarkDetector_TCN')
#     model.compile(optimizer=opt, loss='mse')
#     model.build((None, 256, 256, 3))
#     # model.summary()

#     return model

class FacialLandmarkDetector(Model):
    def __init__(self, num_modules=4, num_parts=68):
        super(FacialLandmarkDetector, self).__init__()

        self.num_modules = num_modules
        # base part
        self.conv1 = layers.Conv2D(64, (7, 7), padding='same', strides=2, kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)
        
        self.avg = layers.AveragePooling2D((2, 2), strides=2)

        self.m = []
        self.top_m = []
        self.conv_last = []
        self.bn_end = []
        self.l = []
        self.bl = []
        self.al = []

        for hg_module in range(self.num_modules):
            self.m.append('')
            self.top_m.append('')
            self.conv_last.append('')
            self.bn_end.append('')
            self.l.append('')
            self.bl.append('')
            self.al.append('')

        # Stacking part
        for hg_module in range(self.num_modules):
            self.m[hg_module] = HourGlass(1, 4, 256, 'm' + str(hg_module))
            self.top_m[hg_module] = ConvBlock(256, 256, 'top_m_' + str(hg_module))
            self.conv_last[hg_module] = layers.Conv2D(256, (1, 1), padding='valid', strides=1, kernel_initializer='he_normal', 
                                name='conv_last' + str(hg_module))
            self.bn_end[hg_module] = layers.Conv2D(256, (1, 1), padding='valid', strides=1, kernel_initializer='he_normal',
                                name='bn_end' + str(hg_module))
            self.l[hg_module] = layers.Conv2D(num_parts, (1, 1), padding='valid', strides=1, kernel_initializer='he_normal',
                                name='l' + str(hg_module))

            if hg_module < self.num_modules - 1:
                self.bl[hg_module] = layers.Conv2D(256, (1, 1), padding='valid', strides=1, kernel_initializer='he_normal',
                                    name='bl' + str(hg_module))
                self.al[hg_module] = layers.Conv2D(256, (1, 1), padding='valid', strides=1, kernel_initializer='he_normal',
                                    name='al' + str(hg_module))

        self.m = tuple(self.m)
        self.top_m = tuple(self.top_m)
        self.conv_last = tuple(self.conv_last)
        self.bn_end = tuple(self.bn_end)
        self.l = tuple(self.l)
        self.bl = tuple(self.bl)
        self.al = tuple(self.al)
    
    @tf.function
    def call(self, inputs):
        x = tf.nn.relu(self.bn1(self.conv1(inputs)))
        x = self.conv2(x)
        x = self.avg(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        previous = x
        self.outputs = []
        
        for i in range(self.num_modules):
            hg = self.m[i](previous)
            
            ll = hg
            ll = self.top_m[i](ll)
            
            ll = tf.nn.relu(self.bn_end[i]
                            (self.conv_last[i](ll)))
            
            # Predict heatmaps
            tmp_out = self.l[i](ll)
            self.outputs.append(tmp_out)
            
            if i < self.num_modules - 1:
                ll = self.bl[i](ll)
                tmp_out_ = self.al[i](tmp_out)
                previous = previous + ll + tmp_out_

        return self.outputs[-1]

    