import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import Model as model
from tensorflow.keras import Input


'''
    HourGlass module에서 사용되는 convolution layer block입니다.
'''
class ConvBlock(model):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        
        if in_planes != out_planes:
            self.downsample = 1
            # Pytorch : 
            self.dsbn = layers.BatchNormalization()
            self.dsrl = layers.Activation('relu')
            self.dsconv = layers.Conv2D(out_planes, (1, 1), padding='same', strides=(1, 1), use_bias=False)
        else:
            self.downsample = 0
            
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(int(out_planes / 2), (3, 3), padding='same', strides=(1, 1), use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(int(out_planes / 4), (3, 3), padding='same', strides=(1, 1), use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(int(out_planes / 4), (3, 3), padding='same', strides=(1, 1), use_bias=False)
        
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
class HourGlass(model):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.layers_dict = dict()
        
        self._generate_network(self.depth)
        
    def _generate_network(self, level):
        self.layers_dict['b1_' + str(level)] = ConvBlock(self.features, self.features)
        self.layers_dict['b2_' + str(level)] = ConvBlock(self.features, self.features)
            
        if level > 1:
            self._generate_network(level - 1) 
        else:
            self.layers_dict['b2_plus_' + str(level)] = ConvBlock(self.features, self.features)
        
        self.layers_dict['b3_' + str(level)] = ConvBlock(self.features, self.features)
    
    def _call(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self.layers_dict['b1_' + str(level)](up1)
        
        # Lower branch
        low1 = layers.AveragePooling2D((2, 2), strides=(2, 2))(inp)
        low1 = self.layers_dict['b2_' + str(level)](low1)
        
        if level > 1:
            low2 = self._call(level - 1, low1)
        else:
            low2 = low1
            low2 = self.layers_dict['b2_plus_' + str(level)](low2)
            
        low3 = low2
        low3 = self.layers_dict['b3_' + str(level)](low3)
        
        up2 = layers.UpSampling2D((2, 2), interpolation='nearest')(low3)
        
        if up2.shape[1] != up1.shape[1]:
            up2 = layers.ZeroPadding2D(((1, 0), (1, 0)))(up2)
        
        return up1 + up2
        
    def call(self, x):
        return self._call(self.depth, x)


# Loss function
def HeatmapLoss(y_true, y_pred):
    l = ((y_pred - y_true)**2)
    l = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(l, 3), 2), 1)
    return l ## l of dim bsize

def NME(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))

def lr_scheduler(epoch, lr):
    if epoch == 20:
        return 1e-03
    if epoch == 35:
        return 1e-04
    if epoch == 50:
        return 1e-05
    return lr

def FacialLandmarkDetector(num_modules=4):
    layers_dict = dict()
    
    inputs = Input(shape=(256, 256, 3), batch_size=16, sparse=False, ragged=False)
    
    # base part
    conv1 = layers.Conv2D(64, (7, 7), padding='same', strides=2)
    bn1 = layers.BatchNormalization()
    
    conv2 = ConvBlock(64, 128)
    conv3 = ConvBlock(128, 128)
    conv4 = ConvBlock(128, 256)
    
    # Stacking part
    for hg_module in range(num_modules):
        layers_dict['m' + str(hg_module)] = HourGlass(1, 4, 256)
        layers_dict['top_m_' + str(hg_module)] = ConvBlock(256, 256)
        layers_dict['conv_last' + str(hg_module)] = layers.Conv2D(256, (1, 1), padding='valid', strides=1)
        layers_dict['bn_end' + str(hg_module)] = layers.BatchNormalization()
        layers_dict['l' + str(hg_module)] = layers.Conv2D(68, (1, 1), padding='valid', strides=1) # 16 heatmaps
        
        if hg_module < num_modules - 1:
            layers_dict['bl' + str(hg_module)] = layers.Conv2D(256, (1, 1), padding='valid', strides=1)
            layers_dict['al' + str(hg_module)] = layers.Conv2D(256, (1, 1), padding='valid', strides=1)

    # Call
    x = tf.nn.relu(bn1(conv1(inputs)))
    x = conv2(x)
    x = layers.AveragePooling2D((2, 2), strides=2)(x)
    x = conv3(x)
    x = conv4(x)
    
    previous = x
    outputs = []
    
    for i in range(num_modules):
        hg = layers_dict['m' + str(i)](previous)
        
        ll = hg
        ll = layers_dict['top_m_' + str(i)](ll)
        
        ll = tf.nn.relu(layers_dict['bn_end' + str(i)]
                        (layers_dict['conv_last' + str(i)](ll)))
        
        # Predict heatmaps
        tmp_out = layers_dict['l' + str(i)](ll)
        outputs.append(tmp_out)
        
        if i < num_modules - 1:
            ll = layers_dict['bl' + str(i)](ll)
            tmp_out_ = layers_dict['al' + str(i)](tmp_out)
            previous = previous + ll + tmp_out_
    
    # initial_learning_rate = 1e-03
    # decay_steps = 12.0
    # decay_rate = 0.2
    # learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps, decay_rate)
    # opt = tf.keras.optimizers.Adam(learning_rate_fn)
    opt = tf.keras.optimizers.RMSprop(
        learning_rate=1e-04, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
        name='RMSprop')
    model = models.Model(inputs=inputs, outputs=outputs[-1], name='FacialLandmarkDetector')
    model.compile(optimizer=opt, loss='mse')
    model.build((None, 256, 256, 3))
    model.summary()

    return model