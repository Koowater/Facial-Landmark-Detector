import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

class Residual(layers.Layer):
    def __init__(self, in_ch, out_ch):
        super(Residual, self).__init__()
        
        self.relu = layers.Activation('relu')

        if in_ch != out_ch:
            self.downsample = True
            self.downblock = [
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Conv2D(out_ch, (1, 1), padding='same', strides=1, use_bias=False)
            ]
        else:
            self.downsample = False

        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(int(out_ch / 2), kernel_size=(3, 3), strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(int(out_ch / 2), kernel_size=(3, 3), strides=1, padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(out_ch, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)

    def call(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample == True:
            for layer in self.downblock:
                residual = layer(residual)
        
        return out + residual

class Hourglass(layers.Layer):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = layers.MaxPool2D((2, 2), strides=(2, 2))
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = layers.Conv2DTranspose(256, (2, 2), strides=2, padding='valid')

    def call(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)

        return up1 + up2

class StackedHourglassNetworks(Model):
    def __init__(self, nstack, in_ch, out_ch, bn=False, increase=0):
        super(StackedHourglassNetworks, self).__init__()

        self.nstack = nstack
        self.pre = [
            layers.Conv2D(64, (7, 7), 2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            Residual(64, 128),
            layers.MaxPool2D((2, 2), strides=(2, 2)),
            Residual(128, 128),
            Residual(128, in_ch)
        ]

        self.hgs = [
                Hourglass(4, in_ch, bn, increase) for i in range(nstack)
        ]

        self.features = [
                [
                    Residual(in_ch, in_ch),
                    layers.Conv2D(in_ch, (1, 1), use_bias=False),
                    layers.BatchNormalization(),
                    layers.Activation('relu')
                    ]
             for i in range(nstack)
        ]

        self.outs = [layers.Conv2D(out_ch, (1, 1), 1, use_bias=False) for i in range(nstack)]
        self.merge_features = [layers.Conv2D(in_ch, (1, 1), (1, 1), use_bias=False) for i in range(nstack - 1)]
        self.merge_preds = [layers.Conv2D(in_ch, (1, 1), (1, 1), use_bias=False) for i in range(nstack - 1)]
        self.nstack = nstack

    def call(self, imgs):
        x = imgs
        for layer in self.pre:
            x = layer(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = hg
            for layer in self.features[i]:
                feature = layer(feature)
            # feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return combined_hm_preds[-1]


class FacialLandmarkDetector(Model):
    def __init__(self, lm_metric, hm_size, batch_size, train_step, test_step, **kwargs):
        super(FacialLandmarkDetector, self).__init__(**kwargs)
        self.hm_size = hm_size
        self.batch_size = batch_size
        self.num_train_step = train_step
        self.num_test_step = test_step
        self.loss_tracker = keras.metrics.MeanSquaredError(name='mse')
        self.lm_metric = lm_metric

    def train_step(self, data):
        self.lm_metric.reset_states()
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) # Forward pass
            # Compute out own loss
            loss = self.compiled_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.loss_tracker.update_state(y, y_pred)
        self.lm_metric.update_state(y, y_pred)
        return {'mse': self.loss_tracker.result(), 'lm_loss': self.lm_metric.result()}
    
    def test_step(self, data):
        self.lm_metric.reset_states()
        x, y = data
        y_pred = self(x, training=False) # Forward pass

        self.loss_tracker.update_state(y, y_pred)
        self.lm_metric.update_state(y, y_pred)
        return {'mse': self.loss_tracker.result(), 'lm_loss': self.lm_metric.result()}