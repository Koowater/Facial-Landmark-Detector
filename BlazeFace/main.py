import time
import argparse
import os

from network import FeatureExtractor
from loss import total_loss, conf_metric, box_metric

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, MaxPooling2D, ReLU, Input, BatchNormalization, Reshape, Concatenate
from tensorflow.keras.activations import relu

class BlazeFace():
    def __init__(self, config):
        self.input_shape = config.input_shape
        self.feature_extractor = FeatureExtractor()
        self.n_boxes = [2, 6]
        self.model = self.build_model()

        if config.train:
            self.batch_size = config.batch_size
            self.epochs = config.epochs

        self.checkpoint_path = config.checkpoint_path
        self.total_data = config.total_data
    
    def build_model(self):
        model = self.feature_extractor

        # class는 얼굴 단 하나밖에 없기 때문에 class confidence는 필요하지 않다.

        # bounding box confidence prediction
        bb_16_conf = Conv2D(filters=self.n_boxes[0] * 1,
                                            kernel_size=3,
                                            padding='same',
                                            activation='sigmoid',
                                            name='bb_16_conf')(model.output[0])
        bb_16_conf_reshaped = Reshape((16**2 * self.n_boxes[0], 1))(bb_16_conf)

        bb_8_conf = Conv2D(filters=self.n_boxes[1] * 1,
                           kernel_size=3,
                           padding='same', 
                           activation='sigmoid',
                           name='bb_8_conf')(model.output[1])

        bb_8_conf_reshaped = Reshape((8**2 * self.n_boxes[1], 1))(bb_8_conf)

        conf_of_bb = Concatenate(axis=1)([bb_16_conf_reshaped, bb_8_conf_reshaped])

        # bounding box location prediction
        bb_16_loc = Conv2D(filters=self.n_boxes[0] * 4,
                             kernel_size=3,
                             padding='same',
                             name='bb_16_loc')(model.output[0])
        
        bb_16_loc_reshaped = Reshape((16**2 * self.n_boxes[0], 4))(bb_16_loc)

        bb_8_loc = Conv2D(filters=self.n_boxes[1] * 4,
                          kernel_size=3,
                          padding='same',
                          name='bb_8_loc')(model.output[1])

        bb_8_loc_reshaped = Reshape((8**2 * self.n_boxes[1], 4))(bb_8_loc)

        loc_of_bb = Concatenate(axis=1)([bb_16_loc_reshaped, bb_8_loc_reshaped])

        output_combined = Concatenate(axis=-1)([conf_of_bb, loc_of_bb])

        return CustomModel(model.input, output_combined)

    # def decode_boxes(self, raw_boxes, anchors):
    #     # model의 predictions를 실제 anchor boxes로 변환한다.
    #     # box는 x_center, y_center, w, h로 표현된다.
    #     # anchor shape: [896, 4] (x_center, y_center, width, height), normalized coordinates.
    #     # width, height는 항상 1이다.

    #     x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
    #     y_center = raw_boxes[..., 1] / self.x_scale * anchors[:, 3] + anchors[:, 1]
    #     w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
    #     h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

    #     boxes = tf.zeros_like(raw_boxes)

    #     boxes[..., 0] = y_center - h / 2.
    #     boxes[..., 1] = x_center - w / 2.
    #     boxes[..., 2] = y_center + h / 2.
    #     boxes[..., 3] = x_center + w / 2.

    #     # *** apply anchor boxes ***
    #     for k in range(6):
    #         offset = 4 + k * 2
    #         keypoint_x = raw_boxes[..., offset    ] / self.x_scale * anchors[:, 2] + anchors[:, 0]
    #         keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
    #         boxes[..., offset    ] = keypoint_x
    #         boxes[..., offset + 1] = keypoint_y

    #     return boxes

class CustomModel(Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    
    # hyperparameters
    args.add_argument('--input_shape', type=int, default=[128, 128, 3])
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--epochs', type=int, default=1000)
    args.add_argument('--total_data', type=int, default=2625)
    args.add_argument('--train', type=bool, default=True)
    args.add_argument('--checkpoint_path', type=str, default="./")
    args.add_argument('--dataset_dir', type=str, default="./")
    args.add_argument('--label_path', type=str, default="./")

    config = args.parse_args()

    blazeface = BlazeFace(config)

    model = blazeface.build_model()
    # AMSgrad
    #   Adam에서 활용하던 Exponential Averaging 방식을 버리고 
    #   Second Momentum 값들 중 가장 큰 값을 활용한다.
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
    model.compile(loss=total_loss, optimizer=opt, metrics=[conf_metric, box_metric])
    model.summary()