import time
import argparse

from network import FeatureExtractor

import tensorflow as tf
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

        # bounding box confidence prediction
        bb_16_conf = Conv2D(filters=self.n_boxes[0] * 1,
                                            kernel_size=3,
                                            padding='same',
                                            activation='sigmoid')(model.output[0])
        bb_16_conf_reshaped = Reshape((16**2 * self.n_boxes[0], 1))(bb_16_conf)

        bb_8_conf = Conv2D(filters=self.n_boxes[1] * 1,
                           kernel_size=3,
                           padding='same', 
                           activation='sigmoid')(model.output[1])

        bb_8_conf_reshaped = Reshape((8**2 * self.n_boxes[1], 1))(bb_8_conf)

        conf_of_bb = Concatenate(axis=1)([bb_16_conf_reshaped, bb_8_conf_reshaped])

        # bounding box location prediction
        bb_16_loc = Conv2D(filters=self.n_boxes[0] * 4,
                             kernel_size=3,
                             padding='same')(model.output[0])
        
        bb_16_loc_reshaped = Reshape((16**2 * self.n_boxes[0], 4))(bb_16_loc)

        bb_8_loc = Conv2D(filters=self.n_boxes[1] * 4,
                          kernel_size=3,
                          padding='same')(model.output[1])

        bb_8_loc_reshaped = Reshape((8**2 * self.n_boxes[1], 4))(bb_8_loc)

        loc_of_bb = Concatenate(axis=1)([bb_16_loc_reshaped, bb_8_loc_reshaped])

        output_combined = Concatenate(axis=-1)([conf_of_bb, loc_of_bb])

        return tf.keras.models.Model(model.input, output_combined)

    def train(self):
        opt = tf.keras.optimizers.Adam(amsgrad=True) # amsgrad 사용 여부는 실험 후 정확도를 확인해보자.
        model = self.model
        model.compile(loss=['categorical_crossentropy', 'MAE'], optimizer=opt) # MAE 대신 smooth_l1_loss도 적용해보자.

        tb = tf.keras.callbacks.TensorBoard(log_dir="./logs/", update_freq='batch')

        STEP_SIZE_TRAIN = self.total_data // self.batch_size

        t0 = time.time()

        data_gen = dataloader(config.dataset_dir, config.label_path, self.batch_size)
        
        for epoch in range(self.epochs):
            t1 = time.time()
            hist = model.fit_generator(generator=data_gen,
                                       steps_per_epoch=STEP_SIZE_TRAIN,
                                       initial_epoch=epoch,
                                       callbacks=[reduce_lr, tb],
                                       verbose=1,
                                       shuffle=True)

            t2 = time.time()

            print(hist.history)

            print(f'Training time for one epoch : {t2 - t1:.3}')

            if epoch % 100 == 0:
                model.save_weights(os.path.join(config.checkpoint_path, str(epoch)))

        print(f'Total training time : {time.time() - t0:.3}')

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
    model.summary()