from tasks.model_TCN import FacialLandmarkDetector
from libs.dp import Dataset
import tensorflow as tf
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print(tf.config.list_physical_devices('GPU'))

model = FacialLandmarkDetector()

# checkpoint

checkpoint_path = "data/checkpoint_model_TCN/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq='epoch')

model.save_weights(checkpoint_path.format(epoch=0))

IMG_RES = 256
HMS_RES = 64
NUM_LANDMARKS = 68
BATCH_SIZE = 16
DATASET_DIR = '../Data/300W_train/train.csv'

dataset = Dataset(IMG_RES, HMS_RES, NUM_LANDMARKS, DATASET_DIR)
dataset_generator = dataset.tf_dataset_from_generator(BATCH_SIZE)

hist = model.fit(dataset_generator.take(240), callbacks=[cp_callback], epochs=100)

with open('hist_SHG_TCN.pickle', 'wb') as f:
    pickle.dump(hist, f)