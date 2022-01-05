import importlib
MODEL_NAME = 'UNet'

from libs.dp import Dataset
import tensorflow as tf
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print(tf.config.list_physical_devices('GPU'))

model_lib = importlib.import_module('tasks.model_' + MODEL_NAME)
model = model_lib.FacialLandmarkDetector()

# checkpoint

checkpoint_path = os.path.join(f'data/checkpoint_model_{MODEL_NAME}', 'cp-{epoch:04d}.ckpt')
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq='epoch')

model.save_weights(checkpoint_path.format(epoch=0))

IMG_RES = 256
HMS_RES = 256
NUM_LANDMARKS = 68
BATCH_SIZE = 16
DATASET_DIR = '../Data/300W_train/train.csv'

dataset = Dataset(IMG_RES, HMS_RES, NUM_LANDMARKS, DATASET_DIR)
dataset_generator = dataset.tf_dataset_from_generator(BATCH_SIZE)

hist = model.fit(dataset_generator.take(240), callbacks=[cp_callback], epochs=100)

with open(f'data/hist_{MODEL_NAME}.pickle', 'wb') as f:
    pickle.dump(hist.history, f)