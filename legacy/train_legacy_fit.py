import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
import random
import datetime
import importlib
import io
from tqdm import tqdm
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from libs.dp import Dataset
from libs import eval
from libs.eval import plot_to_image
from libs.utils import NME, lr_scheduler, HeatmapLoss, my_loss

from legacy.model_legacy import FAN

np.set_printoptions(precision=6, suppress=True)

# Fix the seed
def my_seed_everywhere(seed):
    random.seed(seed) # random
    np.random.seed(seed) # np
    os.environ["PYTHONHASHSEED"] = str(seed) # os
    tf.random.set_seed(seed) # tensorflow
my_seed = 42
my_seed_everywhere(my_seed)

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

# MODELS = ['SHG', 'SHG_TCN', 'SHG_TCN_he', 'UNet', 'VGG16', 'ResNet52']

parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model')
parser.add_argument('--num_gpu', dest='num_gpu')
parser.add_argument('--hm_size', dest='hm_size', default=64)

os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().num_gpu
# print(tf.config.list_physical_devices('GPU'))

# Set Const Variable
MODEL_NAME    = parser.parse_args().model
BATCH_SIZE    = 8
HM_SIZE       = parser.parse_args().hm_size
NUM_LANDMARKS = 68
IMG_RES       = 256
EPOCHS        = 120
TOTAL_STEP    = np.ceil(3837 / BATCH_SIZE).astype(int)

initial_learning_rate = 1e-04
decay_steps = 10.0
decay_rate = 5.

# model_path = 'tasks.model_' + MODEL_NAME
# model = importlib.import_module(model_path).FacialLandmarkDetector()
# print(f'model_path: {model_path}')
model = FAN(4)
opt = tf.keras.optimizers.RMSprop(
    learning_rate=1e-04, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop')
model.build(input_shape=((None, 256, 256, 3)))
model.compile(loss=NME, optimizer=opt)


# Set Dataset
DATASET_DIR    = '../Datasets/300W_train/train.csv'
BASE_DIR       = '/media/cvmi-koo/HDD/Facial-Landmark-Detector/'
CP_DIR         = os.path.join(BASE_DIR, 'data','checkpoint')

# Set tensorboard
current_time   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
TB_DIR         = os.path.join(BASE_DIR, 'data', 'tensorboard', MODEL_NAME + '_' + current_time)
summary_writer = tf.summary.create_file_writer(TB_DIR)

dataset           = Dataset(IMG_RES, HM_SIZE, NUM_LANDMARKS, DATASET_DIR)
dataset_generator = dataset.tf_dataset_from_generator(BATCH_SIZE)
eval_dataset      = Dataset(256, HM_SIZE, 68, '../Datasets/300W/eval.csv')
EVAL_STEPS        = np.ceil(600 / BATCH_SIZE).astype(int)

# Hyper Parameter
# learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps, decay_rate)
optimizer        = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
loss_fn          = my_loss(True, None, True)

print(f'\nEpochs: {EPOCHS}, Total step: {TOTAL_STEP}, Batch size: {BATCH_SIZE}')

cp_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(CP_DIR, 'cp_{epoch}.ckpt'), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch', options=None)


model.fit(dataset_generator.take(TOTAL_STEP), epochs=20, batch_size=BATCH_SIZE, callbacks=[cp_callback])
