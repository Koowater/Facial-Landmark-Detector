import argparse
import os
import random
import importlib
import datetime
import numpy as np
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from libs.dp import Dataset
from libs.utils import NME, LandmarkLoss

tf.config.run_functions_eagerly(True) 

# Fix the seed
def my_seed_everywhere(seed):
    random.seed(seed) # random
    np.random.seed(seed) # np
    os.environ["PYTHONHASHSEED"] = str(seed) # os
    tf.random.set_seed(seed) # tensorflow
my_seed = 42
my_seed_everywhere(my_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model',         dest='model')
parser.add_argument('--batch_size',    dest='batch_size',    default=8)
parser.add_argument('--hm_size',       dest='hm_size',       default=64)
parser.add_argument('--num_landmarks', dest='num_landmarks', default=68)
parser.add_argument('--img_res',       dest='img_res',       default=256)
parser.add_argument('--epochs',        dest='epochs',        default=120)
parser.add_argument('--num_gpu',       dest='num_gpu',       default='0')

os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().num_gpu
# print(tf.config.list_physical_devices('GPU'))

MODEL_NAME    = parser.parse_args().model
BATCH_SIZE    = parser.parse_args().batch_size
HM_SIZE       = parser.parse_args().hm_size
NUM_LANDMARKS = parser.parse_args().num_landmarks
IMG_RES       = parser.parse_args().img_res
EPOCHS        = int(parser.parse_args().epochs)
TOTAL_STEP    = np.ceil(3837 / BATCH_SIZE).astype(int)
EVAL_STEP     = np.ceil(600 / BATCH_SIZE).astype(int)

initial_learning_rate = 1e-04
# decay_steps = 10.0
# decay_rate = 5.

# Set Dataset
DATASET_DIR       = '../Datasets/300W_train/train.csv'
BASE_DIR          = '/media/cvmi-koo/HDD/Facial-Landmark-Detector/'
CP_DIR            = os.path.join(BASE_DIR, 'data','checkpoint')
dataset           = Dataset(IMG_RES, HM_SIZE, NUM_LANDMARKS, DATASET_DIR)
dataset_generator = dataset.tf_dataset_from_generator(BATCH_SIZE)
eval_dataset      = Dataset(IMG_RES, HM_SIZE, NUM_LANDMARKS, '../Datasets/300W/eval.csv')
eval_generator    = eval_dataset.tf_dataset_from_generator(BATCH_SIZE)

# Hyper Parameter
# learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps, decay_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
loss_fn   = NME
# checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(CP_DIR, os.path.join(MODEL_NAME, 'cp_{epoch}.ckpt')), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch', options=None)

# Landmark loss metric
lm_metric = LandmarkLoss(IMG_RES, HM_SIZE)

# Set Tensorboard directory
current_time   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
TB_DIR = os.path.join(BASE_DIR, 'data', 'tensorboard', MODEL_NAME + '_' + current_time)

# Load a Model
model_path = 'tasks.model_' + MODEL_NAME
SHG = importlib.import_module(model_path).StackedHourglassNetworks(4, 256, 68)
inputs = tf.keras.Input(shape=(256, 256, 3))
outputs = SHG(inputs)
model = importlib.import_module(model_path).FacialLandmarkDetector(
    inputs=inputs, outputs=outputs, lm_metric=lm_metric, 
    hm_size=HM_SIZE, batch_size=BATCH_SIZE, train_step=TOTAL_STEP, test_step=EVAL_STEP
)
model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())
print(f'model_path: {model_path}')

model.fit(dataset_generator.take(TOTAL_STEP), epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=eval_generator.take(EVAL_STEP), callbacks=[cp_callback, tf.keras.callbacks.TensorBoard(log_dir=TB_DIR, histogram_freq=1)])