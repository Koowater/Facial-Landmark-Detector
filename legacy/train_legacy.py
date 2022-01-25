import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 5})
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
parser.add_argument('--num_gpu', dest='num_gpu', default='0')
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
# TOTAL_STEP = 1

initial_learning_rate = 1e-04
decay_steps = 10.0
decay_rate = 5.

# Hyper Parameter
# learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps, decay_rate)
optimizer        = tf.keras.optimizers.Adam(learning_rate=0.0001)
# loss_fn          = my_loss(True, None, True)
loss_fn = NME

model_path = 'tasks.model_' + MODEL_NAME
FAN = importlib.import_module(model_path).FAN(4, 256, 68)
inputs = tf.keras.Input(shape=(256, 256, 3))
outputs = FAN(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
print(f'model_path: {model_path}')
model.build(input_shape=((None, 256, 256, 3)))
model.compile(loss=NME, optimizer=optimizer)
model.summary()

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


print(f'\nEpochs: {EPOCHS}, Total step: {TOTAL_STEP}, Batch size: {BATCH_SIZE}')

figure = plt.figure(figsize=(1, 1))

for epoch in range(EPOCHS):
    print(f"\nStart of epoch {epoch}")
    # print(f"Learning rate: {optimizer.lr(epoch)}")
    # Train
    for step, (images, labels) in enumerate(dataset_generator.take(TOTAL_STEP)):
        with tf.GradientTape() as tape:
            # print(images.shape, labels.shape, images.numpy().max(), images.numpy().min())
            # test_image = images.numpy()[0]
            # test_label = labels.numpy()[0,:,:,:3]
            # test_image = cv2.resize(test_image, (64, 64))
            # test_image = (test_image + test_label ) /2
            # cv2.imshow('test', test_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            full_predictions = model(images)
            predictions = full_predictions[:,-1,:,:,:]
            loss_value = loss_fn(labels, predictions)
        # print(labels.shape)
        # print(predictions.shape)
        # print(loss_value.shape)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        cur_time = datetime.datetime.now().strftime("%X")
        print(f'[{cur_time}] Epoch: {epoch:3} | Step: {step+1:<3} / {TOTAL_STEP} | Loss: {tf.reduce_mean(loss_value):.10f}')
    
    swap_pred = np.swapaxes(np.array([labels[0]]), 0, 3)
    # for hm_idx, hm in enumerate(swap_pred):
    #     plt.subplot(10, 7, hm_idx + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(hm, cmap=plt.cm.plasma)
    print(swap_pred[0].shape)
    with summary_writer.as_default():
        plt.imshow(swap_pred[0])
        tf.summary.image('label', plot_to_image(figure), step=epoch)

    # Checkpoint
    checkpoint_path = os.path.join(CP_DIR, f'model_{MODEL_NAME}', f'cp-{epoch:04}.ckpt')
    model.save_weights(checkpoint_path)
    print(f'Checkpoint is saved. - [{checkpoint_path}]')

    # Log(loss)
    with summary_writer.as_default():
        # tf.summary.scalar('learning rate', optimizer.lr(epoch), step=epoch)
        tf.summary.scalar('loss', tf.reduce_mean(loss_value), step=epoch)

    # Evaluate
    error = 0.
    eval_dataset_generator = eval_dataset.tf_eval_dataset_from_generator(BATCH_SIZE)
    pbar = tqdm(eval_dataset_generator.take(EVAL_STEPS), desc='eval_dataset', total=EVAL_STEPS)
    for img, kps, center, scale in pbar:
        preds = model(img) 
        kps = kps * 256 / HM_SIZE
        for i, pred in enumerate(preds):
            pred_kps = eval.parse_heatmap(pred) * 256 / HM_SIZE
            _error = eval.l2_distance(pred_kps, kps[i])
            error += _error
            pbar.set_postfix({'Error': _error})
    
    preds_arr = preds.numpy()

    for idx, pred in enumerate(preds_arr):
        example_img = np.reshape(img[idx], (-1, 256, 256, 3))
        tf.summary.image('image_' + str(idx), example_img, step=epoch)
        swap_pred = np.swapaxes(np.array([pred]), 0, 3)
        figure = plt.figure(figsize=(7, 10))

        for hm_idx, hm in enumerate(swap_pred):
            plt.subplot(10, 7, hm_idx + 1, title=str(hm_idx))
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(hm, cmap=plt.cm.plasma)
        plt.subplot(10, 7, 69, title='sum')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(swap_pred.sum(axis=0), cmap=plt.cm.plasma)
        with summary_writer.as_default():
            tf.summary.image('predictions_' + str(idx), plot_to_image(figure), step=epoch)

    error = error / (BATCH_SIZE * EVAL_STEPS)
    # errors.append((epoch, error))
    # errors = np.array(errors)
    print(f'eval_dataset\'s error: {error}')  
    # np.savetxt(f'data/errors/errors_{MODEL_NAME}.csv', errors)

    # Log(error)
    with summary_writer.as_default():
        tf.summary.scalar('error', error, step=epoch)
