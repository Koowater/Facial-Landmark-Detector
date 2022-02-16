import matplotlib.pyplot as plt
import numpy as np
import cv2
from parso import parse
import tensorflow as tf 
import io

from tasks.model_SHG import StackedHourglassNetworks

def l2_distance(a, b):
    d = a - b
    error = 0.
    for i, p in enumerate(d):
        error += tf.math.sqrt(p[0] ** 2 + p[1] ** 2)
    error = error / d.shape[0]
    return error

def parse_heatmap(hms):
    if hms.shape[0] == None:
        pred_kps = tf.Variable(np.zeros((1, hms.shape[-1], 2)), dtype=tf.float32)
    else:
        pred_kps = tf.Variable(np.zeros((hms.shape[0], hms.shape[-1], 2)), dtype=tf.float32)
    size = hms.shape[-2]
    
    for i, hm in enumerate(hms):
        flatten = tf.reshape(hm, (hm.shape[0] * hm.shape[1], hm.shape[2]))
        argmax = tf.argmax(flatten, axis=0)
        argmax_x = argmax % hm.shape[1]
        argmax_y = argmax // hm.shape[0]
        pred_kp = tf.stack([argmax_x, argmax_y], axis=1)
        pred_kps[i].assign(tf.cast(pred_kp, tf.float32))
    return tf.convert_to_tensor(pred_kps)

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
    # Convert PNG buffer to input image, 0)
    return image

class Inferencer():
    def __init__(self, path):
        # self.model = StackedHourglassNetworks(4, 256, 68)
        # self.model.load_weights(path)
        self.model = tf.saved_model.load(path)

    def __call__(self, input):
        # if input.ndim == 3:
        if type(input) is str:
            heatmaps = self.from_image_path(input)
        else:
            heatmaps = self.from_ndarray(input)
        landmarks = parse_heatmap(heatmaps)
        return landmarks

    def from_ndarray(self, ndarray):
        # input size  : (None, 256, 256, 3)
        # input dtype : np.float32
        return self.model(ndarray)

    def from_image_path(self, path):
        load_image = cv2.imread(path)
        resized_image = cv2.resize(load_image, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        return self.model(np.array([resized_image]).astype(np.float32) / 255.)

            