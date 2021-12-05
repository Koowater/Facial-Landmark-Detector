import matplotlib.pyplot as plt
import copy
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
import sys
import io

def l2_distance(a, b):
    d = a - b
    error = 0.
    for i, p in enumerate(d):
        error += np.sqrt(p[0] ** 2 + p[1] ** 2)
    error = error / d.shape[0]
    return error

def parse_heatmap(hms):
    pred_kps = np.zeros((hms.shape[-1], 2))
    size = hms.shape[0]
    for i in range(hms.shape[-1]):
        hm = hms[:,:,i]
        idx = np.argmax(hm)
        x = idx % size
        y = idx // size
        pred_kps[i] = np.array([x, y])
    return pred_kps

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