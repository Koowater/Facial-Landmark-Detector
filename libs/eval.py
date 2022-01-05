import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import io

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
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


