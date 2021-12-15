import numpy as np
import cv2
import math
import tensorflow as tf
import eval

class my_loss(tf.keras.losses.Loss):
    def __init__(self, NM=None, HM=None,  L2=None, imgSize=256, HMSize=64, batchSize = 8):
        super(my_loss, self).__init__()
        # Loss Check Parameter
        self.NM = NM
        self.HM = HM
        self.L2 = L2

        # Loss Using Value
        self.imgSize   = imgSize
        self.HMSize    = HMSize
        self.batchSize = batchSize

    def call(self, y_ture, y_pred):
        loss_value = 0

        if self.NM is not None:
            loss_value += self.NME(y_ture, y_pred)

        if self.HM is not None:
            loss_value += self.HeatmapLoss(y_ture, y_pred)
        
        if self.L2 is not None:
            loss_value += self.L2Loss(y_ture, y_pred)
        
        return loss_value

    def NME(self, y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))

    def HeatmapLoss(self, y_true, y_pred):
        l = ((y_pred - y_true)**2)
        l = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(l, 3), 2), 1)
        return l ## l of dim bsize
    
    def L2Loss(self, y_true, y_pred):
        loss = 0
        for t, p  in zip(y_true, y_pred):
            t = eval.parse_heatmap(t) * self.imgSize / self.HMSize
            p = eval.parse_heatmap(p) * self.imgSize / self.HMSize
            loss += eval.l2_distance(p, t)
        
        return loss / self.batchSize

        # for i, pred in enumerate(y_pred):
        # eval.l2_distance(pred_kps[i], kps[i])



def NME(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))

def lr_scheduler(epoch, lr):
    if epoch == 15:
        return 1e-05
    if epoch == 30 :
        return 1e-06
    return lr

# Loss function
def HeatmapLoss(y_true, y_pred):
    l = ((y_pred - y_true)**2)
    l = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(l, 3), 2), 1)
    return l ## l of dim bsize

def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)

def crop(img, center, scale, res, rot=0):
    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    return cv2.resize(new_img, res)

def create_bounding_box(target_landmarks, expansion_factor=0.0):
    """
    gets a batch of landmarks and calculates a bounding box that includes all the landmarks per set of landmarks in
    the batch
    :param target_landmarks: batch of landmarks of dim (n x 68 x 2). Where n is the batch size
    :param expansion_factor: expands the bounding box by this factor. For example, a `expansion_factor` of 0.2 leads
    to 20% increase in width and height of the boxes
    :return: a batch of bounding boxes of dim (n x 4) where the second dim is (x1,y1,x2,y2)
    """
    # Calc bounding box
    x_y_min = np.reshape(target_landmarks, (-1, 209, 2)).min(axis=1)
    x_y_max = target_landmarks.reshape(-1, 209, 2).max(axis=1)
    # expanding the bounding box
    expansion_factor /= 2
    bb_expansion_x = (x_y_max[:, 0] - x_y_min[:, 0]) * expansion_factor
    bb_expansion_y = (x_y_max[:, 1] - x_y_min[:, 1]) * expansion_factor
    x_y_min[:, 0] -= bb_expansion_x
    x_y_max[:, 0] += bb_expansion_x
    x_y_min[:, 1] -= bb_expansion_y
    x_y_max[:, 1] += bb_expansion_y
    
    return np.concatenate((x_y_min, x_y_max), axis=1)

def _gaussian(
        size=3, sigma=0.15, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss

def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] > image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = int(6 * sigma + 1)
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return image

def create_target_heatmap(target_landmarks, hm_res, centers, scales, hmscale):
    heatmaps = np.zeros((hm_res, hm_res, 68), dtype=np.float32)
    for p in range(68):
        landmark_cropped_coor = transform(target_landmarks[p] + 1, centers, scales, (hm_res, hm_res), invert=False)
        heatmaps[:,:,p] = draw_gaussian(heatmaps[:,:,p], landmark_cropped_coor + 1, hmscale)
    return heatmaps

def create_target_landmarks(target_landmarks, center, scale, size):
    landmarks = np.zeros(target_landmarks.shape)
    for i, kp in enumerate(target_landmarks):
        landmarks[i] = transform(kp, center, scale, (size, size))
    return landmarks

def make_bb(face_locations):
    bb = []
    for face_location in face_locations:
        top, right, bottom, left = face_location
        bb.append([left, top, right, bottom])
    return np.array(bb)

def crop_and_normalize(image, center, scale):
    cropped = crop(image, center, scale, (256, 256))
    float_image = cropped.astype(np.float32)
    normalized_image = float_image / 255.
    return normalized_image