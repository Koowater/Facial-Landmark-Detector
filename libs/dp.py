import cv2
import sys
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.getcwd()))

import libs.utils



class Dataset():
    def __init__(self, img_res, hms_res, num_parts, csv_dir):
        self.img_res = img_res
        self.hms_res = hms_res
        self.num_parts = num_parts
        self.csv_dir = csv_dir
        self.base_dir = os.path.dirname(self.csv_dir)
        self.load_datalist(self.csv_dir)
        self.idx_list = np.arange(len(self.train_list))
        self.hmscale = hms_res / 64
        print(f'scale: {self.hmscale}')

    def load_datalist(self, csv_dir):
        self.df = pd.read_csv(csv_dir)
        del self.df['Unnamed: 0']
        self.train_list = self.df.values.tolist()
        print(f'\nTrain dataset: {self.csv_dir}')
        print(f'Train dataset is loaded. Shape: {self.df.shape}')

    def get_path(self, path):
        return os.path.join(self.base_dir, path)

    def get_label(self, label_file):
        kps = np.genfromtxt(self.get_path(label_file))
        top = kps[:,1].min()
        right = kps[:,0].max()
        bottom = kps[:,1].max()
        left = kps[:,0].min()
        bb = np.array([left, top, right, bottom])
        center = [bb[2] - (bb[2] - bb[0]) / 2.0, bb[3] - (bb[3] - bb[1]) / 2.0]
        scale = (bb[2] - bb[0] + bb[3] - bb[1]) / 256 # 195
        return kps, center, scale

    def get_img(self, image_file):
        img_path = self.get_path(image_file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = tf.io.read_file(img_path)
        # img = tf.image.decode_jpeg(img, channels=3)
        return img

    def load_image(self, idx):
        # load + crop
        image_file, label_file = self.train_list[idx]
        orig_img = self.get_img(image_file)
        orig_kps, center, scale = self.get_label(label_file)
        
        orig_img, orig_kps = self.random_rotate(orig_img, orig_kps, center)
        
        orig_img = self.random_color_augmentation(orig_img)

        cropped = libs.utils.crop_and_normalize(orig_img, center, scale)
        heatmaps = libs.utils.create_target_heatmap(orig_kps, self.hms_res, center, scale, self.hmscale)

        # heatmap = np.zeros((self.hms_res, self.hms_res))
        # for i in range(heatmaps.shape[2]):
        #     heatmap += heatmaps[:,:,i]
        # heatmap = heatmap.reshape((heatmap.shape[0], heatmap.shape[1], 1))

        # plt.imshow(heatmap)
        # plt.savefig('heatmaps')
        # plt.show()
    
        # plt.imshow(cropped)
        # plt.savefig('result')
        # plt.show()
        # print(cropped.max())

        return cropped.astype(np.float32), heatmaps.astype(np.float32)

    def load_eval(self, idx):
        image_file, label_file = self.train_list[idx]
        orig_img = self.get_img(image_file)
        orig_kps, center, scale = self.get_label(label_file)

        cropped = libs.utils.crop_and_normalize(orig_img, center, scale)
        kps = libs.utils.create_target_landmarks(orig_kps, center, scale, self.hms_res)

        return cropped, kps, center, scale

    def rotate_landmarks(self, landmarks, angle, center):
        radian = np.radians(angle)
        rotated_landmarks = []
        for i in landmarks:
            _x = i[0] - center[0]
            _y = i[1] - center[1]
            x = _x * np.cos(radian) - _y * np.sin(radian)
            y = _x * np.sin(radian) + _y * np.cos(radian)
            x += center[0]
            y += center[1]
            rotated_landmarks.append([x, y])

        return np.array(rotated_landmarks, dtype='float32')

    def random_rotate(self, image, kps, center):
        src = image
        angle = np.random.uniform(-5., 5.)
        height, width, _ = src.shape
        matrix = cv2.getRotationMatrix2D(tuple(center), -angle, 1)
        dst = cv2.warpAffine(src, matrix, (width, height))
        rotated_landmarks = self.rotate_landmarks(kps, angle, center)

        return dst, rotated_landmarks

    def random_color_augmentation(self, src):
        hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        i = (np.random.random() - 0.5) * 0.8
        # img = tf.image.adjust_saturation(src, i)
        s = s * (1 - i)
        s = np.clip(s, 0, 255)
        s = np.ndarray.astype(s, 'uint8')

        merged = cv2.merge([h, s, v])
        dst = cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)

        a = 1.0 + (np.random.random() - 0.5) * 0.4
        b = np.random.randint(-20, 20)
        dst = dst.astype(np.float32)
        # img = tf.image.adjust_contrast(img, a)
        # img = tf.image.adjust_brightness(img, b)
        dst = dst * a + b
        dst = np.clip(dst, 0, 255)
        dst = dst.astype(np.uint8)

        return dst

    def gen(self): 
        np.random.shuffle(self.idx_list)
        for idx in self.idx_list:
            img, heatmaps = self.load_image(idx)
            yield img, heatmaps

    def gen_eval(self):
        for idx in self.idx_list:
            image, landmarks, center, scale = self.load_eval(idx)
            yield image, landmarks, center, scale

    def tf_eval_dataset_from_generator(self, batch_size):
        return tf.data.Dataset.from_generator(
            self.gen_eval,
            output_types=(
                np.float32, np.float32, np.float32, np.float32),
            output_shapes=(
                [self.img_res, self.img_res, 3],
                [self.num_parts, 2],
                2, None),
            args=()
        ).shuffle(128, reshuffle_each_iteration=True).repeat().batch(batch_size)

    def tf_dataset_from_generator(self, batch_size):
        return tf.data.Dataset.from_generator(
            self.gen,
            output_types=(
                np.float32, np.float32),
            output_shapes=(
                [self.img_res, self.img_res, 3],
                [self.hms_res, self.hms_res, self.num_parts]),
            args=()
        ).shuffle(128, reshuffle_each_iteration=True).repeat().batch(batch_size)

    # for VGG-16
    def load_with_no_heatmaps(self, idx):
        image_file, label_file = self.train_list[idx]
        orig_img = self.get_img(image_file)
        orig_kps, center, scale = self.get_label(label_file)
        
        orig_img, orig_kps = self.random_rotate(orig_img, orig_kps, center)
        
        orig_img = self.random_color_augmentation(orig_img)

        cropped = libs.utils.crop_and_normalize(orig_img, center, scale)
        kps = libs.utils.create_target_landmarks(orig_kps, center, scale, 256)
        kps = kps / 256.

        return cropped.astype(np.float32), kps.astype(np.float32)

    def gen_vgg16(self):
        np.random.shuffle(self.idx_list)
        for idx in self.idx_list:
            img, kps = self.load_with_no_heatmaps(idx)
            np_kps = np.zeros((136), dtype=np.float32)
            for i in range(68):
                np_kps[i*2] = kps[i][0]
                np_kps[i*2+1] = kps[i][1]
            yield img, np_kps


if __name__ == "__main__":
    # dataset = Dataset(256, 64, 68, '../Data/300W_train/train.csv')
    # dataset.load_image(4)
    scale = 64
    dataset = Dataset(256, scale, 68, '../Datasets/300W/eval.csv')
    img, hm = dataset.load_image(0)
    
    rotation = [-5., 0., 5.]
    saturation = [-0.6, 1., 1.4]
    contrast = [0.8, 1., 1.2]
    brightness = [-20, 0, 20]

    print(img.max(), img.min())
    print(hm.shape)
    print(hm.max(), hm.min())