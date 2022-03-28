import tensorflow as tf
import os

# Wider Face
class WiderFaceLoader:
    def __init__(self, dataset_dir, batch_size, img_size=(128, 128)):
        if os.path.isdir(dataset_dir):
            self.dataset_dir = dataset_dir
        else: 
            raise Exception("Can't find the dataset directory.")
        self.batch_size = batch_size
        self.img_size = img_size
        
    def load_train(self):
        

    # 구현 목록
    # - training을 위한 generator
    # - image loader
    # - overviewing whole dataset.

    def summary(self):




    # resized, normalized image
    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(self.img_size)
        img = tf.math.multiply(img, 1. / 255.)
        return img
    
    # def tf_generator(self):
        
            

