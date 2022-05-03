import tensorflow as tf
import os
import pickle

# Wider Face
class WiderFaceLoader:
    def __init__(self, dataset_dir, dict_dir, batch_size, input_size=(128, 128), name='name'):
        self.name = name
        if os.path.isdir(dataset_dir):
            self.dataset_dir = dataset_dir
        else: 
            raise Exception("Can't find the dataset directory.")

        if os.path.isfile(dict_dir):
            with open(dict_dir,'rb') as fr:
                self.label_dict = pickle.load(fr)
        else:
            raise Exception("Can't find the pickle.")

        self.batch_size = batch_size
        self.input_size = input_size
        self.summary()


    def load_train(self):
        pass

    # 구현 목록
    # - training을 위한 generator
    # - image loader
    # - overviewing whole dataset.

    def summary(self):
        n_boxes = 0
        n_boxes = 
        
        print("===========================================")
        print(f"{self.name.capitalize()}")
        print("-------------------------------------------")
        print(f"{len(self.label_dict)} images")
        print(f"")
        print(f"")
        print("===========================================")



    # resized, normalized image
    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(self.img_size)
        img = tf.math.multiply(img, 1. / 255.)
        return img
    
    # def tf_generator(self):
        
            
if __name__ == "__main__":
    dataset_dir = os.path.join('..', '..', 'Dataset', 'WIDER_FACE')
    dict_path = os.path.join('BlazeFace', 'label_train.pickle')
    batch_size = 32
    dataset = WiderFaceLoader(dataset_dir, dict_path, batch_size, (128, 128), 'train_set')
