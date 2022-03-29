# For training BlazeFace, input image and label(box's x, y, width, height, image's width and height information for box mapping) are required.

import os
import glob
import pickle
from tqdm import tqdm
import pandas as pd
import cv2 

dataset_dir = os.path.join('..', '..', 'Dataset', 'WIDER_FACE')
if not os.path.isdir(dataset_dir):
    raise Exception("Dataset directory is not exist. Check dataset's path.")

train_label_path = os.path.join(dataset_dir, 'wider_face_split', 'wider_face_train_bbx_gt.txt')
if not os.path.isfile(train_label_path):
    raise Exception("Can't find [wider_face_train_bbx_gt.txt] file.")

val_label_path = os.path.join(dataset_dir, 'wider_face_split', 'wider_face_val_bbx_gt.txt')
if not os.path.isfile(train_label_path):
    raise Exception("Can't find [wider_face_val_bbx_gt.txt] file.")

gts = (('train', train_label_path), ('val', val_label_path))

columns = ('image_path', 'x', 'y', 'w', 'h', 'blur', 'expression', 'illumination', 'invalid', 'occlusion', 'pose')


# For label.csv

for label_type, label_path in gts:
    print(f"\n{label_type.upper()} txt file is converting to csv...")
    file = open(label_path, 'r')
    strings = file.readlines()
    file.close()
    idx = 0
    cur_line = 0
    end_line = len(strings)

    labels = []

    with tqdm(total=end_line, desc=f'{label_type}') as pbar:
        while True:
            if cur_line >= end_line:
                break
            image_path = strings[cur_line].strip()
            n_boxes = int(strings[cur_line + 1])
            if n_boxes:
                labels += list(map(lambda x: [image_path] + list(map(int, x.split())), strings[cur_line + 2 : cur_line + 2 + n_boxes]))
                cur_line += 2 + n_boxes
            else: 
                cur_line += 3

            pbar.update(cur_line - pbar.n)

        df = pd.DataFrame(labels, columns=columns)
        df.to_csv(os.path.join(dataset_dir, f"{label_type}.csv"), index=False)

print('')

# For label.pickle

def get_size_and_path(path):
    img = cv2.imread(path)
    img_splited = path.split(os.sep)[-2:]
    img_path = os.path.join(img_splited[0], img_splited[1])
    return img_path, (img.shape[1], img.shape[0]) # width, height

for t in ('train', 'val'):
    label_dict = {}
    df = pd.read_csv(os.path.join(dataset_dir, f"{t}.csv"))
    img_list = glob.glob(os.path.join(dataset_dir, f"WIDER_{t}", "images", "*/*.*"))
    print(f"{t} images\n{len(img_list)} images are exist. Data mapping... Please wait.")
    img_info = tuple(map(get_size_and_path, img_list))
    print(f'{t.capitalize()} dataset dictionary is generated from images and csv file.')
    for i in tqdm(img_info):
        img_path, img_size = i[0], i[1]
        label_dict[img_path] = {}
        label_dict[img_path]['size'] = img_size
        bbox_array = df.loc[df['image_path'] == img_path, ['x', 'y', 'w', 'h']].to_numpy()
        label_dict[img_path]['bbox'] = bbox_array
    
    with open(f"BlazeFace/label_{t}.pickle", "wb") as fw:
        pickle.dump(label_dict, fw)


# This code is too slow!!!

# df = pd.DataFrame(columns=('image_path', 'x', 'y', 'w', 'h', 'blur', 'expression', 'illumination', 'invalid', 'occlusion', 'pose'))
# file = open(train_label_path, 'r')
# while True:
#     image_path = file.readline().strip()
#     if not image_path:
#         break
#     n_boxes = int(file.readline())
#     if n_boxes == 0:
#         dummy = file.readline()
#         continue
#     for i in range(n_boxes):
#         annotation = file.readline().split()
#         annotation.insert(0, image_path)
#         df = df.append(pd.Series(annotation, index=df.columns), ignore_index=True)

# file.close()