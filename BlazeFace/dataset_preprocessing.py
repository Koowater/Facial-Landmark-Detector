import os
from tqdm import tqdm
import pandas as pd

dataset_dir = os.path.join('..', '..', '..', 'Dataset', 'WIDER_FACE')
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