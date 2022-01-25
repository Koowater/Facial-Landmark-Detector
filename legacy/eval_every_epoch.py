import argparse
import importlib
from libs import dp, eval
import numpy as np
from tqdm import tqdm
import os

np.set_printoptions(precision=6, suppress=True)

# MODELS = ['SHG', 'SHG_TCN', 'SHG_TCN_he', 'UNet', 'VGG16', 'ResNet52']

parser = argparse.ArgumentParser()
parser.add_argment(dest='model')
parser.add_argment(dest='num_gpu')
parser.add_argment(dest='hm_size')

MODEL_NAME = parser.parse_args().model
NUM_GPU = parser.parse_args().num_gpu

HM_SIZE = parser.parse_args().hm_size

os.environ["CUDA_VISIBLE_DEVICES"] = NUM_GPU

model_lib = importlib.import_module('tasks.model_' + MODEL_NAME)
model = model_lib.FacialLandmarkDetector()

errors = []
for i in tqdm(range(120), leave=True, position=0):
    checkpoint_path = "data/checkpoint_model/cp-{epoch:04d}.ckpt"
    epoch = i
    model.load_weights(checkpoint_path.format(epoch=epoch))
    eval_300W = dp.Dataset(256, HM_SIZE, 68, '../Data/300W/eval.csv')
    generator = eval_300W.gen_eval()
    error = 0.
    for img, kps, center, scale in tqdm(generator, desc='Evaluation_' + str(i), leave=True, position=1):
        pred = model.predict(np.array([img]))[0]
        kps = kps * 256 / HM_SIZE
        pred_kps = eval.parse_heatmap(pred) * 256 / HM_SIZE
        error += eval.l2_distance(pred_kps, kps)
        # break
    error = error / 600
    errors.append((i, error))

errors = np.array(errors)
np.savetxt(f'data/errors/errors_{MODEL_NAME}.csv', errors)
