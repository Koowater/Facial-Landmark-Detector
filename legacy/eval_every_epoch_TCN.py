from tasks.model_TCN import FacialLandmarkDetector
from libs import dp, eval
import numpy as np
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model = FacialLandmarkDetector()

errors = []
for i in tqdm(range(120), leave=True, position=0):
    checkpoint_path = "data/checkpoint_model_TCN/cp-{epoch:04d}.ckpt"
    epoch = i
    model.load_weights(checkpoint_path.format(epoch=epoch))
    eval_300W = dp.Dataset(256, 64, 68, '../Data/300W/eval.csv')
    generator = eval_300W.gen_eval()
    error = 0.
    for img, kps, center, scale in tqdm(generator, desc='Evaluation_' + str(i), leave=True, position=1):
        pred = model.predict(np.array([img]))[0]
        kps = kps * 256 / 64
        pred_kps = eval.parse_heatmap(pred) * 256 / 64
        error += eval.l2_distance(pred_kps, kps)
        # break
    error = error / 600
    errors.append((i, error))

errors = np.array(errors)   
np.savetxt('model_TCN_errors.csv', errors)