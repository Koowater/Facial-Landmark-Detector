from tasks.model import FacialLandmarkDetector
from libs import dp, eval
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
model = FacialLandmarkDetector()


checkpoint_path = "data/checkpoint_model/cp-{epoch:04d}.ckpt"
epoch = 36
model.load_weights(checkpoint_path.format(epoch=epoch))
eval_300W = dp.Dataset(256, 64, 68, '../Data/300W/eval.csv')
generator = eval_300W.gen_eval()
i = 0
for img, kps, center, scale in tqdm(generator, desc='Evaluation', leave=True, total=600):
    pred = model.predict(np.array([img]))[0]
    kps = kps * 256 / 64
    pred_kps = eval.parse_heatmap(pred) * 256 / 64
    error = eval.l2_distance(pred_kps, kps)
    plt.axis('off')
    plt.imshow(img)
    plt.scatter(kps[:,0], kps[:,1], s=15, c='red',  edgecolors=None)
    plt.scatter(pred_kps[:,0], pred_kps[:,1], s=15,marker='x', c='cyan', edgecolors=None)
    plt.savefig(f'evaluated_img/model/{str(i)}_{int(error*10)}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    i += 1