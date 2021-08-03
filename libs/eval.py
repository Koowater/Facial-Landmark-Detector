import matplotlib.pyplot as plt
import copy
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
import sys

def l2_distance(a, b):
    d = a - b
    error = 0.
    for i, p in enumerate(d):
        error += np.sqrt(p[0] ** 2 + p[1] ** 2)
    error = error / d.shape[0]
    return error

def show_hms(hms, gts=None):
    parts = ['right ankle', 'right knee', 'right hip',
             'left hip', 'left knee', 'left ankle',
             'pelvis', 'thorax', 'neck', 'head',
             'right wrist', 'right elbow', 'right shoulder',
             'left shoulder', 'left elbow', 'left wrist']
    # 4x4
    f, axes = plt.subplots(4, 4)
    f.set_size_inches((16, 16))
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            axes[i][j].set_title(parts[idx])
            axes[i][j].imshow(hms[:, :, idx])
            if gts is not None:
                axes[i][j].scatter(gts[idx][0], gts[idx][1])
            # axes[i][j].axis("off")
    plt.show()


def mpii_eval(pred, gt, normalizing, bound=0.5):
    """
    Use PCK with threshold of .5 of normalized distance (presumably head size)
    """

    correct = {'all': {'total': 0, 'ankle': 0, 'knee': 0, 'hip': 0, 'pelvis': 0,
               'thorax': 0, 'neck': 0, 'head': 0, 'wrist': 0, 'elbow': 0,
                       'shoulder': 0},
               'visible': {'total': 0, 'ankle': 0, 'knee': 0, 'hip': 0, 'pelvis': 0,
                           'thorax': 0, 'neck': 0, 'head': 0, 'wrist': 0, 'elbow': 0,
                           'shoulder': 0},
               'not visible': {'total': 0, 'ankle': 0, 'knee': 0, 'hip': 0, 'pelvis': 0,
                               'thorax': 0, 'neck': 0, 'head': 0, 'wrist': 0, 'elbow': 0,
                               'shoulder': 0}}
    count = copy.deepcopy(correct)
    for p, g, normalize in zip(pred, gt, normalizing):
        for j in range(g.shape[0]):
            vis = 'visible'
            if g[j, 0] == 0:  # not in picture!
                continue
            if g[j, 2] == 0:
                vis = 'not visible'
            joint = 'ankle'
            if j == 1 or j == 4:
                joint = 'knee'
            elif j == 2 or j == 3:
                joint = 'hip'
            elif j == 6:
                joint = 'pelvis'
            elif j == 7:
                joint = 'thorax'
            elif j == 8:
                joint = 'neck'
            elif j == 9:
                joint = 'head'
            elif j == 10 or j == 15:
                joint = 'wrist'
            elif j == 11 or j == 14:
                joint = 'elbow'
            elif j == 12 or j == 13:
                joint = 'shoulder'

            count['all']['total'] += 1
            count['all'][joint] += 1
            count[vis]['total'] += 1
            count[vis][joint] += 1

            error = np.linalg.norm(p[j, :2]-g[j, :2]) / normalize

            if bound > error:
                correct['all']['total'] += 1
                correct['all'][joint] += 1
                correct[vis]['total'] += 1
                correct[vis][joint] += 1

    # breakdown by validation set / training set
    for k in correct:
        print(k, ':')
        for key in correct[k]:
            print('PCK @,', bound, ',', key, ':',
                  round(correct[k][key] / max(count[k][key], 1), 3), ', count:', count[k][key])
        print('\n')


def parse_heatmaps(heatmaps):
    landmarks = []
    for i in range(heatmaps.shape[2]):
        idx = np.argmax(heatmaps[:, :, i])
        landmarks.append((idx % 64, idx // 64))
    return landmarks


class HumanPoseEstimator:
    def __init__(self, input_res, output_res, num_parts, epoch):
        self.model = HPE()
        self.cp_epoch = epoch
        checkpoint_path = "data/checkpoint_v005/cp-{epoch:04d}.ckpt"
        self.model.load_weights(checkpoint_path.format(epoch=self.cp_epoch))
        self.dataset = None

        self.input_res = input_res
        self.output_res = output_res
        self.num_parts = num_parts
        self.valid = 'valid.h5'

        dummy = np.zeros((1, 256, 256, 3))
        self.model.predict(dummy)

    def load_val(self):
        self.dataset = Dataset(
            self.input_res, self.output_res, self.num_parts, self.valid)

    def eval_PCK(self):
        if self.dataset is None:
            self.dataset = Dataset(
                self.input_res, self.output_res, self.num_parts, self.valid)
        generator = self.dataset.gen_eval()

        total_joints = []
        total_gts = []
        total_n = []
        for data in tqdm(generator, total=len(self.dataset.idx_list), file=sys.stdout):
            img, gts, n = (data[0], data[1][0], data[2])
            hms = self.model.predict(img)[0]

            joints = parse_heatmaps(hms)
            total_joints.append(joints)
            total_gts.append(gts)
            total_n.append(n)

        total_joints = np.array(total_joints)
        total_gts = np.array(total_gts)
        total_n = np.array(total_n)

        mpii_eval(total_joints, total_gts, total_n)

    def inference_val(self):
        if self.dataset is None:
            self.dataset = Dataset(
                self.input_res, self.output_res, self.num_parts, self.valid)
        generator = self.dataset.gen_eval()

        total_img = []
        total_joints = []
        total_gts = []
        total_n = []
        for data in tqdm(generator, total=len(self.dataset.idx_list), file=sys.stdout):
            img, gts, n = (data[0], data[1][0], data[2])
            hms = self.model.predict(img)[0]

            joints = parse_heatmaps(hms)
            total_joints.append(joints)
            total_gts.append(gts)
            total_n.append(n)

        total_joints = np.array(total_joints)
        total_gts = np.array(total_gts)
        total_n = np.array(total_n)

        return total_img, total_joints, total_gts, total_n

    def inference(self, _src, save_path=None):
        src = cv2.imread(_src)
        img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        # img = cv2.flip(img, 1)
        img = cv2.resize(img, (256, 256))
        img = np.reshape(img, (1, 256, 256, 3))
        img = img / 255
        hms = self.model.predict(img)[0]
        joints = parse_heatmaps(hms)
        if save_path is not None:

            pairs = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [10, 11], [11, 12], [12, 7], [15, 14], [14, 13], [13, 7]]
            colors = [
                (255, 127, 0), (255, 0, 0), (175, 0, 0), 
                (0, 0, 175), (0, 0, 255), (100, 100, 255), 
                (127, 0, 57), (0, 175, 175), (0, 255, 255), 
                (100, 255, 100), (0, 255, 0), (0, 175, 0), (100, 127, 255), (0, 127, 255), (0, 57, 127)]
            bone = []
            for pair in pairs:
                b = []
                for p in pair:
                    b.append(joints[p])
                bone.append(b)

            result = copy.deepcopy(src)
            orig_joints = np.array(bone)
            orig_joints[:,:,0] = orig_joints[:,:,0] / 64 * result.shape[1]
            orig_joints[:,:,1] = orig_joints[:,:,1] / 64 * result.shape[0]
            for i, pts in enumerate(orig_joints):
                start = tuple(pts[0])
                end = tuple(pts[1])
                color = colors[i]
                # if i < 2:
                #     color = (255, 50, 50)
                # elif i < 4:
                #     color = (50, 50, 50)
                # elif i < 6:
                #     color = (50, 50, 255)
                # elif i < 9:
                #     color = (255, 255, 255)
                # elif i < 11:
                #     color = (50, 255, 50)
                # else:
                #     color = (0, 0, 0)
                cv2.line(result, start, end, color, 5, cv2.LINE_AA)
            cv2.imwrite(save_path, result)
            print('Saved "{}"'.format(save_path))



        return img, hms, joints
