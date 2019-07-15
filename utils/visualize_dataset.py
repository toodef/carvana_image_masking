import cv2
import numpy as np
from cv_utils.viz import ColormapVisualizer
import os

from train_config.dataset import create_augmented_dataset, create_dataset

if __name__ == '__main__':
    dataset = create_augmented_dataset(is_train=True, is_hq=False, to_pytorch=False)
    vis = ColormapVisualizer([0.5, 0.5])

    for img_idx, d in enumerate(dataset):
        img = vis.process_img(d['data'], (d['target'] * 255).astype(np.uint8))
        cv2.imwrite('test_vis/{}.jpg'.format(img_idx), img)
        cv2.imwrite('test_vis/{}_mask.jpg'.format(img_idx), d['target'].astype(np.float32))
        # cv2.waitKey()

        if img_idx > 5:
            break
