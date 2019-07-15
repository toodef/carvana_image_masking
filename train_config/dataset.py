import os

import cv2
import pydicom
# from cv_utils.utils import rle2mask
from pydicom.data import get_testdata_files

import numpy as np
import torch
from albumentations import Compose, HorizontalFlip, CenterCrop, GaussNoise, RandomBrightnessContrast, RandomCrop
from cv_utils.datasets.common import BasicDataset

__all__ = ['Dataset', 'AugmentedDataset', 'create_dataset', 'create_augmented_dataset']

DATA_HEIGHT, DATA_WIDTH = 1280, 1280


def rle2mask(rle: [int], width: int, height: int):
    mask = np.zeros(width * height)
    array = np.asarray(rle)
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position = start
        mask[current_position:current_position + lengths[index]] = 255

    return mask.reshape(width, height)


class Dataset(BasicDataset):
    env_name = 'CARVANA_IMAGE_MASKING_DATASET'

    def __init__(self, is_hq: bool):
        if self.env_name not in os.environ:
            raise Exception("Cant find dataset root path. Please set {} environment variable".format(self.env_name))

        root = os.environ[self.env_name]

        labels = np.genfromtxt(os.path.join(root, 'train_masks.csv'), delimiter=',', dtype=str)[1:]
        items = self._parse_items(os.path.join(root, 'train_hq' if is_hq else 'train'), labels)

        super().__init__(items)

    @staticmethod
    def _interpret_item(it) -> any:
        img = cv2.imread(it[0])
        rles = it[1]
        if type(rles) is list:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
            for rle in rles:
                mask += rle2mask(rle, img.shape[0], img.shape[1]).astype(np.float32) / 255
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        return {'data': img, 'target': mask}

    def _parse_items(self, images_root: str, labels: np.ndarray) -> []:
        images = self._parse_images(images_root)

        pairs = {}
        for label in labels:
            ident = os.path.splitext(label[0])[0]
            if ident in images:
                rle = str(label[1]).strip()
                if ident in pairs:
                    pairs[ident].append(rle)
                else:
                    pairs[ident] = [rle]
            else:
                raise Exception("Not all targets have paired images")

        res = []
        for ident, rle in pairs.items():
            if rle == ['-1']:
                rle = -1
            else:
                rle = [[int(v) for v in r.split(' ')] for r in rle]
            res.append([images[ident], rle])

        return res

    @staticmethod
    def _parse_test_items(images_root: str) -> []:
        images = []
        for cur_root, dirs, files in os.walk(images_root):
            for file in files:
                if os.path.splitext(file)[1] == ".dcm":
                    images.append(os.path.join(cur_root, file))
        return images

    @staticmethod
    def _parse_images(images_root: str) -> []:
        images = {}
        for cur_root, dirs, files in os.walk(images_root):
            for file in files:
                if os.path.splitext(file)[1] == ".jpg":
                    identifier = os.path.splitext(file)[0]
                    if identifier in images:
                        raise RuntimeError('Images names are duplicated [{} and {}]'.format(file, images[identifier]))
                    images[identifier] = os.path.join(cur_root, file)
        return images


class AugmentedDataset:
    def __init__(self, dataset):
        self._dataset = dataset
        self._augs = {}
        self._augs_for_whole = []

    def add_aug(self, aug: callable, identificator=None) -> 'AugmentedDataset':
        if identificator is None:
            self._augs_for_whole.append(aug)
        else:
            self._augs[identificator] = aug
        return self

    def __getitem__(self, item):
        res = self._dataset[item]
        for k, v in res.items() if isinstance(res, dict) else enumerate(res):
            if k in self._augs:
                res[k] = self._augs[k](v)
        for aug in self._augs_for_whole:
            res = aug(res)
        return res

    def __len__(self):
        return len(self._dataset)


class Augmentations:
    def __init__(self, is_train: bool, to_pytorch: bool):
        preprocess = RandomCrop(height=DATA_HEIGHT, width=DATA_WIDTH, always_apply=True, p=1)
        transforms = Compose([HorizontalFlip(), RandomBrightnessContrast(), GaussNoise()], p=0.8)

        if is_train:
            self._aug = Compose([preprocess, transforms])
        else:
            self._aug = preprocess

        self._need_to_pytorch = to_pytorch

    def augmentate(self, data: {}):
        augmented = self._aug(image=data['data'], mask=data['target'])
        if self._need_to_pytorch:
            image = augmented['image'].astype(np.float32) / 128 - 1
            target = np.expand_dims(augmented['mask'], 0)
            return {'data': torch.from_numpy(image), 'target': torch.from_numpy(target)}
        else:
            return {'data': augmented['image'], 'target': augmented['mask']}


def create_dataset(is_hq: bool, indices_path: str = None) -> 'BasicDataset':
    dataset = Dataset(is_hq)
    if indices_path is not None:
        dataset.load_indices(indices_path, remove_unused=True)
    return dataset


def create_augmented_dataset(is_train: bool, is_hq: bool, to_pytorch: bool = True, indices_path: str = None) -> 'AugmentedDataset':
    dataset = create_dataset(is_hq, indices_path)
    augs = Augmentations(is_train, to_pytorch)

    return AugmentedDataset(dataset).add_aug(augs.augmentate)
