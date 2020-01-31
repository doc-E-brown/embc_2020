#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. currentmodule:: 

"""
__author__ = 'Ben Johnston'

import os
import pickle
import tensorflow as tf
import random
from pathlib import Path
from imageio import imread

from PIL import Image

import numpy as np

from src.data import augment_data, create_mask

DATA_DIR = os.path.join(Path(__file__).parents[2], 'data')
EXTERNAL_DIR = os.path.join(DATA_DIR, 'external')
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')
MUCT_76 = os.path.join(INTERIM_DIR, 'muct76-opencv.csv')
MUCT_DIR = os.path.join(DATA_DIR, 'muct')
MUCT_FEATURES = os.path.join(MUCT_DIR, 'features')
NOSE_BOXES = os.path.join(MUCT_DIR, 'muct_noses.pkl')

IMG = 0
LAB = 1
BBOX = 2


def load_data(dat_dir=EXTERNAL_DIR, nose_boxes=NOSE_BOXES, train_ratio=0.7, seed=0):

    data = [[], [], []]

    with open(NOSE_BOXES, 'rb') as _f:
        bboxes = pickle.load(_f)

    for filename in os.listdir(dat_dir):

        if '.txt' in filename:
            continue

        basename, _ = os.path.splitext(filename)

        if filename in bboxes:
            _img = imread(os.path.join(dat_dir, filename)).astype(np.float32)
            _lab = np.loadtxt(os.path.join(dat_dir, f'{basename}.txt')).astype(np.float32)

            data[IMG].append(_img)
            data[LAB].append(_lab)
            data[BBOX].append(bboxes[filename][:2])

    x = np.array(data[IMG])
    y = np.array(data[LAB])
    z = np.array(data[BBOX]).astype(np.int32)

    return x, y, z

    # indices = list(range(len(x)))
    # random.seed(seed)
    # random.shuffle(indices)

    # num_train = int(len(x) * train_ratio)
    # train_idx = indices[:num_train]
    # valid_idx = indices[num_train:]


    # train_ds = tf.data.Dataset.from_tensor_slices((
    #     tf.convert_to_tensor(x[train_idx], dtype=tf.float32),
    #     tf.convert_to_tensor(y[train_idx], dtype=tf.float32),
    #     tf.convert_to_tensor(z[train_idx], dtype=tf.float32),
    # ))

    # valid_ds = tf.data.Dataset.from_tensor_slices((
    #     tf.convert_to_tensor(x[valid_idx], dtype=tf.float32),
    #     tf.convert_to_tensor(y[valid_idx], dtype=tf.float32),
    #     tf.convert_to_tensor(z[valid_idx], dtype=tf.float32),
    # ))

    # return train_ds, valid_ds


def load_coords(coords_csv=MUCT_76):

    with open(coords_csv, 'r') as _f:

        _f.readline()

        for line in _f:

            line = line.strip().split(',')

            label, _, *coords = line
            x = [float(a) for a in coords[::2]]
            y = [float(a) for a in coords[1::2]]
            coords_out = np.zeros((len(x), 2))

            coords_out[:,0] = x 
            coords_out[:,1] = y 
 
            coords_out = coords_out[[39, 43]]

            savepath = os.path.join(
                os.path.abspath(os.path.dirname(coords_csv)),
                f'{label}.txt'
            )
            np.savetxt(savepath, coords_out)


def generate_data(iters=10):

    x, y, z = load_data()

    for _iter in range(iters):

        count = 0

        for _x, _y, _z in zip(x, y, z):


            _x = _x.reshape((1, _x.shape[0], _x.shape[1], _x.shape[2]))
            _y = _y.reshape((1, _y.shape[0], _y.shape[1]))
            _z = _z.reshape((1, _z.shape[0]))

            _x, _y = augment_data(_x, _y, _z, size=200, jit_rate=20)

            width, height = _x[0].numpy().shape[:-1]
            mask = create_mask(_y[0].numpy().astype(np.int), width, height)

            target_name = os.path.join(MUCT_FEATURES, f'{_iter}_{count}')

            img = Image.fromarray(_x[0].numpy().astype(np.uint8))

            img.save(f'{target_name}.png')
            mask.save(f'{target_name}_mask.png')

            count += 1



if __name__ == "__main__":
    generate_data()