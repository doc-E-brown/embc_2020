import os
import pickle

import tensorflow as tf
import numpy as np
from PIL import Image

from src.data.muct import DATA_DIR
from src.data import resize, create_mask, flip_lr, adjust_image

POST_TRAIN = os.path.join(DATA_DIR, 'nnet_post_train_data.pkl')
POST_TRAIN_DIR = os.path.join(DATA_DIR, 'post_train')

def gen_posttrain(dat=POST_TRAIN, size=256):

    x, y = [], []

    with open(dat, 'rb') as _f:
        post_data = pickle.load(_f)

    for i in range(10):

        count = 0

        for key, value in post_data.items():

            if '_lr' in key:
                continue

            _x, _y, ratio = resize(value['img'], value['resized_coords'], 256)

            if np.any(_y < 0) or np.any(_y > size):
                continue

            _x = tf.convert_to_tensor(_x, dtype=tf.float32)
            _y = tf.convert_to_tensor(_y, dtype=tf.float32)

            _x, _y = flip_lr(_x, _y, 0.5, 0)

            _x = adjust_image(_x, 0)

            _x = _x.numpy()
            _y = _y.numpy()

            mask = create_mask(_y, size, size) 

            target_name = os.path.join(POST_TRAIN_DIR, f'{i}_{count}')

            img = Image.fromarray(_x.astype(np.uint8))

            img.save(f'{target_name}.png')
            mask.save(f'{target_name}_mask.png')

            count += 1



if __name__ == "__main__":
    gen_posttrain()