import os
import pickle
import tensorflow as tf
import numpy as np
from PIL import Image
from src.data.muct import DATA_DIR
from src.data import resize
from src.models.losses import dice_loss
from src.models.model1 import *

MODEL_DIR = os.path.join('models')
CUSTOM_MASK_DIR = os.path.join(DATA_DIR, 'custom_masks')
TEST_DATA = os.path.join(DATA_DIR,
    'nnet_test_data.pkl'
)


_model = 'saved_model_0473.hdf5'

model = model3()
model.load_weights(os.path.join(MODEL_DIR, _model))

# Load the test data
with open(TEST_DATA, 'rb') as _f:
    test_dat = pickle.load(_f)

prep_for_test = {}

for key, value in test_dat.items():

    _x, _y, _resize = resize(value['img'], value['resized_coords']) 
    _x = _x.reshape((1, 256, 256, 3))
    _mask = model.predict(_x / 255.)
    # _mask[np.where(_mask < 0.8)] = 0
    _mask = _mask.reshape((256, 256))
    _mask = _mask * 255.
    _mask = Image.fromarray(_mask).convert("L")
    img = Image.fromarray(_x[0])

    img.save(os.path.join(CUSTOM_MASK_DIR, key))
    _mask.save(os.path.join(CUSTOM_MASK_DIR, key.replace('.jpg', '_mask.jpg')))

    # prep_for_test[key] = {
    #     'img': _x,
    #     'resized_coords': _y,
    #     'resized_ratio': _resize
    # }

