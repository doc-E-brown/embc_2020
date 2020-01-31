#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. currentmodule:: 

"""
__author__ = 'Ben Johnston'

import tensorflow as tf

from src.models.model1 import model1, toymod
from src.data.muct import load_tensors
from src.data import augment_data 

from src.models.losses import dice_loss

import numpy as np 

x, y = load_tensors(seed=0)

model = model1()

callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        save_weights_only=True,
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(patience=10),
]

model.fit(
    x=x,
    y=y,
    batch_size=64,
    epochs=1000,
    shuffle=True,
    validation_split=0.3,


)