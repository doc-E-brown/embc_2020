#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. currentmodule:: 

"""
__author__ = 'Ben Johnston'

import tensorflow as tf

from src.models.model1 import * 
from src.data.muct import load_tensors
from src.data import augment_data 

from src.models.losses import dice_loss

import numpy as np 

x, y = load_tensors(seed=0)

model = model5()

callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.ModelCheckpoint('saved_model.hdf5',
        save_weights_only=False,
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(patience=10),
    tf.keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1, min_lr=0.0001),
]

model.fit(
    x=x,
    y=y,
    batch_size=16,
    epochs=1000,
    shuffle=True,
    validation_split=0.3,
    callbacks=callbacks,
)
