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

train_data, valid_data = load_tensors(seed=0)
train_data = train_data.shuffle(1000, seed=0).batch(32)
valid_data = valid_data.batch(32)

loss_object = dice_loss 
optimizer = tf.keras.optimizers.SGD(lr=0.1, momentum=0.95)

model = model1()

def loss(model, x, y):

    _y = model(x)
    return loss_object(y_true=y, y_pred=_y)

def grad(model, inputs, targets):

    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)

train_loss_results = []
valid_loss_results = []

min_loss = tf.constant(np.inf) 
best_epoch = tf.constant(0)

for epoch in range(20):

    epoch_loss_avg = tf.keras.metrics.Mean()

    # batches
    for x, y in train_data:

        loss_val, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss_avg(loss_val)

    valid_loss_avg = tf.keras.metrics.Mean()

    # Validation data
    for x, y in valid_data:

        loss_val = loss(model, x, y)
        valid_loss_avg(loss_val)

    train_loss = epoch_loss_avg.result()
    valid_loss = epoch_loss_avg.result()

    if valid_loss < min_loss:
        min_loss = valid_loss
        best_epoch = epoch

        model.save_weights('model_weights.rb')

    if 1:
    # if epoch % 50:
        print('Epoch {:03d}: Train Err: {:.3f}, Valid Err: {:0.3f},'\
            ' Best Err: {:0.3f}@{:03d}'.format(
            epoch, train_loss, valid_loss, min_loss, best_epoch 
        ))

