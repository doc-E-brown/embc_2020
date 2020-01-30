#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. currentmodule:: 

"""
__author__ = 'Ben Johnston'

import tensorflow as tf


def toymod():

    _int = tf.keras.layers.Input(shape=(100, 100, 3))
    flat = tf.keras.layers.Flatten()(_int)
    fc1 = tf.keras.layers.Dense(4, activation='relu')(flat)

    return tf.keras.Model(inputs=_int, outputs=fc1)


def model1():

    backbone = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=(100, 100, 3))  

    for i in range(82):
        backbone.layers[i].trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output) 
    x = tf.keras.layers.Dense(4, activation='sigmoid')(x)

    _model = tf.keras.Model(inputs=backbone.inputs, outputs=x)

    return _model 

    
if __name__ == "__main__":
    model1()