#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. currentmodule:: 

"""
__author__ = 'Ben Johnston'

import tensorflow as tf

from src.models.losses import dice_loss
from src.models.metrics import dice_coeff


def toymod():

    _int = tf.keras.layers.Input(shape=(100, 100, 3))
    flat = tf.keras.layers.Flatten()(_int)
    fc1 = tf.keras.layers.Dense(4, activation='relu')(flat)

    return tf.keras.Model(inputs=_int, outputs=fc1)


# def model1():

#     backbone = tf.keras.applications.ResNet50(
#         include_top=False, weights='imagenet', input_shape=(100, 100, 3))  

#     for i in range(82):
#         backbone.layers[i].trainable = False

#     x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output) 
#     x = tf.keras.layers.Dense(4, activation='sigmoid')(x)

#     _model = tf.keras.Model(inputs=backbone.inputs, outputs=x)

#     return _model 


def model1(img_shape=(256, 256, 3)):
    n_filters = 32

    vgg = tf.keras.applications.VGG16(include_top=False, input_shape=img_shape)
    input_tensor = vgg.input

    # Get convolution and pooling layers from VGG
    c_layers = []
    for idx, layer in enumerate(vgg.layers):
        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            c_layers.append(vgg.layers[idx - 1])
        if idx < (len(vgg.layers) - 4):
            layer.trainable = False


    c3 = tf.keras.layers.Conv2D(n_filters * 16, 3, padding="same", name='start_unet',
        kernel_initializer="he_normal", activation='relu')(vgg.output)
    c3 = tf.keras.layers.Conv2D(n_filters * 16, 3, padding="same",
        kernel_initializer="he_normal", activation='relu')(c3)

    # Expanding Path
    u4 = tf.keras.layers.Conv2DTranspose(n_filters * 8, 3, strides=(2, 2), padding="same")(c3)
    u4 = tf.keras.layers.Concatenate()([u4, c_layers[-1].output])

    c4 = tf.keras.layers.Conv2D(n_filters * 8, 3, padding="same",
        kernel_initializer="he_normal", activation='relu')(u4)
    c4 = tf.keras.layers.Conv2D(n_filters * 8, 3, padding="same",
        kernel_initializer="he_normal", activation='relu')(c4)
    c4 = tf.keras.layers.BatchNormalization()(c4)

    u5 = tf.keras.layers.Conv2DTranspose(n_filters * 4, 3, strides=(2, 2), padding="same")(c4)
    u5 = tf.keras.layers.Concatenate()([u5, c_layers[-2].output])

    c5 = tf.keras.layers.Conv2D(n_filters * 4, 3, padding="same",
        kernel_initializer="he_normal", activation='relu')(u5)
    c5 = tf.keras.layers.Conv2D(n_filters * 4, 3, padding="same",
        kernel_initializer="he_normal", activation='relu')(c5)
    c5 = tf.keras.layers.BatchNormalization()(c5)

    u6 = tf.keras.layers.Conv2DTranspose(n_filters * 4, 3, strides=(2, 2), padding="same")(c5)
    u6 = tf.keras.layers.Concatenate()([u6, c_layers[-3].output])

    c6 = tf.keras.layers.Conv2D(n_filters * 4, 3, padding="same",
        kernel_initializer="he_normal", activation='relu')(u6)
    c6 = tf.keras.layers.Conv2D(n_filters * 4, 3, padding="same",
        kernel_initializer="he_normal", activation='relu')(c6)
    c6 = tf.keras.layers.BatchNormalization()(c6)


    u7 = tf.keras.layers.Conv2DTranspose(n_filters * 2, 3, strides=(2, 2), padding="same")(c6)
    u7 = tf.keras.layers.Concatenate()([u7, c_layers[-4].output])

    c7 = tf.keras.layers.Conv2D(n_filters * 2, 3, padding="same",
        kernel_initializer="he_normal", activation='relu')(u7)
    c7 = tf.keras.layers.Conv2D(n_filters * 2, 3, padding="same",
        kernel_initializer="he_normal", activation='relu')(c7)
    c7 = tf.keras.layers.BatchNormalization()(c7)

    u7 = tf.keras.layers.Conv2DTranspose(n_filters, 3, strides=(2, 2), padding="same")(c7)
    u7 = tf.keras.layers.Concatenate()([u7, c_layers[-5].output])

    c7 = tf.keras.layers.Conv2D(n_filters, 3, padding="same",
        kernel_initializer="he_normal", activation='relu')(u7)
    c7 = tf.keras.layers.Conv2D(n_filters, 3, padding="same",
        kernel_initializer="he_normal", activation='relu')(c7)
    c7 = tf.keras.layers.BatchNormalization()(c7)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(u7)

    model = tf.keras.Model(inputs=[input_tensor], outputs=[outputs])

    return model

#     model.compile(
# #        loss='binary_crossentropy',
#         loss=dice_loss,
#         optimizer=tf.keras.optimizers.SGD(lr=0.1, momentum=0.95),
# #        optimizer=optimizers.Adam(lr=0.3, epsilon=0.1),
#         metrics=[dice_coeff],
#     )

#     return model, 'unet1'
    
if __name__ == "__main__":
    model1()