#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. currentmodule:: 

"""
__author__ = 'Ben Johnston'

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

SEED = 0

def flip_lr(x, y, rate=0.5, seed=SEED):
    
    if tf.random.uniform(shape=[], seed=seed) > rate:

        x_lr = tf.image.flip_left_right(x)
        _y = -1 * (y[:,0] - tf.cast(tf.shape(x)[1], tf.float32))
        y_lr = tf.stack((tf.transpose(_y), tf.transpose(y[:,1])), axis=1)

        return x_lr, y_lr
    else:
        return x, y


def jitter(x, y, bbox, jit_rate=5, size=100, seed=0):

    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]

    offset = np.random.randint(-jit_rate, jit_rate, 2)

    new_bbox = np.zeros(4)

    new_bbox[:2] = bbox[:2] - offset
    new_bbox[2] = new_bbox[0] + box_width
    new_bbox[3] = new_bbox[1] + box_height

    y_crop = y - new_bbox[:2]
    
    resize_ratio = size / box_width
    

    img = Image.fromarray(x.astype(np.uint8))
    img = img.crop(new_bbox).resize((size, size))
    
    mu_crop = y_crop.mean(axis=0)
    y_crop -= mu_crop
    y_crop *= resize_ratio
    y_crop += (mu_crop * resize_ratio)

    return np.array(img), y_crop



# def jitter(x, y, bbox, jitter=30, size=100, seed=SEED):
    
#     batch_size = tf.shape(x)[0]
#     channels = tf.shape(x)[-1]

#     offset = tf.random.uniform(
#         shape=[batch_size, 2],
#         minval=-jitter,
#         maxval=jitter,
#         seed=seed,
#         dtype=tf.float32,
#     )
#     new_bbox = bbox - offset
#     new_bbox = tf.reshape(
#         tf.keras.backend.repeat_elements(new_bbox, 2, 0),
#         (batch_size, 2, 2)
#     )
    
#     y_crop = y - new_bbox
    
#     new_bbox = tf.cast(new_bbox, tf.int32)
    
#     i = tf.constant(0)
#     while_condition = lambda i: tf.less(i, batch_size)
    
#     x_crop = tf.Variable(
#         tf.zeros([batch_size, size, size, channels],
#                  dtype=tf.float32))
    
#     def crop(i):
#         _x = tf.image.crop_to_bounding_box(
#             x[i],
#             offset_height=new_bbox[i][0][1],
#             offset_width=new_bbox[i][0][0],
#             target_width=size,
#             target_height=size,        
#         )
#         x_crop[i].assign(_x)
#         return [tf.add(i, 1)]
    
#     tf.while_loop(while_condition, crop, [i])

   
#     return x_crop, y_crop


def adjust_image(x, seed=SEED):
    
    x_adjust = tf.image.random_brightness(x, 0.05, seed=seed)
    x_adjust = tf.image.random_contrast(x_adjust, lower=0.8, upper=0.9, seed=seed)
    x_adjust = tf.image.random_hue(x_adjust, 0.01, seed=seed)
    x_adjust = tf.image.random_saturation(x_adjust, 0.9, 1.05, seed=seed)
    
    return x_adjust

def augment_data(x, y, bbox, jit_rate=20, rate=0.5, size=100, seed=SEED):

    _x, _y = jitter(x, y, bbox, jit_rate, size)

    _x = tf.convert_to_tensor(_x, dtype=tf.float32)
    _y = tf.convert_to_tensor(_y, dtype=tf.float32)

    _x, _y = flip_lr(_x, _y, rate, seed)

    _x = adjust_image(_x, seed)

    return _x.numpy(), _y.numpy()

# def augment_data(x, y, bbox, jit_rate=20, rate=0.5, size=100, seed=SEED):

#     i = tf.constant(0)
#     batch_size = tf.shape(x)[0]

#     while_condition = lambda i: tf.less(i, batch_size)
#     x_flip = tf.Variable(
#         tf.zeros(tf.shape(x),
#                  dtype=tf.float32))
#     y_flip = tf.Variable(
#         tf.zeros(tf.shape(y), dtype=tf.float32)
#     )

#     def _flip(i):

#         _x, _y = flip_lr(x[i], y[i], rate, seed)
#         x_flip[i].assign(_x)
#         y_flip[i].assign(_y)
#         return [tf.add(i, 1)]


#     x_aug, y_aug = jitter(x, y, bbox, jit_rate, size, seed)

#     x_adjust = tf.Variable(
#         tf.zeros(tf.shape(x_aug),
#                  dtype=tf.float32))

#     i = tf.constant(0)
#     def _adjust(i):
#         _x = adjust_image(x_aug[i], seed)
#         x_adjust[i].assign(_x)

#         return [tf.add(i, 1)]

#     tf.while_loop(while_condition, _adjust, [i])

#     return x_adjust, y_aug

def create_mask(y, width, height, radius=2):
    
    mask = Image.new("L", (width, height))
    draw = ImageDraw.Draw(mask)
    
    for _y in y:
        
        draw.ellipse(
            [
                _y[0] - radius, _y[1] - radius,
                _y[0] + radius, _y[1] + radius
            ],
            fill='white',
            outline='white'
        )
        
    return mask