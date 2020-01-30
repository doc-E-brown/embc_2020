#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. currentmodule:: 

"""
__author__ = 'Ben Johnston'

import tensorflow as tf

SEED = 0

# def flip_lr(x, y, rate=0.8, seed=SEED):
    
#     if tf.random.uniform(shape=[], seed=seed) > rate:

#         x_lr = tf.image.flip_left_right(x)
#         _y = -1 * (y[:,0] - tf.shape(x)[0])
#         y_lr = tf.stack((tf.transpose(_y), tf.transpose(y[:,1])), axis=1)

#         return x_lr, y_lr
#     else:
#         return x, y

def flip_lr(x, y, rate=0.5, seed=10):
    
    if tf.random.uniform(shape=[], seed=seed) > rate:

        x_lr = tf.image.flip_left_right(x)
        _y = -1 * (y[:,0] - tf.cast(tf.shape(x)[1], tf.float32))
        y_lr = tf.stack((tf.transpose(_y), tf.transpose(y[:,1])), axis=1)

        return x_lr, y_lr
    else:
        return x, y


# def jitter(x, y, bbox, jitter=20, size=100, seed=SEED):

#     offset = tf.random.uniform(
#         shape=[2],
#         minval=-jitter,
#         maxval=jitter,
#         dtype=tf.int32,
#         seed=seed)

#     new_bbox = bbox - offset
    
#     offset = tf.broadcast_to(tf.cast(new_bbox, dtype=tf.float32), [2, 2])
#     y_crop = y - offset 
    
#     x_crop = tf.image.crop_to_bounding_box(
#         x,
#         offset_height=new_bbox[1],
#         offset_width=new_bbox[0],
#         target_width=size,
#         target_height=size,        
#     )
    
#     return x_crop, y_crop


def jitter(x, y, bbox, jitter=30, size=100, seed=SEED):
    
    batch_size = tf.shape(x)[0]
    channels = tf.shape(x)[-1]
    
    offset = tf.random.uniform(
        shape=[batch_size, 2],
        minval=-jitter,
        maxval=jitter,
        seed=seed,
        dtype=tf.float32,
    )
    new_bbox = bbox - offset
    new_bbox = tf.reshape(
        tf.keras.backend.repeat_elements(new_bbox, 2, 0),
        (batch_size, 2, 2)
    )
    
    y_crop = y - new_bbox
    
    new_bbox = tf.cast(new_bbox, tf.int32)
    
    i = tf.constant(0)
    while_condition = lambda i: tf.less(i, batch_size)
    
    x_crop = tf.Variable(
        tf.zeros([batch_size, size, size, channels],
                 dtype=tf.float32))
    
    def crop(i):
        _x = tf.image.crop_to_bounding_box(
            x[i],
            offset_height=new_bbox[i][0][1],
            offset_width=new_bbox[i][0][0],
            target_width=size,
            target_height=size,        
        )
        x_crop[i].assign(_x)
        return [tf.add(i, 1)]
    
    tf.while_loop(while_condition, crop, [i])

    y_crop = y_crop - (size / 2)
    y_crop = y_crop / tf.keras.backend.max(y_crop)

    y_crop = tf.reshape(y_crop, (batch_size, -1))
    
    return x_crop, y_crop


def adjust_image(x, seed=SEED):
    
    x_adjust = tf.image.random_brightness(x, 0.05, seed=seed)
    x_adjust = tf.image.random_contrast(x_adjust, lower=0.8, upper=0.9, seed=seed)
    x_adjust = tf.image.random_hue(x_adjust, 0.01, seed=seed)
    x_adjust = tf.image.random_saturation(x_adjust, 0.9, 1.05, seed=seed)
    
    return x_adjust

def augment_data(x, y, bbox, jit_rate=20, rate=0.5, size=100, seed=SEED):

    i = tf.constant(0)
    batch_size = tf.shape(x)[0]

    while_condition = lambda i: tf.less(i, batch_size)
    x_flip = tf.Variable(
        tf.zeros(tf.shape(x),
                 dtype=tf.float32))
    y_flip = tf.Variable(
        tf.zeros(tf.shape(y), dtype=tf.float32)
    )

    def _flip(i):

        _x, _y = flip_lr(x[i], y[i], rate, seed)
        x_flip[i].assign(_x)
        y_flip[i].assign(_y)
        return [tf.add(i, 1)]


    x_aug, y_aug = jitter(x, y, bbox, jit_rate, size, seed)

    x_adjust = tf.Variable(
        tf.zeros(tf.shape(x_aug),
                 dtype=tf.float32))

    i = tf.constant(0)
    def _adjust(i):
        _x = adjust_image(x_aug[i], seed)
        x_adjust[i].assign(_x)

        return [tf.add(i, 1)]

    tf.while_loop(while_condition, _adjust, [i])

    x_adjust = x_adjust / 255.0

    return x_adjust, y_aug
    # return x_aug, y_aug
