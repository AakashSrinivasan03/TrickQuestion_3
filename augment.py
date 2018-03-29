import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
# -*- coding: utf-8 -*-
import pickle
import argparse
import sys
from datetime import datetime
import os
import pandas as pd
import numpy as np
import math
import scipy as sp
import scipy.special
import sklearn
import sklearn.metrics
import sklearn.preprocessing
import tensorflow as tf
import numpy as np
import numpy.random
import random

'''def rotate_images(X_imgs):
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (28,28,1))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            print("IMG")
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
                X_rotate.append(rotated_img)
        
    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate'''


def rotate_images(X_imgs, start_angle, end_angle, n_images,y):
    X_rotate = []
    ##iterate_at = (end_angle - start_angle) / (n_images - 1)
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (None, 28,28,1))
    radian = tf.placeholder(tf.float32, shape = (len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("IMG")
        if True:
            degrees_angle = numpy.random.uniform(low=-7,high=7)
            radian_value = degrees_angle * 3.14 / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict = {X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype = np.float32)
    print(X_rotate.shape)
    X_rotate=X_rotate.reshape(55000,784)
    numpy.savetxt("train_rotate_1.csv", np.c_[np.zeros(55000),X_rotate,y], delimiter=",",fmt='%d')
    return X_rotate    
	
#rotated_imgs = rotate_images(X_imgs)
def add_salt_pepper_noise(X_imgs,y):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
    
    for X_img in X_imgs_copy:
        # Add Salt noise
        #print("IMG",i-1)
        coords = [np.random.randint(np.minimum(0, i - 1)-1,np.maximum(0, i - 1) , int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(np.minimum(0, i - 1)-1,np.maximum(0, i - 1), int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    numpy.savetxt("train_salt_n_pepper_1.csv", np.c_[np.zeros(55000),X_imgs_copy.reshape(55000,784),y], delimiter=",",fmt='%d')    
    return X_imgs_copy
  
#salt_pepper_noise_imgs = add_salt_pepper_noise(X_imgs)
from math import ceil, floor

def get_translate_parameters(index):
    if index == 0: # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype = np.float32)
        size = np.array([28, ceil(0.8 * 28)], dtype = np.int32)
        w_start = 0
        w_end = int(ceil(0.8 * 28))
        h_start = 0
        h_end = 28
    elif index == 1: # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype = np.float32)
        size = np.array([28, ceil(0.8 * 28)], dtype = np.int32)
        w_start = int(floor((1 - 0.8) * 28))
        w_end = 28
        h_start = 0
        h_end = 28
    elif index == 2: # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * 28), 28], dtype = np.int32)
        w_start = 0
        w_end = 28
        h_start = 0
        h_end = int(ceil(0.8 * 28)) 
    else: # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * 28), 28], dtype = np.int32)
        w_start = 0
        w_end = 28
        h_start = int(floor((1 - 0.8) * 28))
        h_end = 28 

        
    return offset, size, w_start, w_end, h_start, h_end

def translate_images(X_imgs,y):
    offsets = np.zeros((len(X_imgs), 2), dtype = np.float32)
    n_translations = 1
    X_translated_arr = []
    j=[0,1,2,3]
    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), 28, 28, 1), 
                    dtype = np.float32)
            X_translated.fill(1.0) # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(random.choice(j))
            offsets[:, :] = base_offset 
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)
            
            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], \
             w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype = np.float32)
    numpy.savetxt("train_translate_1.csv", np.c_[np.zeros(55000),X_translated_arr.reshape(55000,784),y], delimiter=",",fmt='%d')
    return X_translated_arr
    
#translated_imgs = translate_images(X_imgs)

data_train = pd.read_csv("../fashionmnist/train.csv").iloc[:,:].as_matrix()
print(data_train.shape)
train_X = data_train[:,1:-1]
train_Y= data_train[:,-1]
print(train_Y.shape)
train_X = train_X.reshape(train_X.shape[0],28,28,1)
##rotate_images(train_X)
##translate_images(train_X,train_Y)
print("a")
##add_salt_pepper_noise(train_X,train_Y)
print("a")
##rotate_images(train_X,0,0,1,train_Y)
data_train = pd.read_csv("train_augmented.csv").iloc[:,:].as_matrix()
print(data_train.shape)

