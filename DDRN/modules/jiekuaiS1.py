#from PIL import Image
import numpy as np
#import os
#import matplotlib.pyplot as plt
#import ps
import tensorflow as tf
import random
import math

def crop(image, crop_size, image_size):
    roi1 = []
    #roi2 = []
    roi1 = tf.slice(image,[0,0,crop_size,crop_size,0],[16,1,image_size,image_size,3])
    #roi2 = tf.slice(gt,[0,0,batch_size,batch_size,0],[16,1,image_size,image_size,3])
    
    #image[:, :, crop_size:image.shape[2] - crop_size, crop_size:image.shape[3] - crop_size, :])
    return roi1

def extend(image, extend_size):
    num, t,w,h,ch=image.shape
    EX = extend_size*2
    #img = cv2.imread(pic_path)
    w = int(image.shape[2])
    h = int(image.shape[3])
    #img_ex = np.zeros([num, t, w + EX, h + EX, 3])
    #print(img_ex.shape)
    #image0 = tf.pad(image, [[0,24],[0,39],[4,4],[4,4],[0,37]], mode="CONSTANT")
    img_ex = image
    slice1 = tf.slice(img_ex,[0,0,0,0,0],[16,1,extend_size,h,3])
    slice2 = tf.slice(img_ex,[0,0,w-extend_size,0,0],[16,1,extend_size,h,3])
    img_ex =tf.concat([slice1, img_ex], 2)
    img_ex =tf.concat([img_ex, slice2], 2)
    slice3 = tf.slice(img_ex,[0,0,0,0,0],[16,1,w+EX,extend_size,3])
    slice4 = tf.slice(img_ex,[0,0,0,h-extend_size,0],[16,1,w+EX,extend_size,3])
    img_ex =tf.concat([slice3, img_ex], 3)
    img_ex =tf.concat([img_ex, slice4], 3)
    #print(image0.shape)
    #w1 = int(img_ex.shape[2])
    #h1 = int(img_ex.shape[3])
    #a = int(EX // 2)
    '''#for i in range(0, w):
        #for j in range(0, h):
            #img_ex[:, :, i + EX // 2, j + EX // 2, :] = image0[:, :, i, j, :]
    for i in range(0, a):
        img_ex[:, :, i, a:a + h, :] = img_ex[:, :, a + i, a:a + h, :]
    for i in range(a + w, w1):
        img_ex[:, :, i, a:a + h, :] = img_ex[:, :, i - a, EX // 2:a + h, :]
    for i in range(0, a):
        img_ex[:, :, :, i, :] = img_ex[:, :, :, a + i, :]
    for i in range(a + h, h1):
        img_ex[:, :, :, i, :] = img_ex[:,:, i - a, :]
        
    img_ex = tf.slice(img_ex,[0,0,0,0,0],[16,1,w+EX,h+EX,3])'''
    
    return image, img_ex

