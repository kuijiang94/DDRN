#from PIL import Image
import numpy as np
#import os
#import matplotlib.pyplot as plt
#import ps
# import tensorflow as tf
import random

def batch_image(image, gt, batch_size=40):
    '''#batch_size = 40
    num, t,rows,cols,ch=x.shape
    roi_1 = np.zeros((num, t, rows, cols, ch))
    roi_2 = np.zeros((num, t, rows, cols, ch))
    
    r=random.randint(0,x.shape[2]-batch_size-1) 
    c=random.randint(0,x.shape[3]-batch_size-1)
    #for i in range(r, r+batch_size):
    #for j in range(c, c+batch_size):
    roi_1[:, :, 0:batch_size, 0:batch_size, :] = x[:, :, r:r+batch_size, c:c+batch_size, :]
    roi_2[:, :, 0:batch_size, 0:batch_size, :] = y[:, :, r:r+batch_size, c:c+batch_size, :]
    #roi_2[:, :, i-r, j-c, :] = y[:, :, i, j, :]
            
    #print(roi_1.shape)
    return roi_1[:, :, :, :, :],roi_2[:, :, :, :, :]
    '''
    #roi_1 = x.copy()
    #roi_1 = map(lambda x: x, image)
    #roi_2 = map(lambda y: y, gt)
    #roi_2 = y.copy() 
    num, t,rows,cols,ch=image.shape    
    r=random.randint(0,rows-batch_size) 
    c=random.randint(0,cols-batch_size)
    '''for i in range(0, r):
        image = np.delete(image, i, axis=2) 
        gt = np.delete(gt, i, axis=2)
    for j in range(batch_size,rows-r):
        image = np.delete(image, j, axis=2)
        gt = np.delete(gt, j, axis=2)
    for m in range(0, c):
        image = np.delete(image, m, axis=3)
        gt = np.delete(gt, m, axis=3)
    for n in range(batch_size, cols-c):
        image = np.delete(image, n, axis=3)
        gt = np.delete(gt, n, axis=3)
    return image, gt'''
    roi1 = []
    roi2 = []
    for i in range(rows):
        if 0 <= i:
            if i<r:
                roi1 = np.delete(image, i, axis=2) 
                roi2 = np.delete(gt, i, axis=2)
        #if (0 =< i and i<r) or ((r+batch_size-1)=<i and i<rows):
    print(roi1.shape)
    a = roi1.shape[2]
    roi3 = []
    roi4 = []
    for i in range(a):
        if (batch_size)<=i:
            if i<a:
                roi3 = np.delete(roi1, i, axis=2) 
                roi4 = np.delete(roi2, i, axis=2)
    b = roi3.shape[3]
    roi5 = []
    roi6 = []
    for j in range(b):
        if 0<=j:
            if j<c:
                roi5 = np.delete(roi3, j, axis=3) 
                roi6 = np.delete(roi4, j, axis=3)
        #if (0=<j and i<c) or ((c+batch_size-1)=<j and j<cols):
    d = roi5.shape[3]
    roi7 = []
    roi8 = []
    for j in range(d):
        if (batch_size)<=j:
            if j<d:
                roi7 = np.delete(roi5, j, axis=3) 
                roi8 = np.delete(roi6, j, axis=3)
    return roi7, roi8

