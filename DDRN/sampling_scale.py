import os
from PIL import Image
from PIL import ImageChops
import numpy as np
import math
import cv2

OFF = 1
X=-1

def offset(img, x, y=None):
    img0 = img.copy()
    x = (int)(x)
    if y is None:
        y = x
    else:
        y = (int)(y)
    if x < 0:
        for i in range(img0.shape[1] + x):
            img0[:, i, :] = img0[:, i - x, :]
    else:
        for i in range(img0.shape[1] - 1, x - 1, -1):
            img0[:, i, :] = img0[:, i - x, :]

    if y < 0:
        for i in range(img0.shape[0] + y):
            img0[i, :, :] = img0[i - y, :, :]
    else:
        for i in range(img0.shape[0] - 1, y - 1, -1):
            img0[i, :, :] = img0[i - y, :, :]

    return img0


def save_pic(img, save_path, scale, index):
    img.save(os.path.join(save_path, 'truth', '{:0>3}.png'.format(index)))
    img = img.resize((int(img.size[0]//scale), int(img.size[1]//scale)),
                     Image.CUBIC)
    img.save(os.path.join(save_path, 'input{}'.format(scale), '{:0>3}.png'.format(index)))


def pn_pic(img, save_path, scale):

    img1 = Image.fromarray(img)
    save_pic(img1, save_path, scale, 1)

if __name__ == '__main__':
    data_path = 'F:\\jiangkui\\shujuji\\wxtrain\\caijian\\val10\\'
    file = os.listdir(data_path)
    save_path = 'F:\\jiangkui\\shujuji\\wxtrain\\val10\\'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    count =0
    for scale in range(3,5):
        #print(scale)
        for f in file:
            pic_path = os.path.join(data_path, f)
            img = Image.open(pic_path)
            img = img.crop([0,0,img.size[0]-img.size[0]%12,img.size[1]-img.size[1]%12])
            img = np.asarray(img)
            every_path = os.path.join(save_path, 'val_{:0>5}'.format(
                file.index(f)))
            tr_path = os.path.join(every_path, 'truth')
            in_path = os.path.join(every_path, 'input{}'.format(scale))
            #os.mkdir(in_path)
            if not os.path.exists(every_path):
                os.mkdir(every_path)
                os.mkdir(tr_path)
                os.mkdir(in_path)
            pn_pic(img, every_path, scale)
            count+=1
            print(count)
        print(scale)
        
