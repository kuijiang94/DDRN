import os
from PIL import Image
from PIL import ImageChops
import numpy as np
import math

def save_pic(img, save_path, scale, index):
    img.save(os.path.join(save_path,'{:0>3}.png'.format(index)))


def pn_pic(img, save_path, scale):

    img1 = Image.fromarray(img)
    save_pic(img1, save_path, scale, 1)


if __name__ == '__main__':
    data_path = 'F:\\jiangkui\\shujuji\\Kaggle\\'
    #file = os.listdir(data_path)
    save_path = 'F:\\jiangkui\\shujuji\\Kaggle_96\\'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    count =0
    num =0
    files = [
        os.path.join(data_path, filename)
        for filename in os.listdir(data_path)
        if 'png' in filename or 'tif' in filename]
    for filename in files:
        pic_path = os.path.join(data_path, filename)
        img = Image.open(pic_path)

        if count%1==0:
            for i in range(0,img.size[0],96):
                for j in range(0,img.size[1],96):
                    IMG = img.crop([i,j,i+96,j+96])
                    IMG.save(os.path.join(save_path,'{}_{}_{}.png'.format(num,i//96,j//96)))
                    num+=1

        count+=1
        print(count)
