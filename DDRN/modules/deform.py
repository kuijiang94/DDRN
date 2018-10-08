'''Utilities'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

def leak_relu(x, alpha=0.02, max_value=None):
    '''ReLU.

    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0.0, dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x
    
def conv2d(input_, filter_shape, strides = [1,1,1,1], padding = 'SAME', activation = leak_relu, scope = None):
    with tf.variable_scope(scope or "conv"):
        w = tf.get_variable(name="w", shape = filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False)) 
        conv = tf.nn.conv2d(input_, w, strides=strides, padding='SAME')
        b = tf.get_variable(name="b", shape = filter_shape[-1], initializer=tf.constant_initializer(0.001))
        y = leak_relu(conv + b, alpha=0.02)
        return y
    
    '''
    Args:
        input_ - 4D tensor
            Normally NHWC format
        filter_shape - 1D array 4 elements
            [height, width, inchannel, outchannel]
        strides - 1D array 4 elements
            default to be [1,1,1,1]
        padding - bool
            Deteremines whether add padding or not
            True => add padding 'SAME'
            Fale => no padding  'VALID'
        activation - activation function
            default to be None
        batch_norm - bool
            default to be False
            used to add batch-normalization
        istrain - bool
            indicate the model whether train or not
        scope - string
            default to be None
    Return:
        4D tensor
        activation(batch(conv(input_)))
    
    with tf.variable_scope(scope or "conv"):
        if padding:
            padding = 'SAME'
        else:
            padding = 'VALID'
        w = tf.get_variable(name="w", shape = filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False)) 
        conv = tf.nn.conv2d(input_, w, strides=strides, padding=padding)
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay = 0.8, is_training=istrain, scope='batch_norm')
            if activation is None:
                return norm
            return activation(norm)
        else:
            b = tf.get_variable(name="b", shape = filter_shape[-1], initializer=tf.constant_initializer(0.001))
            if activation is None:
                return conv + b
            return activation(conv + b)'''

def deform_conv2d(x, offset_shape, filter_shape, activation = None, scope=None):
    '''
    Args:
        x - 4D tensor [batch, i_h, i_w, i_c] NHWC format # i_c 输入数据通道
        offset_shape - list with 4 elements
            [o_h, o_w, o_ic, o_oc] # o_ic 输入通道数； o_oc位移量(x，y)= f_h*f_w  o_h, o_w 搜索框
        filter_shape - list with 4 elements
            [f_h, f_w, f_ic, f_oc]  # f_ic 输入通道数； f_oc输出通道数  f_h, f_w, 滤波器大小5x5
    '''

    batch, i_h, i_w, i_c = x.get_shape().as_list()
    f_h, f_w, f_ic, f_oc = filter_shape
    o_h, o_w, o_ic, o_oc = offset_shape
    assert f_ic==i_c and o_ic==i_c, "# of input_channel should match but %d, %d, %d"%(i_c, f_ic, o_ic) # i_c = f_ic = o_ic
    assert o_oc==2*f_h*f_w, "# of output channel in offset_shape should be 2*filter_height*filter_width but %d and %d"%(o_oc, 2*f_h*f_w)
    # o_oc==2*f_h*f_w  对应着原来batch中每个像素的偏移量，所以是o_oc==2*f_h*f_w，后面展开成2维的对应分量
    with tf.variable_scope(scope or "deform_conv"):
        offset_map = conv2d(x, offset_shape, padding='SAME', scope="offset_conv") # offset_map : [batch, i_h, i_w, o_oc(=2*f_h*f_w)]
    offset_map = tf.reshape(offset_map, [batch, i_h, i_w, f_h, f_w, 2]) # 2维的拆分成两个[0, 1], tile函数将第一维batch复制了i_c倍即batch*i_c
    
    # 获得每个像素点对应当前卷积核内中心坐标下的偏移量  offset_map_h， offset_map_w  (100, 28, 28, 5, 5) pn
    offset_map_h = tf.tile(tf.reshape(offset_map[...,0], [batch, i_h, i_w, f_h, f_w]), [i_c,1,1,1,1]) # offset_map_h [batch*i_c, i_h, i_w, f_h, f_w]
    offset_map_w = tf.tile(tf.reshape(offset_map[...,1], [batch, i_h, i_w, f_h, f_w]), [i_c,1,1,1,1]) # offset_map_w [batch*i_c, i_h, i_w, f_h, f_w]
    
    # 图像块的尺寸范围 卷积核中心点位置(coord_w : [i_h, i_w], coord_h : [i_h, i_w])
    coord_w, coord_h = tf.meshgrid(tf.range(i_w, dtype=tf.float32), tf.range(i_h, dtype=tf.float32)) # coord_w : [i_h, i_w], coord_h : [i_h, i_w]
    
    # 滤波器的尺寸范围 原先卷积核范围内像素相对于中心点的相对坐标 (coord_fw : [f_h, f_w], coord_fh : [f_h, f_w])
    # R = {(-1;-1); (-1; 0); : : : ; (0; 1); (1; 1)}
    coord_fw, coord_fh = tf.meshgrid(tf.range(f_w, dtype=tf.float32), tf.range(f_h, dtype=tf.float32)) # coord_fw : [f_h, f_w], coord_fh : [f_h, f_w]
    '''
    coord_w 
        [[0,1,2,...,i_w-1],...]
    coord_h
        [[0,...,0],...,[i_h-1,...,i_h-1]]
    '''
    # tile函数将第一维batch复制了i_c倍即batch*i_c
    # 强制将coord_h ， coord_w ， coord_fh，coord_fw转化成维度向量[batch*i_c, i_h, i_w, f_h, f_w) 和offset_map_h， offset_map_w一样
   
    # coord_h, coord_w 图像卷积过程中卷积核对应的中心坐标
    coord_h = tf.tile(tf.reshape(coord_h, [1, i_h, i_w, 1, 1]), [batch*i_c, 1, 1, f_h, f_w]) # coords_h [batch*i_c, i_h, i_w, f_h, f_w) 
    coord_w = tf.tile(tf.reshape(coord_w, [1, i_h, i_w, 1, 1]), [batch*i_c, 1, 1, f_h, f_w]) # coords_w [batch*i_c, i_h, i_w, f_h, f_w) 
    
    # coord_fh, coord_fw 图像卷积过程中卷积核区域内像素对应中心坐标的偏移量
    coord_fh = tf.tile(tf.reshape(coord_fh, [1, 1, 1, f_h, f_w]), [batch*i_c, i_h, i_w, 1, 1]) # coords_fh [batch*i_c, i_h, i_w, f_h, f_w) 
    coord_fw = tf.tile(tf.reshape(coord_fw, [1, 1, 1, f_h, f_w]), [batch*i_c, i_h, i_w, 1, 1]) # coords_fw [batch*i_c, i_h, i_w, f_h, f_w) 
    
    # 得到的对应的像素偏移坐标
    # tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
    # 即将索引范围压缩在图像块以内
    coord_h = coord_h + coord_fh + offset_map_h
    coord_w = coord_w + coord_fw + offset_map_w
    coord_h = tf.clip_by_value(coord_h, clip_value_min = 0, clip_value_max = i_h-1) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_w = tf.clip_by_value(coord_w, clip_value_min = 0, clip_value_max = i_w-1) # [batch*i_c, i_h, i_w, f_h, f_w]
    
    # 将补偿后的坐标上下取整得到对应的四组坐标
    # tf.floor(coord_h) 向下取整; tf.ceil(coord_h)向上取整
    coord_hm = tf.cast(tf.floor(coord_h), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w] 
    coord_hM = tf.cast(tf.ceil(coord_h), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]  整数
    coord_wm = tf.cast(tf.floor(coord_w), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_wM = tf.cast(tf.ceil(coord_w), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    
    # tf.transpose(x, [3, 0, 1, 2])将第三维移 i_c 到最前面
    x_r = tf.reshape(tf.transpose(x, [3, 0, 1, 2]), [-1, i_h, i_w]) # [i_c*batch, i_h, i_w]

    bc_index= tf.tile(tf.reshape(tf.range(batch*i_c), [-1,1,1,1,1]), [1, i_h, i_w, f_h, f_w])# bc_index=[i_c*batch, i_h, i_w, f_h, f_w]
    
    # 加上坐标位置 得到四组坐标索引,将bc_index，（coord_hm，coord_wm）的最后一维整合到一起
    # tf.expand_dims(bc_index,-1) Inserts a dimension of 1 into a tensor’s shape.  在第axis位置增加一个维度
    coord_hmwm = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hm,-1), tf.expand_dims(coord_wm,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wm)
    coord_hmwM = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hm,-1), tf.expand_dims(coord_wM,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wM)
    coord_hMwm = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hM,-1), tf.expand_dims(coord_wm,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wm)
    coord_hMwM = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hM,-1), tf.expand_dims(coord_wM,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wM)
    # 在张量x_r中根据索引coord_hmwm得到对应的张量
    var_hmwm = tf.gather_nd(x_r, coord_hmwm) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hmwM = tf.gather_nd(x_r, coord_hmwM) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwm = tf.gather_nd(x_r, coord_hMwm) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwM = tf.gather_nd(x_r, coord_hMwM) # [batch*ic, i_h, i_w, f_h, f_w]
    
    coord_hm = tf.cast(coord_hm, tf.float32) 
    coord_hM = tf.cast(coord_hM, tf.float32) 
    coord_wm = tf.cast(coord_wm, tf.float32)
    coord_wM = tf.cast(coord_wM, tf.float32)
    # 双线性插值
    x_ip = var_hmwm*(coord_hM-coord_h)*(coord_wM-coord_w) + \
           var_hmwM*(coord_hM-coord_h)*(1-coord_wM+coord_w) + \
           var_hMwm*(1-coord_hM+coord_h)*(coord_wM-coord_w) + \
            var_hMwM*(1-coord_hM+coord_h)*(1-coord_wM+coord_w) # [batch*ic, ih, i_w, f_h, f_w]
    x_ip = tf.transpose(tf.reshape(x_ip, [i_c, batch, i_h, i_w, f_h, f_w]), [1,2,4,3,5,0]) # [batch, i_h, f_h, i_w, f_w, i_c]
    x_ip = tf.reshape(x_ip, [batch, i_h*f_h, i_w*f_w, i_c]) # [batch, i_h*f_h, i_w*f_w, i_c]
    with tf.variable_scope(scope or "deform_conv"):
        deform_conv = conv2d(x_ip, filter_shape, strides=[1, f_h, f_w, 1], activation=activation, scope="deform_conv")
    return deform_conv
