import os
import time
import glob
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from tflearn.layers.conv import global_avg_pool
import scipy.misc
import random
import subprocess
import cv2
from datetime import datetime

from modules.videosr_ops import *
from modules.utils import *
from modules.SSIM_Index import *
import modules.ps

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#DATA_TEST = './data/test/calendar'
#DATA_TRAIN = './data/train/'


def weight_loader(model_weights=None, tfvars=None):
    if tfvars is None:
        tfvars = tf.trainable_variables()
    loaders = list()
    for v in tfvars:
        layer_name, param_name = v.name.split('/')[-2:]
        param_name = param_name.split(':')[0]
        if (layer_name in model_weights) and (param_name in model_weights[layer_name]):
            print('initializing: ', layer_name, '\t', param_name, model_weights[layer_name][param_name].shape)

            # if 'conv'
            loaders.append(tf.assign(v, model_weights[layer_name][param_name], name='as_' + layer_name))
    return tf.group(*loaders)


class VIDEOSR(object):
    def __init__(self):
        self.num_frames = 1
        self.num_block = 15
        self.crop_size = 32
        self.scale_factor = 4

        self.max_steps = int(1e6)
        self.batch_size = 16
        self.eval_batch_size=10
        self.ratio = 1.0
        self.lstm_loss_weight = np.linspace(0.5, 1.0, self.num_frames)
        self.lstm_loss_weight = self.lstm_loss_weight / np.sum(self.lstm_loss_weight)
        self.learning_rate = 1e-3
        self.beta1 = 0.9
        self.beta2=0.999
        self.decay_steps=1e4
        self.train_dir = './videosr_log/DDRN_15_X4'

        self.pathlist = open('./data/filelist_train.txt', 'rt').read().splitlines()
        random.shuffle(self.pathlist)
        self.vallist = open('./data/filelist_val.txt', 'rt').read().splitlines()

    def input_producer(self, batch_size=16):
        def read_data():
            idx0 = self.num_frames // 2
            data_seq = tf.random_crop(self.data_queue, [2, self.num_frames])
            input = tf.stack(
                [tf.image.decode_png(tf.read_file(data_seq[0][i]), channels=3) for i in range(self.num_frames)])
            gt = tf.stack([tf.image.decode_png(tf.read_file(data_seq[1][idx0]), channels=3)])
            # gt = tf.stack([tf.image.decode_png(tf.read_file(data_seq[1][i]), channels=3) for i in range(self.num_frames)])
            input, gt = prepprocessing(input, gt)
            print('Input producer shape: ', input.get_shape(), gt.get_shape())
            return input, gt

        def prepprocessing(input, gt=None):
            input = tf.cast(input, tf.float32) / 255.0
            gt = tf.cast(gt, tf.float32) / 255.0

            shape = tf.shape(input)[1:]
            size = tf.convert_to_tensor([self.crop_size, self.crop_size, 3], dtype=tf.int32, name="size")
            check = tf.Assert(tf.reduce_all(shape >= size), ["Need value.shape >= size, got ", shape, size])
            shape = control_flow_ops.with_dependencies([check], shape)

            limit = shape - size + 1
            offset = tf.random_uniform(tf.shape(shape), dtype=size.dtype, maxval=size.dtype.max, seed=None) % limit

            offset_in = tf.concat([[0], offset], axis=-1)
            size_in = tf.concat([[self.num_frames], size], axis=-1)
            input = tf.slice(input, offset_in, size_in)
            offset_gt = tf.concat([[0], offset[:2] * self.scale_factor, [0]], axis=-1)
            size_gt = tf.concat([[1], size[:2] * self.scale_factor, [3]], axis=-1)
            gt = tf.slice(gt, offset_gt, size_gt)

            input.set_shape([self.num_frames, self.crop_size, self.crop_size, 3])
            gt.set_shape([1, self.crop_size * self.scale_factor, self.crop_size * self.scale_factor, 3])
            return input, gt

        with tf.variable_scope('input'):
            inList_all = []
            gtList_all = []
            for dataPath in self.pathlist:
                inList = sorted(glob.glob(os.path.join(dataPath, 'input{}/*.png'.format(self.scale_factor))))
                gtList = sorted(glob.glob(os.path.join(dataPath, 'truth/*.png')))
                inList_all.append(inList)
                gtList_all.append(gtList)
            inList_all = tf.convert_to_tensor(inList_all, dtype=tf.string)
            gtList_all = tf.convert_to_tensor(gtList_all, dtype=tf.string)

            self.data_queue = tf.train.slice_input_producer([inList_all, gtList_all], capacity=batch_size*2)
            input, gt = read_data()
            batch_in, batch_gt = tf.train.batch([input, gt], batch_size=batch_size, num_threads=3, capacity=batch_size*2)
        return batch_in, batch_gt

    def forward(self, frames_lr, is_training=True, reuse=False):
        num_batch, num_frame, height, width, num_channels = frames_lr.get_shape().as_list()
        out_height = height * self.scale_factor
        out_width = width * self.scale_factor
        # calculate flow
        idx0 = num_frame // 2
        frames_y = frames_lr
        frame_ref_y = frames_y[:, int(idx0), :, :, :]
        self.frames_y = frames_y
        self.frame_ref_y = frame_ref_y

        frame_bic_ref = tf.image.resize_images(frame_ref_y, [out_height, out_width], method=2)
        tf.summary.image('inp_0', im2uint8(frames_y[0, :, :, :, :]), max_outputs=3)
        tf.summary.image('bic', im2uint8(frame_bic_ref), max_outputs=3)

        x_unwrap = []
        def Global_avg_pooling(x):
            return global_avg_pool(x, name='Global_avg_pooling')
        
        def Sigmoid(x) :
            return tf.nn.sigmoid(x)

        #frames_y = tf.gather(frames_y, indices=[0, 2, 1], axis=1)
        for i in range(num_frame):
            if i > 0 and not reuse:
                reuse = True
            frame_i = frames_y[:, i, :, :, :]
            print('Build model - frame_{}'.format(i), frame_i.get_shape())
            frame_i_fw = frame_i
            max_feature=64
            slice_ratio=self.ratio
            with tf.variable_scope('srmodel', reuse=reuse) as scope_sr:
                with slim.arg_scope([slim.conv2d], activation_fn=prelu , stride=1,
                                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                    biases_initializer=tf.constant_initializer(0.0)), \
                     slim.arg_scope([slim.batch_norm], center=True, scale=False, updates_collections=None,
                                    activation_fn=prelu, epsilon=1e-5, is_training=is_training):
                    # conv2d_bn = lambda inputs, num_outputs, kernel_size, scope: \
                    #     slim.batch_norm(slim.conv2d(inputs, num_outputs, kernel_size, scope='conv' + scope), scope='bn' + scope)
                    # rnn_input = tf.concat([frame_bic_ref, frame_i_fw], 3)
                    rnn_input = tf.concat([frame_i_fw], 3)
                    conv0 = slim.conv2d(rnn_input, max_feature*2, [3, 3], scope='enc0')
                    conv1 = slim.conv2d(conv0, max_feature+max_feature//2, [3, 3], scope='enc1')

                    resblock_in1 = conv1
                    reuse_block=False
                    for p in range(15):
                        with tf.variable_scope('srmodel_block', reuse=reuse_block) as scope_sr_block:
                            #000
                            conv0_0=slim.conv2d(resblock_in1, max_feature, [3, 3], scope='conv0_0_{}'.format(p))
                            conv1_0=slim.conv2d(conv0_0, max_feature, [3, 3], scope='conv1_0_{}'.format(p))
                            conv2_0=slim.conv2d(conv0_0, max_feature, [3, 3], scope='conv2_0_{}'.format(p))
                            
                            all_0=all_1=all_2=tf.concat([conv0_0,conv1_0,conv2_0],3)
                            
                            
                            for k in range(2):
                                conv0_1_0=slim.conv2d(all_0, max_feature, [1, 1], scope='conv0_{}_0_{}'.format(k+1,p))
                                conv1_1_0=slim.conv2d(all_1, max_feature, [1, 1], scope='conv1_{}_0_{}'.format(k+1,p))
                                conv2_1_0=slim.conv2d(all_2, max_feature, [1, 1], scope='conv2_{}_0_{}'.format(k+1,p))
                                
                                conv0_1=slim.conv2d(conv0_1_0, max_feature, [3, 3], scope='conv0_{}_{}'.format(k+1,p))
                                conv1_1=slim.conv2d(conv1_1_0, max_feature, [3, 3], scope='conv1_{}_{}'.format(k+1,p))
                                conv2_1=slim.conv2d(conv2_1_0, max_feature, [3, 3], scope='conv2_{}_{}'.format(k+1,p))
                                
                                all_0=tf.concat([conv0_1_0,conv0_1,conv1_1,conv2_1],3)
                                all_1=tf.concat([conv1_1_0,conv0_1,conv1_1,conv2_1],3)
                                all_2=tf.concat([conv2_1_0,conv0_1,conv1_1,conv2_1],3)
                                
                            block_out1 = tf.concat([conv0_1,conv1_1,conv2_1],3)
                            block_out1 = slim.conv2d(block_out1, max_feature+max_feature//2, [1, 1], scope='blockout_{}'.format(p))
                            resblock_in1 +=block_out1
                    if self.scale_factor==2 or self.scale_factor==3:
                        conv7 = slim.conv2d(resblock_in1, max_feature, [3, 3], scope='conv7')
                        rnn_out = slim.conv2d(conv7, (self.scale_factor**2)*3, [3, 3], activation_fn=None, scope='rnn_out')
                        
                        rnn_out = modules.ps._PS(rnn_out, 2, 3)
                        rnn_out = rnn_out + frame_bic_ref
                    if self.scale_factor==4:
                        conv7 = slim.conv2d(resblock_in1, max_feature, [3, 3], scope='conv7')
                        larger = modules.ps._PS(conv7, 2, max_feature//4)

                        conv8= slim.conv2d(larger, max_feature, [3, 3], scope='conv8')
                        rnn_out = slim.conv2d(conv8, ((self.scale_factor//2)**2)*3, [3, 3], activation_fn=None, scope='rnn_out')
                        
                        rnn_out = modules.ps._PS(rnn_out, 2, 3)
                        rnn_out = rnn_out + frame_bic_ref
                if i >= 0:
                    x_unwrap.append(rnn_out)
                if i == 0:
                    tf.get_variable_scope().reuse_variables()

        x_unwrap = tf.stack(x_unwrap, 1)
        return x_unwrap
        
    def build_model(self):
        frames_lr, frame_gt = self.input_producer(batch_size=self.batch_size)
        n, t, h, w, c = frames_lr.get_shape().as_list()
        output = self.forward(frames_lr)

        # calculate loss
        # reconstruction loss
        frame_gt_y = frame_gt
        mse = tf.reduce_mean(tf.sqrt((output - frame_gt_y) ** 2+(1e-3)**2), axis=[0, 2, 3, 4])
        #mse = tf.reduce_mean((output - frame_gt_y) ** 2, axis=[0, 2, 3, 4])
        self.mse = mse
        for i in range(self.num_frames):
            tf.summary.scalar('mse_%d' % i, mse[i])
        tf.summary.image('out_0', im2uint8(output[0, :, :, :, :]), max_outputs=3)
        tf.summary.image('res', im2uint8(output[:, -1, :, :, :]), max_outputs=3)
        tf.summary.image('gt', im2uint8(frame_gt_y[:, 0, :, :, :]), max_outputs=3)

        self.loss_mse = tf.reduce_sum(mse * self.lstm_loss_weight)
        tf.summary.scalar('loss_mse', self.loss_mse)


        #tf.summary.image('uv', flowToColor(self.uv[0, :, :, :, :], maxflow=3.0), max_outputs=3)

        # total loss
        self.loss = self.loss_mse
        tf.summary.scalar('loss_all', self.loss)

    def evaluation(self):
        print('Evaluating ...')
        inList_all = []
        gtList_all = []
        for dataPath in self.vallist:
            inList = sorted(glob.glob(os.path.join(dataPath, 'input{}/*.png'.format(self.scale_factor))))
            gtList = sorted(glob.glob(os.path.join(dataPath, 'truth/*.png')))
            inList_all.append(inList)
            gtList_all.append(gtList)

        if not hasattr(self, 'sess'):
            sess = tf.Session()
            self.flownets.load_easyflow(sess, os.path.join('./easyflow_log/model1', 'checkpoints'))
        else:
            sess = self.sess

        # out_h = 528
        # out_w = 960
        out_h = 512
        out_w = 640
        in_h = out_h // self.scale_factor
        in_w = out_w // self.scale_factor
        if not hasattr(self, 'eval_input'):
            self.eval_input = tf.placeholder(tf.float32, [self.eval_batch_size, self.num_frames, in_h, in_w, 3])
            self.eval_gt = tf.placeholder(tf.float32, [self.eval_batch_size, 1, out_h, out_w, 3])
            self.eval_output = self.forward(self.eval_input, is_training=False, reuse=True)

            # calculate loss
            frame_gt_y = self.eval_gt
            self.eval_mse = tf.reduce_mean((self.eval_output[:, :, :, :, :] - frame_gt_y) ** 2, axis=[2, 3, 4])

        batch_in = []
        batch_gt = []
        radius = self.num_frames // 2
        mse_acc = None
        ssim_acc = None
        batch_cnt = 0
        #batch_name=[]
        for inList, gtList in zip(inList_all, gtList_all):
            for idx0 in range(self.num_frames//2, len(inList), 6):
                #batch_name.append(gtList[idx0])
                inp = [scipy.misc.imread(inList[0]) for i in range(idx0 - radius, 0)]
                inp.extend([scipy.misc.imread(inList[i]) for i in range(max(0, idx0 - radius), idx0)])
                inp.extend([scipy.misc.imread(inList[i]) for i in range(idx0, min(len(inList), idx0 + radius + 1))])
                inp.extend([scipy.misc.imread(inList[-1]) for i in range(idx0 + radius, len(inList) - 1, -1)])
                inp = [i[:in_h, :in_w, :].astype(np.float32) / 255.0 for i in inp]
                gt = [scipy.misc.imread(gtList[idx0])]
                gt = [i[:out_h, :out_w, :].astype(np.float32) / 255.0 for i in gt]

                batch_in.append(np.stack(inp, axis=0))
                batch_gt.append(np.stack(gt, axis=0))

                if len(batch_in) == self.eval_batch_size:
                    batch_cnt += self.eval_batch_size
                    batch_in = np.stack(batch_in, 0)
                    batch_gt = np.stack(batch_gt, 0)
                    mse_val, eval_output_val = sess.run([self.eval_mse, self.eval_output],
                                                        feed_dict={self.eval_input: batch_in, self.eval_gt: batch_gt})
                    ssim_val = np.array(
                        [[compute_ssim(eval_output_val[ib, it, :, :, 0], batch_gt[ib, 0, :, :, 0], l=1.0)
                          for it in range(self.num_frames)] for ib in range(self.eval_batch_size)])
                    if mse_acc is None:
                        mse_acc = mse_val
                        ssim_acc = ssim_val
                    else:
                        mse_acc = np.concatenate([mse_acc, mse_val], axis=0)
                        ssim_acc = np.concatenate([ssim_acc, ssim_val], axis=0)
                    #print(batch_name)
                    #batch_name=[]
                    batch_in = []
                    batch_gt = []
                    print('\tEval batch {} - {} ...'.format(batch_cnt, batch_cnt + self.eval_batch_size))

        # if len(batch_in) is not 0:
        #     batch_cnt += len(batch_in)
        #     batch_in = np.stack(batch_in, 0)
        #     batch_gt = np.stack(batch_gt, 0)
        #     mse_val = sess.run(mse, feed_dict={frames_lr: batch_in, frame_gt: batch_gt})
        #     mse_acc += mse_val
        #     batch_in = []
        #     batch_gt = []
        psnr_acc = 10 * np.log10(1.0 / mse_acc)
        mse_avg = np.mean(mse_acc, axis=0)
        psnr_avg = np.mean(psnr_acc, axis=0)
        ssim_avg = np.mean(ssim_acc, axis=0)
        for i in range(mse_avg.shape[0]):
            tf.summary.scalar('val_mse{}'.format(i), tf.convert_to_tensor(mse_avg[i], dtype=tf.float32))
        print('Eval MSE: {}, PSNR: {}'.format(mse_avg, psnr_avg))
        # write to log file
        with open(os.path.join(self.train_dir, 'eval_log.txt'), 'a+') as f:
            f.write('Iter {} - MSE: {}, PSNR: {}, SSIM: {}\n'.format(sess.run(self.global_step), mse_avg, psnr_avg,
                                                                     ssim_avg))
        np.save(os.path.join(self.train_dir, 'eval_iter_{}'.format(sess.run(self.global_step))),
                {'mse': mse_acc, 'psnr': psnr_acc, 'ssim': ssim_acc})

    def train(self):
        def train_op_func(loss, var_list, is_gradient_clip=False):
            if is_gradient_clip:
                train_op = tf.train.AdamOptimizer(lr, self.beta1)
                grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
                rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                train_op = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
            else:
                # train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss, var_list=var_list, global_step=global_step)
                train_op = tf.train.AdamOptimizer(lr,self.beta1,self.beta2).minimize(loss, var_list=var_list, global_step=global_step)
            return train_op

        """Train video sr network"""
        global_step = tf.Variable(initial_value=0, trainable=False)
        self.global_step = global_step

        # Create folder for logs
        if not tf.gfile.Exists(self.train_dir):
            tf.gfile.MakeDirs(self.train_dir)

        self.build_model()
        decay_steps = self.decay_steps
        #lr = tf.train.polynomial_decay(self.learning_rate, global_step, decay_steps, end_learning_rate=1e-4, power=0.9)
            #tf.train.exponential_decay(self.10*self.learning_rate,global_step, decay_steps, decay_rate=0.1,staircase=False)
        lr=tf.train.exponential_decay(self.learning_rate,global_step, decay_steps, decay_rate=0.5,staircase=False)+1e-5
        tf.summary.scalar('learning_rate', lr)
        vars_all = tf.trainable_variables()
        vars_sr = [v for v in vars_all if 'srmodel' in v.name]
        vars_flow = [v for v in vars_all if 'flow' in v.name]
        train_all = train_op_func(self.loss, vars_all, is_gradient_clip=True)
        train_sr = train_op_func(self.loss_mse, vars_sr, is_gradient_clip=False)

        sess = tf.Session()
        self.sess = sess
        sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        #self.flownets.load_easyflow(sess, os.path.join('./easyflow_log/model1', 'checkpoints'))
        self.load(sess, os.path.join(self.train_dir, 'checkpoints'))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)

        for step in range(sess.run(global_step), self.max_steps):
            if step < 10000:
                train_op = train_sr
            elif step < 20000:
                train_op = train_sr
            else:
                train_op = train_sr

            start_time = time.time()
            _, loss_value, mse_value, loss_mse_value = sess.run(
                [train_op, self.loss, self.mse, self.loss_mse])
            duration = time.time() - start_time
            # print (loss_value)
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 5 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = (%.3f: %.3f), mse = %s  (%.1f data/s; %.3f '
                              's/bch)')
                print((format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_value, loss_mse_value,
                                      str(mse_value), examples_per_sec, sec_per_batch)))

            if step % 50 == 0:
                # summary_str = sess.run(summary_op, feed_dict={inputs:batch_input, gt:batch_gt})
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            if step % 500 == 0:
                self.evaluation()

            # Save the model checkpoint periodically.
            if step % 500 == 499 or (step + 1) == self.max_steps:
                checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                self.save(sess, checkpoint_path, step)
                #if step<10000 or step>20000:
                    #self.flownets.save(sess, './easyflow_log/model1/checkpoints', step)

    def save(self, sess, checkpoint_dir, step):
        model_name = "videoSR.model"
        # model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        #checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading SR checkpoints...")
        model_name = "videoSR.model"

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading checkpoints...{} Success".format(ckpt_name))
            return True
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def test(self, path=None, scale_factor=4, num_frames=1):

        import scipy.misc
        import time
        import math
        file=os.listdir(path)
        file=[os.path.join(path,f) for f in file]
        file=[f for f in file if os.path.isdir(f)]
        startid=0
        cnt=0
        for pat in file:
            if file.index(pat)>=startid:
                '''
                for j in range(20):
                    time.sleep(5)
                    print(time.ctime())'''
                dataPath = pat
                #inList = sorted(glob.glob(os.path.join(dataPath, 'input{}/*.png').format(scale_factor)))
                inList = sorted(glob.glob(os.path.join(dataPath, 'truth/*.png')))
                inp = [scipy.misc.imread(i).astype(np.float32) / 255.0 for i in inList]

                print('Testing path: {}'.format(dataPath))
                print('# of testing frames: {}'.format(len(inList)))

                '''
                DATA_TEST_OUT = pat + \
                    '_SR_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
                os.mkdir(DATA_TEST_OUT)'''

                #cnt = 0
                self.scale_factor = scale_factor
                if file.index(pat)==startid:
                    reuse = False
                else:
                    reuse=True

                for idx0 in range(self.num_frames//2,self.num_frames//2+1):#:(len(inList))
                    cnt += 1
                    T = num_frames // 2

                    imgs = [inp[0] for i in np.arange(idx0 - T, 0)]
                    imgs.extend([inp[i] for i in np.arange(max(0, idx0 - T), idx0)])
                    imgs.extend([inp[i] for i in np.arange(
                        idx0, min(len(inList), idx0 + T + 1))])
                    imgs.extend([inp[-1]
                                 for i in np.arange(idx0 + T, len(inList) - 1, -1)])

                    # imgs = [i.astype(np.float32) / 255.0 for i in imgs]
                    # imgs = imgs * num_frames
                    dims = imgs[0].shape
                    if len(dims) == 2:
                        imgs = [np.expand_dims(i, -1) for i in imgs]
                    h, w, c = imgs[0].shape
                    out_h = h * scale_factor
                    out_w = w * scale_factor
                    padh = int(math.ceil(h / 4.0) * 4.0 - h)
                    padw = int(math.ceil(w / 4.0) * 4.0 - w)
                    imgs = [np.pad(i, [[0, padh], [0, padw], [0, 0]], 'edge')
                            for i in imgs]
                    imgs = np.expand_dims(np.stack(imgs, axis=0), 0)

                    if idx0 == self.num_frames//2:
                        # frames_lr = tf.convert_to_tensor(imgs, tf.float32)
                        frames_lr = tf.placeholder(dtype=tf.float32, shape=imgs.shape)
                        frames_ref_ycbcr = frames_lr[:, T:T + 1, :, :, :]
                        frames_ref_ycbcr = tf.tile(
                            frames_ref_ycbcr, [1, num_frames, 1, 1, 1])
                        output = self.forward(frames_lr, is_training=False, reuse=reuse)

                        output_rgb = output
                        output = output[:, :, :out_h, :out_w, :]
                        output_rgb = output_rgb[:, :, :out_h, :out_w, :]

                    if cnt == 1:
                        sess = tf.Session()
                        reuse = True
                        # summary_op = tf.summary.merge_all()
                        # summary_writer = tf.summary.FileWriter('E:/PyProject/out0', sess.graph, flush_secs=30)
                        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
                        self.load(sess, os.path.join(self.train_dir, 'checkpoints'))
                        #self.flownets.load_easyflow(sess, os.path.join(
                            #'./easyflow_log/model1', 'checkpoints'))
                    # frames_ref_warp = imwarp_backward(self.uv, tf.tile(tf.expand_dims(self.frame_ref_y, 1), [1, num_frames, 1, 1, 1]),
                    #                                   [200, 244])
                    # warp_err = tf.reduce_mean((self.frames_y - frames_ref_warp) ** 2, axis=4)
                    case_path = dataPath.split('/')[-1]
                    print('Testing - ', case_path, len(imgs))
                    [imgs_hr_rgb] = sess.run([output_rgb],feed_dict={frames_lr: imgs})

                    if len(dims) == 3 and idx0==self.num_frames//2:
                        scipy.misc.imsave(os.path.join(path, 'rgb_%03d.png' % (file.index(pat))),
                                          im2uint8(imgs_hr_rgb[0, -1, :, :, :]))

                        # summary_str = sess.run(summary_op)
                        # summary_writer.add_summary(summary_str, idx0)
                print('SR results path: {}'.format(path))

            else:
                print('startid error')
def main(_):
    model = VIDEOSR()
    # model.train()
    # model.evaluation()
    model.test('F:\\jiangkui\\shujuji\\wxtrain\\test30')


if __name__ == '__main__':
    tf.app.run()