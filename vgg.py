#!/usr/bin/env python

from __future__ import division

import argparse
import numpy as np
import os
import tempfile
import time

parser = argparse.ArgumentParser(
    description='Train and evaluate a net on the MIT mini-places dataset.')
parser.add_argument('--image_root', default='./images/',
    help='Directory where images are stored')
parser.add_argument('--crop', type=int, default=96,
    help=('The edge length of the random image crops'
          '(defaults to 96 for 96x96 crops)'))
parser.add_argument('--disp', type=int, default=10,
    help='Print loss/accuracy every --disp training iterations')
parser.add_argument('--snapshot_dir', default='./snapshot',
    help='Path to directory where snapshots are saved')
parser.add_argument('--snapshot_prefix', default='place_net',
    help='Snapshot filename prefix')
parser.add_argument('--iters', type=int, default=50*1000,
    help='Total number of iterations to train the network')
parser.add_argument('--batch', type=int, default=256,
    help='The batch size to use for training')
parser.add_argument('--iter_size', type=int, default=1,
    help=('The number of iterations (batches) over which to average the '
          'gradient computation. Effectively increases the batch size '
          '(--batch) by this factor, but without increasing memory use '))
parser.add_argument('--lr', type=float, default=0.01,
    help='The initial learning rate')
parser.add_argument('--gamma', type=float, default=0.1,
    help='Factor by which to drop the learning rate')
parser.add_argument('--stepsize', type=int, default=10*1000,
    help='Drop the learning rate every N iters -- this specifies N')
parser.add_argument('--momentum', type=float, default=0.9,
    help='The momentum hyperparameter to use for momentum SGD')
parser.add_argument('--decay', type=float, default=5e-4,
    help='The L2 weight decay coefficient')
parser.add_argument('--seed', type=int, default=1,
    help='Seed for the random number generator')
parser.add_argument('--cudnn', action='store_true',
    help='Use CuDNN at training time -- usually faster, but non-deterministic')
parser.add_argument('--gpu', type=int, default=0,
    help='GPU ID to use for training and inference (-1 for CPU)')
args = parser.parse_args()

# disable most Caffe logging (unless env var $GLOG_minloglevel is already set)
key = 'GLOG_minloglevel'
if not os.environ.get(key, ''):
    os.environ[key] = '3'

import caffe
from caffe.proto import caffe_pb2
from caffe import layers as L
from caffe import params as P

if args.gpu >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
else:
    caffe.set_mode_cpu()

def to_tempfile(file_content):
    """Serialize a Python protobuf object str(proto), dump to a temporary file,
       and return its filename."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(file_content)
        return f.name

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2

zero_filler     = dict(type='constant', value=0)
fc_filler       = dict(type='gaussian', std=0.005)
xavier          = dict(type='xavier')
xavier_no_local     = dict(type='xavier', local='false')

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=xavier, bias_filler=zero_filler,
              train=False):
    # set CAFFE engine to avoid CuDNN convolution -- non-deterministic results
    engine = {}
    if train and not args.cudnn:
        engine.update(engine=P.Pooling.CAFFE)
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group, param=param,
                         weight_filler=weight_filler, bias_filler=bias_filler,
                         **engine)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=fc_filler, bias_filler=zero_filler):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler, bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def fc_softmax(bottom, nout, param=learned_param,
            weight_filler=fc_filler, bias_filler=zero_filler):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler, bias_filler=bias_filler)
    return fc, L.Softmax(fc, in_place=True)

def max_pool(bottom, ks, stride=1, pad=0, train=False):
    # set CAFFE engine to avoid CuDNN pooling -- non-deterministic results
    engine = {}
    if train and not args.cudnn:
        engine.update(engine=P.Pooling.CAFFE)
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, pad=pad, stride=stride,
                     **engine)

def minivgg(data, labels=None, train=False, param=learned_param,
                num_classes=100, with_labels=True):
    n = caffe.NetSpec()
    n.data = data
    conv_kwargs = dict(param=param, train=train)
    
    n.conv1, n.relu1 = conv_relu(n.data, 3, 32, stride=1, pad=1, **conv_kwargs)
    n.conv2, n.relu2 = conv_relu(n.relu1, 3, 32, stride=1, pad=1, **conv_kwargs)
    n.pool1 = max_pool(n.relu2, 2, stride=2, pad=1, train=train)
    
    n.conv3, n.relu3 = conv_relu(n.pool1, 3, 64, stride=1, pad=1, **conv_kwargs)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 64, stride=1, pad=1, **conv_kwargs)
    n.pool2 = max_pool(n.relu4, 2, stride=2, pad=1, train=train)
    
    n.conv5, n.relu5 = conv_relu(n.pool2, 3, 128, stride=1, pad=1, **conv_kwargs)
    n.conv6, n.relu6 = conv_relu(n.relu5, 3, 128, stride=1, pad=1, **conv_kwargs)
    n.conv7, n.relu7 = conv_relu(n.relu6, 3, 128, stride=1, pad=1, **conv_kwargs)
    n.conv8, n.relu8 = conv_relu(n.relu7, 3, 128, stride=1, pad=1, **conv_kwargs)
    n.pool3 = max_pool(n.relu8, 2, stride=2, pad=1, train=train)
    
    n.conv9, n.relu9 = conv_relu(n.pool3, 3, 256, stride=1, pad=1, **conv_kwargs)
    n.conv10, n.relu10 = conv_relu(n.relu9, 3, 256, stride=1, pad=1, **conv_kwargs)
    n.conv11, n.relu11 = conv_relu(n.relu10, 3, 256, stride=1, pad=1, **conv_kwargs)
    n.pool4 = max_pool(n.relu11, 2, stride=2, train=train)

    n.fc1, n.relu12 = fc_relu(n.pool4, 1024, param=param)
    n.drop1 = L.Dropout(n.fc1, in_place=True, dropout_ratio=0.5)
    n.fc2, n.relu13 = fc_relu(n.drop1, 1024, param=param)
    n.drop2 = L.Dropout(n.fc2, in_place=True, dropout_ratio=0.5)
    n.fc3, n.softMax1 = fc_softmax(n.drop2, num_classes, param=param)

    preds  = n.softMax1

    if with_labels:
        n.label = labels
        n.loss = L.SoftmaxWithLoss(n.fc3, n.label)
        n.accuracy_at_1 = L.Accuracy(preds, n.label)
        n.accuracy_at_5 = L.Accuracy(preds, n.label,
                                     accuracy_param=dict(top_k=5))
    else:
        n.ignored_label = labels
        n.silence_label = L.Silence(n.ignored_label, ntop=0)
    return to_tempfile(str(n.to_proto()))

if __name__ == '__main__':
    print 'Training net...\n'
    train_net(minivgg)

    print '\nTraining complete. Evaluating...\n'
    for split in ('train', 'val', 'test'):
        eval_net(minivgg, split)
        print
    print 'Evaluation complete.'
