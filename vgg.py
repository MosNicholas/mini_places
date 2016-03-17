#!/usr/bin/env python

from train_places import *

weight_param    = dict(lr_mult=0.1, decay_mult=0.1)
bias_param      = dict(lr_mult=2, decay_mult=0)
learned_param   = [weight_param, bias_param]
no_bias         = [weight_param, dict(bias_term=False)]
frozen_param    = [dict(lr_mult=0)] * 2
batch_norm_param= [dict(lr_mult=0)] * 3

zero_filler          = dict(type='constant', value=0)
fc_filler            = dict(type='gaussian', std=0.005)
msra                 = dict(type='msra')
msra_no_local        = dict(type='msra', variance_norm=1)
xavier               = dict(type='xavier')
xavier_no_local      = dict(type='xavier', variance_norm=1)

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=weight_param,
              weight_filler=xavier, bias_filler=zero_filler,
              train=False):
    # set CAFFE engine to avoid CuDNN convolution -- non-deterministic results
    engine = {}
    if train and not args.cudnn:
        engine.update(engine=P.Pooling.CAFFE)
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group, param=param,
                         weight_filler=weight_filler, bias_term=False, **engine)
    
    norm = L.BatchNorm(conv, param=batch_norm_param)
    lrn = L.LRN(norm)
    return conv, norm, lrn, L.ReLU(lrn, in_place=True)

def fc_relu(bottom, nout, param=weight_param,
            weight_filler=xavier_no_local, bias_filler=zero_filler):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler, bias_term=False)
    norm = L.BatchNorm(fc, param=batch_norm_param)
    return fc, norm, L.ReLU(norm, in_place=True)

def max_pool(bottom, ks, stride=1, pad=0, train=False):
    # set CAFFE engine to avoid CuDNN pooling -- non-deterministic results
    engine = {}
    if train and not args.cudnn:
        engine.update(engine=P.Pooling.CAFFE)
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, pad=pad, stride=stride,
                     **engine)

def minivgg(data, labels=None, train=False, param=weight_param,
                num_classes=100, with_labels=True):
    n = caffe.NetSpec()
    n.data = data
    conv_kwargs = dict(param=param, train=train)
    
    n.Convolution1, n.BatchNorm1, n.LRN1, n.relu1 = conv_relu(n.data, 3, 32, stride=1, pad=1, **conv_kwargs)
    n.Convolution2, n.BatchNorm2, n.LRN2, n.relu2 = conv_relu(n.relu1, 3, 32, stride=1, pad=1, **conv_kwargs)
    n.pool1 = max_pool(n.relu2, 2, stride=2, pad=1, train=train)
    
    n.Convolution3, n.BatchNorm3, n.LRN3, n.relu3 = conv_relu(n.pool1, 3, 64, stride=1, pad=1, **conv_kwargs)
    n.Convolution4, n.BatchNorm4, n.LRN4, n.relu4 = conv_relu(n.relu3, 3, 64, stride=1, pad=1, **conv_kwargs)
    n.pool2 = max_pool(n.relu4, 2, stride=2, pad=1, train=train)
    
    n.Convolution5, n.BatchNorm5, n.LRN5, n.relu5 = conv_relu(n.pool2, 3, 128, stride=1, pad=1, **conv_kwargs)
    n.Convolution6, n.BatchNorm6, n.LRN6, n.relu6 = conv_relu(n.relu5, 3, 128, stride=1, pad=1, **conv_kwargs)
    n.Convolution7, n.BatchNorm7, n.LRN7, n.relu7 = conv_relu(n.relu6, 3, 128, stride=1, pad=1, **conv_kwargs)
    n.Convolution8, n.BatchNorm8, n.LRN8, n.relu8 = conv_relu(n.relu7, 3, 128, stride=1, pad=1, **conv_kwargs)
    n.pool3 = max_pool(n.relu8, 2, stride=2, pad=1, train=train)
    
    n.Convolution9, n.BatchNorm9, n.LRN9, n.relu9 = conv_relu(n.pool3, 3, 256, stride=1, pad=1, **conv_kwargs)
    n.Convolution10, n.BatchNorm10, n.LRN10, n.relu10 = conv_relu(n.relu9, 3, 256, stride=1, pad=1, **conv_kwargs)
    n.Convolution11, n.BatchNorm11, n.LRN11, n.relu11 = conv_relu(n.relu10, 3, 256, stride=1, pad=1, **conv_kwargs)
    n.ConvolutionAdded1, n.BatchNormAdded1, n.LRNAdded1, n.relu_added = conv_relu(n.relu11, 3, 256, stride=1, pad=1, **conv_kwargs)
    n.pool4 = max_pool(n.relu_added, 2, stride=2, train=train)

    n.InnerProduct1, n.BatchNorm12, n.relu12 = fc_relu(n.pool4, 1024, param=param)
    n.drop1 = L.Dropout(n.relu12, in_place=True, dropout_ratio=0.5)
    n.InnerProduct2, n.BatchNorm13, n.relu13 = fc_relu(n.drop1, 1024, param=param)
    n.drop2 = L.Dropout(n.relu13, in_place=True, dropout_ratio=0.5)

    preds = n.fc3 = L.InnerProduct(n.drop2, num_output=num_classes, param=param, bias_term=False)
    
    if not train:
        preds = n.probs = L.Softmax(n.fc3)

    if with_labels:
        n.label = labels
        n.loss = L.SoftmaxWithLoss(n.fc3, n.label)
        n.accuracy_at_1 = L.Accuracy(preds, n.label)
        n.accuracy_at_5 = L.Accuracy(preds, n.label, accuracy_param=dict(top_k=5))
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
