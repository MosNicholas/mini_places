#!/usr/bin/env python

from train_places import *

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2

zero_filler     = dict(type='constant', value=0)
msra_filler     = dict(type='msra')
uniform_filler  = dict(type='uniform', min=-0.1, max=0.1)
fc_filler       = dict(type='gaussian', std=0.005)
# Original AlexNet used the following commented out Gaussian initialization;
# we'll use the "MSRA" one instead, which scales the Gaussian initialization
# of a convolutional filter based on its receptive field size.
# conv_filler     = dict(type='gaussian', std=0.01)
conv_filler     = dict(type='msra')

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=conv_filler, bias_filler=zero_filler,
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

def max_pool(bottom, ks, stride=1, train=False):
    # set CAFFE engine to avoid CuDNN pooling -- non-deterministic results
    engine = {}
    if train and not args.cudnn:
        engine.update(engine=P.Pooling.CAFFE)
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride,
                     **engine)

def minialexnet(data, labels=None, train=False, param=learned_param,
                num_classes=100, with_labels=True):
    """
    Returns a protobuf text file specifying a variant of AlexNet, following the
    original specification (<caffe>/models/bvlc_alexnet/train_val.prototxt).
    The changes with respect to the original AlexNet are:
        - LRN (local response normalization) layers are not included
        - The Fully Connected (FC) layers (fc6 and fc7) have smaller dimensions
          due to the lower resolution of mini-places images (128x128) compared
          with ImageNet images (usually resized to 256x256)
    """
    n = caffe.NetSpec()
    n.data = data
    conv_kwargs = dict(param=param, train=train)
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, **conv_kwargs)
    n.pool1 = max_pool(n.relu1, 3, stride=2, train=train)
    n.conv2, n.relu2 = conv_relu(n.pool1, 5, 256, pad=2, group=2, **conv_kwargs)
    n.pool2 = max_pool(n.relu2, 3, stride=2, train=train)
    n.conv3, n.relu3 = conv_relu(n.pool2, 3, 384, pad=1, **conv_kwargs)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, **conv_kwargs)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, **conv_kwargs)
    n.pool5 = max_pool(n.relu5, 3, stride=2, train=train)
    n.fc6, n.relu6 = fc_relu(n.pool5, 1024, param=param)
    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6, 1024, param=param)
    n.drop7 = L.Dropout(n.relu7, in_place=True)
    preds = n.fc8 = L.InnerProduct(n.drop7, num_output=num_classes, param=param)
    if not train:
        # Compute the per-label probabilities at test/inference time.
        preds = n.probs = L.Softmax(n.fc8)
    if with_labels:
        n.label = labels
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
        n.accuracy_at_1 = L.Accuracy(preds, n.label)
        n.accuracy_at_5 = L.Accuracy(preds, n.label,
                                     accuracy_param=dict(top_k=5))
    else:
        n.ignored_label = labels
        n.silence_label = L.Silence(n.ignored_label, ntop=0)
    return to_tempfile(str(n.to_proto()))

if __name__ == '__main__':
    print 'Training net...\n'
    train_net(minialexnet)

    print '\nTraining complete. Evaluating...\n'
    for split in ('train', 'val', 'test'):
        eval_net(minialexnet, split)
        print
    print 'Evaluation complete.'