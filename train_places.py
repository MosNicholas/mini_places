#!/usr/bin/env python

from __future__ import division

import argparse
import numpy as np
import os
import tempfile
import time
import matplotlib.pyplot as plt

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
parser.add_argument('--stepsize', type=int, default=8*1000,
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

def get_split(split):
    filename = './development_kit/data/%s.txt' % split
    if not os.path.exists(filename):
        raise IOError('Split data file not found: %s' % split)
    return filename

def miniplaces_net(source, fn, train=False, with_labels=True):
    mean = [104, 117, 123]  # per-channel mean of the BGR image pixels
    transform_param = dict(mirror=train, crop_size=args.crop, mean_value=mean)
    batch_size = args.batch if train else 100
    places_data, places_labels = L.ImageData(transform_param=transform_param,
        source=source, root_folder=args.image_root, shuffle=train,
        batch_size=batch_size, ntop=2)
    return fn(data=places_data, labels=places_labels, train=train,
                       with_labels=with_labels)

def snapshot_prefix():
    return os.path.join(args.snapshot_dir, args.snapshot_prefix)

def snapshot_at_iteration(iteration):
    return '%s_iter_%d.caffemodel' % (snapshot_prefix(), iteration)

def miniplaces_solver(train_net_path, test_net_path=None):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        # Test after every 1000 training iterations.
        s.test_interval = 1000
        # Set `test_iter` to test on 100 batches each time we test.
        # With test batch size 100, this covers the entire validation set of
        # 10K images (100 * 100 = 10K).
        s.test_iter.append(100)
    else:
        s.test_interval = args.iters + 1  # don't test during training

    # The number of batches over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = args.iter_size

    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # The following settings (base_lr, lr_policy, gamma, stepsize, and max_iter),
    # define the following learning rate schedule:
    #   Iterations [  0, 20K) -> learning rate 0.01   = base_lr
    #   Iterations [20K, 40K) -> learning rate 0.001  = base_lr * gamma
    #   Iterations [40K, 50K) -> learning rate 0.0001 = base_lr * gamma^2

    # Set the initial learning rate for SGD.
    s.base_lr = args.lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = args.gamma
    s.stepsize = args.stepsize

    # `max_iter` is the number of times to update the net (training iterations).
    s.max_iter = args.iters

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help
    # prevent the model from overfitting.
    s.momentum = args.momentum
    s.weight_decay = args.decay

    # Display the current training loss and accuracy every `display` iterations.
    # This doesn't have an effect for Python training here as logging is
    # disabled by this script (see the GLOG_minloglevel setting).
    s.display = args.disp

    # Number of training iterations over which to smooth the displayed loss.
    # The summed loss value (Iteration N, loss = X) will be averaged,
    # but individual loss values (Train net output #K: my_loss = X) won't be.
    s.average_loss = 10

    # Seed the RNG for deterministic results.
    # (May not be so deterministic if using CuDNN.)
    s.random_seed = args.seed

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot twice per learning rate step to the location specified by the
    # --snapshot_dir and --snapshot_prefix args.
    s.snapshot = args.stepsize // 2
    s.snapshot_prefix = snapshot_prefix()

    # Create snapshot dir if it doesn't already exist.
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    return to_tempfile(str(s))

def train_net(fn, with_val_net=False):
    train_net_file = miniplaces_net(get_split('train'), fn, train=True)
    # Set with_val_net=True to test during training.
    # Environment variable GLOG_minloglevel should be set to 0 to display
    # Caffe output in this case; otherwise, the test result will not be
    # displayed.
    if with_val_net:
        val_net_file = miniplaces_net(get_split('val'), fn, train=False)
    else:
        val_net_file = None
    solver_file = miniplaces_solver(train_net_file, val_net_file)
    solver = caffe.get_solver(solver_file)
    solver.restore('snapshot/place_net_iter_24000.solverstate')
    filters = solver.net.params['Convolution1'][0].data
    vis_square(filters.transpose(0, 2, 3, 1))
    crash
    # solver.net.copy_from('snapshot/place_net_iter_45000.caffemodel')
    outputs = sorted(solver.net.outputs)
    def str_output(output):
        value = solver.net.blobs[output].data
        if output.startswith('accuracy'):
            valstr = '%5.2f%%' % (100 * value, )
        else:
            valstr = '%6f' % value
        return '%s = %s' % (output, valstr)
    def disp_outputs(iteration, iter_pad_len=len(str(args.iters))):
        metrics = '; '.join(str_output(o) for o in outputs)
        return 'Iteration %*d: %s' % (iter_pad_len, iteration, metrics)
    # We could just call `solver.solve()` rather than `step()`ing in a loop.
    # (If we hadn't set GLOG_minloglevel = 3 at the top of this file, Caffe
    # would display loss/accuracy information during training.)
    previous_time = None
    for iteration in xrange(args.iters):
        solver.step(1)
        if (args.disp > 0) and (iteration % args.disp == 0):
            current_time = time.clock()
            if previous_time is None:
                benchmark = ''
            else:
                time_per_iter = (current_time - previous_time) / args.disp
                benchmark = ' (%5f s/it)' % time_per_iter
            previous_time = current_time
            print disp_outputs(iteration), benchmark
    # Print accuracy for last iteration.
    solver.net.forward()
    disp_outputs(args.iters)
    solver.net.save(snapshot_at_iteration(args.iters))

def eval_net(fn, split, K=5):
    print 'Running evaluation for split:', split
    filenames = []
    labels = []
    split_file = get_split(split)
    with open(split_file, 'r') as f:
        for line in f.readlines():
            parts = line.split()
            assert 1 <= len(parts) <= 2, 'malformed line'
            filenames.append(parts[0])
            if len(parts) > 1:
                labels.append(int(parts[1]))
    known_labels = (len(labels) > 0)
    if known_labels:
        assert len(labels) == len(filenames)
    else:
        # create file with 'dummy' labels (all 0s)
        split_file = to_tempfile(''.join('%s 0\n' % name for name in filenames))
    test_net_file = miniplaces_net(split_file, fn, train=False, with_labels=False)
    weights_file = snapshot_at_iteration(args.iters)
    net = caffe.Net(test_net_file, weights_file, caffe.TEST)
    top_k_predictions = np.zeros((len(filenames), K), dtype=np.int32)
    if known_labels:
        correct_label_probs = np.zeros(len(filenames))
    offset = 0
    while offset < len(filenames):
        probs = net.forward()['probs']
        for prob in probs:
            top_k_predictions[offset] = (-prob).argsort()[:K]
            if known_labels:
                correct_label_probs[offset] = prob[labels[offset]]
            offset += 1
            if offset >= len(filenames):
                break
    if known_labels:
        def accuracy_at_k(preds, labels, k):
            assert len(preds) == len(labels)
            num_correct = sum(l in p[:k] for p, l in zip(preds, labels))
            return num_correct / len(preds)
        for k in [1, K]:
            accuracy = 100 * accuracy_at_k(top_k_predictions, labels, k)
            print '\tAccuracy at %d = %4.2f%%' % (k, accuracy)
        cross_ent_error = -np.log(correct_label_probs).mean()
        print '\tSoftmax cross-entropy error = %.4f' % (cross_ent_error, )
    else:
        print 'Not computing accuracy; ground truth unknown for split:', split
    filename = 'top_%d_predictions.%s.csv' % (K, split)
    with open(filename, 'w') as f:
        f.write(','.join(['image'] + ['label%d' % i for i in range(1, K+1)]))
        f.write('\n')
        f.write(''.join('%s,%s\n' % (image, ','.join(str(p) for p in preds))
                        for image, preds in zip(filenames, top_k_predictions)))
    print 'Predictions for split %s dumped to: %s' % (split, filename)


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print 'Visualizing first layer of filters'
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.imshow(data)
    plt.axis('off')
    plt.show()