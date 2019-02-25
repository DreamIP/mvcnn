#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import sys
sys.path.insert(0,'../../../src')
import MVCNNDataLayer
import caffe
import scipy.io as sio
import h5py
import os
caffe.set_mode_gpu()
caffe.set_device(0)
#change to the *solver.prototxt you wish to train
solver = caffe.SGDSolver('prototxt/alexnet_k7_n48_tanh_solver.prototxt')
solver.net.copy_from('caffemodel_relu/alexnet_k5_n48_relu_iter_300000.caffemodel')
solver.solve()
