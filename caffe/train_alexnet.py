#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import sys
caffe_root = './caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import scipy.io as sio
import h5py
import os
caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('alexnet_solver.prototxt')
solver.net.copy_from('alexnet.caffemodel')
solver.solve()
