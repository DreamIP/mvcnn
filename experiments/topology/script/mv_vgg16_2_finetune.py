#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# CPU version for test
import sys
import os
import numpy as np
import caffe
sys.path.insert(0,'../../src')
import MVCNNDataLayer
import scipy.io as sio
import h5py
caffe.set_mode_cpu()
#caffe.set_device(0)
solver = caffe.SGDSolver('mv_vgg16_2_solver.prototxt')
solver.net.copy_from('../../caffe_model/vgg16.caffemodel')
solver.solve()
