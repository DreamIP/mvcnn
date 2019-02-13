#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import os
import numpy as np
import caffe
sys.path.insert(0,'../../src')
import MVCNNDataLayer
import scipy.io as sio
import h5py
import os
caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('resnet_solver.prototxt')
solver.net.copy_from('../caffemodel/resnet.caffemodel')
solver.solve()
