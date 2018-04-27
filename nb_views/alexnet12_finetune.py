#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import os
import numpy as np
os.environ['GLOG_minloglevel'] = '2'
import caffe
sys.path.insert(0,'../src')
caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('alexnet12_solver.prototxt')
solver.net.copy_from('../caffemodel/alexnet1.caffemodel')
solver.solve()
