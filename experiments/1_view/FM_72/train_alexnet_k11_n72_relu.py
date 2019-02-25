#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import caffe
import os
import sys
sys.path.insert(0,'../../src')
caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('prototxt/alexnet_k11_n72_relu_solver.prototxt')
solver.solve()
