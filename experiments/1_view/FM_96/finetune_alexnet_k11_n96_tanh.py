#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import caffe
import os
import sys
sys.path.insert(0,'../../src')
caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('prototxt/alexnet_k11_n96_tanh_solver.prototxt')
solver.net.copy_from('temp_caffemodel/alexnet_k11_n96_tanh_iter_300000.caffemodel')
solver.solve()
