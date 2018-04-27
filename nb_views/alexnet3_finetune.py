#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import os
import numpy as np
import caffe
sys.path.insert(0,'../src')
caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('alexnet3.prototxt')
solver.net.copy_from('../caffemodel/alexnet1.caffemodel')
solver.solve()
