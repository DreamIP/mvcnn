#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import sys
import caffe
import scipy.io as sio
import h5py
import os
caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('mvnetpa12_solver.prototxt')
solver.net.copy_from('netpa.caffemodel')
solver.solve()
