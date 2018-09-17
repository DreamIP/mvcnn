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
solver = caffe.SGDSolver('netpa_solver.prototxt')
solver.solve()
