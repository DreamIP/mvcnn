import numpy as np
import sys
import caffe
import scipy.io as sio
import h5py
import os
caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('mvcnn11_solver.prototxt')
solver.net.copy_from('alexnet_ft.caffemodel')
solver.solve()
