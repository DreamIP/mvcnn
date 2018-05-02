import sys
import numpy as np
import sys
import caffe
import scipy.io as sio
import h5py
import os

sys.path.insert(0,'../src')
import MVCNNDataLayer

caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('netpa3_s1_solver.prototxt')
solver.net.copy_from('../caffemodel/netpa3.caffemodel')
solver.solve()
