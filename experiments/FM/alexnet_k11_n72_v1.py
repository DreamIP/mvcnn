#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import caffe
import os
caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('prototxt/alexnet_k11_n72_v1.prototxt')
solver.solve()
