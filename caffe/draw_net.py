#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format

import caffe
import caffe.draw
from caffe.proto import caffe_pb2

input_net_proto_file = 'mvcnn12.prototxt'
net = caffe_pb2.NetParameter()
text_format.Merge(open(input_net_proto_file).read(), net)
rankdir ='LR'

caffe.draw.draw_net_to_file(net,'./mvcnn12.png',rankdir,caffe.TEST)
