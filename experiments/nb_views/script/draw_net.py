#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format

import caffe
import caffe.draw
from caffe.proto import caffe_pb2

network = 'alexnet12'
input_net_proto_file = network + '.prototxt'
out_png = network + '.png'
net = caffe_pb2.NetParameter()
text_format.Merge(open(input_net_proto_file).read(), net)
rankdir ='LR'
caffe.draw.draw_net_to_file(net,out_png,rankdir,caffe.TEST)
