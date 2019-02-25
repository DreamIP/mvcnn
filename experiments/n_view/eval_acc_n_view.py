import sys
import os
import numpy as np
os.environ['GLOG_minloglevel'] = '2'
import caffe
sys.path.insert(0,'../../src')
import MVCNNDataLayer

def EvalCaffeAcc(proto_file, weight_file, num_iter):
    net = caffe.Net(proto_file,weight_file,caffe.TEST)
    acc = 0
    for itter in range(num_iter):
        acc+=net.forward()['accuracy']
    acc = acc / num_iter
    print(weight_file + ": " + str(100*acc))

def main(argv):
    num_iter = 2000
    ## ENABLE GPU MODE
    caffe.set_mode_gpu()
    caffe.set_device(0)
    for arg in argv:
        proto_file       = arg[0]
        weight_file      = arg[1]      
    EvalCaffeAcc(proto_file,weight_file,num_iter)

if __name__ == '__main__':

    main(sys.argv)
    os.environ['GLOG_minloglevel'] = '0'
