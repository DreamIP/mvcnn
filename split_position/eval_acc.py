import sys
import os
import numpy as np
os.environ['GLOG_minloglevel'] = '2'
import caffe

sys.path.insert(0,'../src')
import MVCNNDataLayer

def EvalCaffeAcc(proto_file, weight_file, num_iter):
    net = caffe.Net(proto_file,weight_file,caffe.TEST)
    acc = 0
    for itter in range(num_iter):
        acc+=net.forward()['accuracy']
    acc = acc / num_iter
    print(weight_file + ": " + str(100*acc))

def main():
    num_iter = 2000
    ## ENABLE GPU MODE
    caffe.set_mode_gpu()
    caffe.set_device(0)
    nb_views = ['2','3','4','11','12']                 # number of views
    nb_views = '3'
    for state_iter  in range(100,2100,100):
        proto_file = 'netpa3_s1.prototxt'          # File name of proto
        weight_file = 'netpa3-split' + '_iter_' + str(state_iter) + '.caffemodel'
        print(weight_file)
        if(os.path.isfile(weight_file) and os.path.isfile(proto_file)) :
            EvalCaffeAcc(proto_file,weight_file,num_iter)
if __name__ == '__main__':

    main()
    os.environ['GLOG_minloglevel'] = '0'
