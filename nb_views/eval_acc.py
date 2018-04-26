import os
import numpy as np
os.environ['GLOG_minloglevel'] = '2'
import caffe

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
    # caffe.set_mode_gpu()
    # caffe.set_device(0)
    nb_views = ['2','3','4','10','11','12']                 # number of views
    caffe_model_root = '/media/kamel/Data/caffemodels/' # Path to caffemodels
    for nv in nb_views:
        proto_file = 'mvcnn' + nv + '.prototxt'          # File name of proto
        for state_itter in range(100,2100,100):      # loop accross snapshots
            weight_file = caffe_model_root
            weight_file += 'mvcnn' + nv + '_iter_' + str(state_itter) + '.caffemodel'
            if(os.path.isfile(weight_file) and os.path.isfile(proto_file)) :
                EvalCaffeAcc(proto_file,weight_file,num_iter)

if __name__ == '__main__':

    main()
    os.environ['GLOG_minloglevel'] = '0'
