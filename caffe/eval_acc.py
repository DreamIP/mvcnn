import os
import numpy as np
os.environ['GLOG_minloglevel'] = '2'
import caffe

def EvalCaffeAcc(proto_file, weight_file, num_itter):
    net = caffe.Net(proto_file,weight_file,caffe.TEST)
    acc = 0
    for itter in range(num_itter):
        acc+=net.forward()['accuracy']
    acc = acc / num_itter
    print(" >> Accuracy of " + weight_file + ": " + str(100*acc))


def EvalAlexnetAcc(proto_file='alexnet_eval.prototxt',weight_file='alexnet.caffemodel',num_itter=50, dataset_path=0):
    net = caffe.Net(proto_file,weight_file,caffe.TEST)
    acc = 0
    for itter in range(num_itter):
        acc+=net.forward()['accuracy']
    acc = acc / num_itter
    print(" >> Accuracy of " + weight_file + ": " + str(100*acc))
 

def main():
    num_itter = 500
    caffe.set_mode_gpu()
    caffe.set_device(0)

    # alexnet_ft
    proto_file = 'alexnet_eval.prototxt'
    weight_files = ['alexnet_ft_100k.caffemodel',
                    'alexnet_ft_200k.caffemodel',
                    'alexnet_ft_300k.caffemodel',
                    'alexnet_ft_400k.caffemodel']

    for weight_file in weight_files:
        EvalCaffeAcc(proto_file,weight_file,num_itter)



    # mvcnn12
    proto_file = 'mvcnn12_eval.prototxt'
    weight_file = 'alexnet_ft.caffemodel'
    EvalCaffeAcc(proto_file,weight_file,num_itter)


    # mvcnn12_ft
    proto_file = 'mvcnn12_eval.prototxt'
    weight_files = ['mvcnn12_ft_100.caffemodel',
                    'mvcnn12_ft_200.caffemodel',
                    'mvcnn12_ft_300.caffemodel',
                    'mvcnn12_ft_400.caffemodel',
                    'mvcnn12_ft_500.caffemodel',
                    'mvcnn12_ft_600.caffemodel',
                    'mvcnn12_ft_700.caffemodel',
                    'mvcnn12_ft_800.caffemodel',
                    'mvcnn12_ft_900.caffemodel',
                    'mvcnn12_ft_1000.caffemodel',
                    'mvcnn12_ft_1100.caffemodel',
                    'mvcnn12_ft_1200.caffemodel',
                    'mvcnn12_ft_1300.caffemodel',
                    'mvcnn12_ft_1400.caffemodel',
                    'mvcnn12_ft_1500.caffemodel',
                    'mvcnn12_ft_1600.caffemodel',
                    'mvcnn12_ft_1700.caffemodel',
                    'mvcnn12_ft_1800.caffemodel',
                    'mvcnn12_ft_1900.caffemodel',
                    'mvcnn12_ft_2000.caffemodel']
    for weight_file in weight_files:
        EvalCaffeAcc(proto_file,weight_file,num_itter)

if __name__ == '__main__':
    # EvalAlexnetAcc()
    main()
    os.environ['GLOG_minloglevel'] = '0'

