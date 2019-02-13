import numpy as np
import matplotlib.pyplot as plt
import caffe
import sys
import os

model_def = './vgg16_2.prototxt'
model_weights = './vgg16_2_tanh.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)
mu = np.load('ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

image1 = caffe.io.load_image('../../dataset/modelnet40v1/table/test/desk_000000010_002.jpg')
image2 = caffe.io.load_image('../../dataset/modelnet40v1/table/test/desk_000000010_003.jpg')
#print(image*255)
# f = plt.figure()
# f.add_subplot(1,2, 1)
# plt.imshow(image1)
# f.add_subplot(1,2, 2)
# plt.imshow(image2)
# plt.show(block=True)

transformed_image1 = transformer.preprocess('data', image1)
transformed_image2 = transformer.preprocess('data', image2)
net.blobs['data'].data[0,...] = transformed_image1
net.blobs['data'].data[1,...] = transformed_image2
output = net.forward()
output_prob = output['prob'][0]
print('predicted class is:', output_prob.argmax())
print('confidance is:', output_prob.max())
