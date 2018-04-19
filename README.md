# Multi View CNN Reboot
Experimenting with Multi-views CNNs with Caffe, work under process.

Ideally, we want to demonstrate that the performance of CNNs for image classification can be improved by providing multiple perspectives --of the same frame-- to the network.
Otherwise, one can reduce the depth (and thus the computations) of a CNN and keep tolerable classification accuracy by providing multi-perspective inputs.


## Resources
- Original paper from [Hang et al.](https://arxiv.org/abs/1505.00880)
- Original git [repository](https://github.com/suhangpro/mvcnn/tree/master/caffe)
- Dataset :  [ModelNet40v1](https://drive.uca.fr/d/80ea3fccdd8942c6a062/)
- Network : Alexnet topology for now, will be experimenting lighter CNNs after ...


## Accuracy
| Network            | Accuracy |
|:-------------------|:---------|
| alexnet            | 1.5%     | Single view, vanilla CNN, authors report 83 %  
| alexnet-ft         | 85.39%   | Single view w/ fine-tuning on ModelNet40v1
| mvcnn12            | 88.4%    | 12 views wo/ fine-tuning
| mvcnn12-ft         | 90.8%    | 12 views w/ fine-tuning on ModelNet40v1

## Todo
1. Evaluate accuracy of vanilla alexnet on modelnetv1, 1.5% is clearly bad
2. Study with 3:12 number of views
3. Explore/define where to put the view-pooling layer

## Weaknesses
1. Shitty ModelNet40v1 Dataset: CAD Images, not sure it with work in real world Images
2. Multi-view images of large objects in demos
