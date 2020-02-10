![Screenshot](logo.png)


_Deep Point Cloud Benchmark_ is a framework for running common deep learning models for point cloud analysis tasks against classic benchmark datasets. It heavily relies on [```Pytorch Geometric```](https://github.com/rusty1s/pytorch_geometric) and [```Facebook Hydra library```](https://hydra.cc/docs/intro) thanks for the great work!

Here is the link to the [```Github```](https://github.com/nicolas-chaulet/deeppointcloud-benchmarks) project.

We aim to build a tool which can be used for benchmarking SOTA models, while also allowing users to efficiently pursue research into point cloud analysis,  with the end-goal of building models which can be applied to real-life applications.

<h2>Core features</h2>

* ```Task``` driven implementation with dynamic model and dataset resolution from arguments.
* ```Core``` implementation of common components for point cloud deep learning - greatly simplying the creation of new models:
    * ```Core Architectures``` - Unet
    * ```Core Modules``` - Residual Block, Down-sampling and Up-sampling convolutions
    * ```Core Transforms``` - Grid Sampling, Rotation, Scaling
    * ```Core Sampling``` - FPS, Random Sampling
    * ```Core Neighbour Finder``` - Radius Search, KNN
* 4 ```Base Convolution``` base classes to simplify the implementation of new convolutions. Each base class supports a different data format (B = number of batches, C = number of features):
    * ```DENSE``` (B, num_points, C)
    * ```PARTIAL DENSE``` (B * num_points, C)
    * ```MESSAGE PASSING``` (B * num_points, C)
    * ```SPARSE``` (B * num_points, C)

* Models can be completely specified using a YAML file, greatly easing reproducability. 
* Several visualiation tools ```(tensorboard, wandb)``` and _dynamic metric-based model checkpointing_, which is easily customizable. 
* _Dynamic customized placeholder resolution_ for smart model definition.

<h2>Current supported models</h2>

* [```RandLA-Net```: Efficient Semantic Segmentation of Large-Scale Point Clouds ](https://arxiv.org/pdf/1911.11236.pdf)
* [```Relation-Shape Convolutional (RSConv)``` Neural Network for Point Cloud Analysis](https://arxiv.org/abs/1904.07601)
* [```KPConv```: Flexible and Deformable Convolution for Point Clouds](https://arxiv.org/abs/1904.08889)
* [```PointCNN```: Convolution On X-Transformed Points](https://arxiv.org/abs/1801.07791)
* [```PointNet++```: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
* [```Submanifold sparse convolutional networks```](https://arxiv.org/pdf/1711.10275.pdf)

and much more to come ...


<h2>Current supported tasks</h2>

* Segmentation
* Classification

## Ressources

* [Pytorch Geometric Slides](http://rusty1s.github.io/pyg_slides.pdf)


![Screenshot](https://uploads-ssl.webflow.com/5a9058c8f7462d00014ad4ff/5a988ceadc6c9b0001cb2511_point%20cloud.JPG)


