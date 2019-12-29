![Screenshot](logo.png)


This is a framework for running common deep learning models for point cloud analysis tasks against classic benchmark. It heavily relies on [```Pytorch Geometric```](https://github.com/rusty1s/pytorch_geometric) and [```Facebook Hydra library```](https://hydra.cc/docs/intro).

We aim at building a tool for both benchmarking SOTA models and to efficiently pursue research for point cloud analysis.

<h2>Core features</h2>

* ```Task driven``` implementation with dynamic model and dataset resolution from arguments
* ```Unet base``` implementation for simplying new model creation [more unet base are coming]
* ```Base Convolution``` to simplify new convolution implementation
* ```Base sampler / neigbours finder``` collections
* ```2 API``` to write models ```(compact / sequential)``` to ease reproducibility
* Several visualiation tool ```(tensorboard, wandb)``` and ```dynamic metric-based model checkpointing``` for one to customize
* ```Dynamic customized placeholder resolution``` for smart model definition

<h2>Current supported tasks</h2>

* Segmentation
* Classification


![Screenshot](https://uploads-ssl.webflow.com/5a9058c8f7462d00014ad4ff/5a988ceadc6c9b0001cb2511_point%20cloud.JPG)


