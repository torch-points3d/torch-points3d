
# Understand hydra configuration

We recommend people willing to use the framework to get familiries with [```Facebook Hydra library```](https://hydra.cc/docs/intro).

Reading quickly through Hydra documentation should give one the basic understanding of its core functionalites.

To make it short, it is [```argparse```](https://docs.python.org/2/library/argparse.html) built on top of yaml file, allowing ```arguments to be defined in a tree structure```.


<h2>Configuration architecture</h2>

* config.yaml
    * models/
        * segmentation.yaml (architecture of all seg models)


<h4>Experiment</h4>

The experiment conf defines are going to be done.


```yaml
experiment:
    experiment_name: "" # Wether to use hydra naming convention for the experiment
    log_dir: "" # Wether to use hydra automatic path generation for saving training
    resume: True  # Wether to load model
    name: RSConv_2LD # Name of the model to be created
    task: segmentation # Task to be benchmarked
    dataset: shapenet # Dataset to be loaded
```

<h6> The associated logging </h6>

As experiment is empty, it will use hydra naming convention for the experiment
As log_dir is empty, it will use hydra naming convention for the log directory
{path_to_project}/outputs/2019-12-28/12-05-45 (Y-M-D/H-M-S)

The ```name``` is let to the user choose.

<h6>The associated dataset</h6>

```experiment.dataset``` value is used as a key to dynamically choose the associated dataset arguments


<h6> The associated visualization </h6>

The framework currently support both [```wandb```](https://www.wandb.com/) and [```tensorboard```](https://www.tensorflow.org/tensorboard)

```yaml
# parameters for Weights and Biases
wandb:
    project: benchmarking
    log: False

# parameters for TensorBoard Visualization
tensorboard:
    log: True
```

<h4> Training arguments </h4>

```yaml
training:
    shuffle: True
    num_workers: 8
    batch_size: 8
    cuda: 1
    epochs: 350
    optimizer: Adam
    lr : 0.001
    weight_name: 'default'
    precompute_multi_scale: False
```

</br>

* ```weight_name```: Used when ```resume is True```, ```select``` with model to load from ```[metric_name..., default]```

* ```precompute_multi_scale```: Compute multiscate features on cpu for faster

<h4> Architecture of the loaded model </h4>

```experiment.name``` value is used as a key to dynamically choose the associated model architecture

Here, ```RSConv_2LD``` is a child model from the ```type: RSConv```.

```python
RSConv_2LD:
    type: RSConv
    down_conv:
        module_name: RSConv
        ratios: [0.2, 0.25]
        radius: [0.1, 0.2]
        local_nn: [[10, 8, FEAT], [10, 32, 64, 64]]
        down_conv_nn: [[FEAT, 16, 32, 64], [64, 64, 128]]
    innermost:
        module_name: GlobalBaseModule
        aggr: max
        nn: [128 + FEAT, 128] 
    up_conv:
        module_name: FPModule
        ratios: [1, 0.25, 0.2]
        radius: [0.2, 0.2, 0.1]
        up_conv_nn: [[128 + 128, 64], [64 + 64, 64], [64, 64]]
        up_k: [1, 3, 3]
        skip: True
    mlp_cls:
        nn: [64, 64, 64, 64]
        dropout: 0.5
```

Model definition will be discussed more in details later on.


# Create a new dataset

Create ShapeNet dataset from ```Pytorch Geometric```

Naming matters as we use them to automatically create the dataset with its given arguments.

One need to create a new file in datasets/ with the following name {dataset_name}_dataset.py.
We create ```shapenet_dataset.py``` file.

One need to create a dataset class with the following name {dataset_name}dataset.py ().
We create ```ShapeNetDataset.py``` class which inherit from ```BaseDataset``` class.
The dataset is going to receive automatically the associated dataset_opt, training_opt.

```python
class ShapeNetDataset(BaseDataset):
    
    def __init__(self, dataset_opt, training_opt):
        super().__init__(dataset_opt, training_opt)

        ...
        
```

In order to implement a working dataset, one should define a train_dataset, test_dataset and (optionally: val_dataset)
and call the ```create_dataloaders``` function with the created {}_dataset from the parent ```BaseDataset``` class


```python
class ShapeNetDataset(BaseDataset):
    
    def __init__(self, dataset_opt, training_opt):
        super().__init__(dataset_opt, training_opt)
        
        self._data_path = os.path.join(dataset_opt.dataroot, 'ShapeNet')
        self._category = dataset_opt.category
        pre_transform = T.NormalizeScale()
        train_dataset = ShapeNet(self._data_path, self._category, train=True,
                                 pre_transform=pre_transform)
        test_dataset = ShapeNet(self._data_path, self._category, train=False,
                                pre_transform=pre_transform)

        # The function creates the associated dataset using the training arguments and some dataset_opt
        self._create_dataloaders(train_dataset, test_dataset, val_dataset=None)
```

Here is the implementation of BaseDataset

```python
class BaseDataset():
    def __init__(self, dataset_opt, training_opt):
        self.dataset_opt = dataset_opt
        self.training_opt = training_opt
        self.strategies = {}

    def _create_dataloaders(self, train_dataset,  test_dataset, val_dataset=None):
        """ Creates the data loaders. Must be called in order to complete the setup of the Dataset
        """
        self._num_classes = train_dataset.num_classes
        self._feature_dimension = self.extract_point_dimension(train_dataset)
        self._train_loader = DataLoader(train_dataset, 
                                        batch_size=self.training_opt.batch_size, 
                                        shuffle=self.training_opt.shuffle,
                                        num_workers=self.training_opt.num_workers)

        self._test_loader = DataLoader(test_dataset, 
                                       batch_size=self.training_opt.batch_size, 
                                       shuffle=False,
                                       num_workers=self.training_opt.num_workers)

    def test_dataloader(self):
        return self._test_loader

    def train_dataloader(self):
        return self._train_loader

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def weight_classes(self):
        return getattr(self._train_loader.dataset, "weight_classes", None)

    @property
    def feature_dimension(self):
        return self._feature_dimension

    @staticmethod
    def extract_point_dimension(dataset: Dataset):
        sample = dataset[0]
        if sample.x is None:
            return 3  # (x,y,z)
        return sample.x.shape[1]

    def _set_multiscale_transform(self, batch_transform):
        for _, attr in self.__dict__.items():
            if isinstance(attr, DataLoader):
                def collate_fn(data_list): return BatchWithTransform.from_data_list_with_transform(
                    data_list, [], batch_transform)
                setattr(attr, "collate_fn", collate_fn)

    def set_strategies(self, model, precompute_multi_scale=False):
        strategies = model.get_sampling_and_search_strategies()
        batch_transform = MultiScaleTransform(strategies, precompute_multi_scale)
        self._set_multiscale_transform(batch_transform)
```

# Create a new model

