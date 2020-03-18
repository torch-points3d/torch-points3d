## Create a new dataset

Let's add [```S3DIS```](http://buildingparser.stanford.edu/dataset.html) dataset to the project.

We are going to go through the successive steps to do so:

*  Choose the associated task related to your dataset.

*  Create a new ```.yaml file``` used for configuring your dataset within ```conf/data/{your_task}/{your_dataset_name}.yaml```

*  Add your own custom configuration needed to parametrize your dataset

*  Create a new file ```src/datasets/{your_task}/{your_dataset}.py```

*  Implement your dataset to inherit from ```BaseDataset```

* Associate a ```metric tracker``` to your dataset.

* Implement your custom ```metric tracker```.

Let's go throught those steps together.

### Choose the associated task for S3DIS

We are going to focus on semantic segmentation.
Our data are going to be a colored rgb pointcloud associated where each point has been associated to its own class.

The associated task is ```segmentation```.

### Create a new configuration file

Let's create ```conf/data/segmentation/s3dis.yaml``` file with our own setting to setup the dataset

```yaml
data:
    task: segmentation
    class: s3dis.S3DISDataset
    dataroot: data
    fold: 5
    class_weight_method: "sqrt"
    room_points: 32768
    num_points: 4096
    first_subsampling: 0.04
    density_parameter: 5.0
    kp_extent: 1.0
```

Here, one need note some very important parameters !

* ```task``` needs to be specified. Currently, the arguments provided by the command line are lost and therefore we need the extra information.

* ```class``` needs to be specified. It is structured in the following: {dataset_file}/{dataset_class_name}. In order to create this dataset, we will look into 
```src/datasets/segmentation/s3dis.py``` file and get the ```S3DISDataset``` from it.
The remaining params will be given to the class along the training params.

### Create the actual implementation

Now, create a new file ```src/datasets/segmentation/s3dis.py``` with the class ```S3DISDataset``` inside.

Before starting, we strongly advice to read the [```Creating Your Own Datasets```](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html) from ```Pytorch Geometric```

```python
class S3DISDataset(BaseDataset):
    def __init__(self, dataset_opt, training_opt):
        super().__init__(dataset_opt, training_opt)
        self._data_path = os.path.join(dataset_opt.dataroot, "S3DIS")

        pre_transform = cT.GridSampling(dataset_opt.first_subsampling, 13)

        transform = T.Compose(
            [T.FixedPoints(dataset_opt.num_points), T.RandomTranslate(0.01), T.RandomRotate(180, axis=2),]
        )

        train_dataset = S3DIS_With_Weights(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=True,
            pre_transform=pre_transform,
            transform=transform,
            class_weight_method=dataset_opt.class_weight_method,
        )
        test_dataset = S3DIS_With_Weights(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=False,
            pre_transform=pre_transform,
            transform=T.FixedPoints(dataset_opt.num_points),
        )

        self._create_dataloaders(train_dataset, test_dataset, val_dataset=None)

    @staticmethod
    def get_tracker(model, task: str, dataset, wandb_opt: bool, tensorboard_opt: bool):
        """Factory method for the tracker

        Arguments:
            task {str} -- task description
            dataset {[type]}
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(dataset, wandb_log=wandb_opt.log, use_tensorboard=tensorboard_opt.log)
```

Let's explain the code more in details there.

```python
class S3DISDataset(BaseDataset):
    def __init__(self, dataset_opt, training_opt):
        super().__init__(dataset_opt, training_opt)
        self._data_path = os.path.join(dataset_opt.dataroot, "S3DIS")
```

* We have create a dataset called ```S3DISDataset``` as referenced within our ```s3dis.yaml``` file.

* We can only observe the dataset inherit from ```BaseDataset```. Without it, the new dataset won't be working within the framework !

* ```self._data_path``` will be the place where the data will be saved.

```python
        pre_transform = cT.GridSampling(dataset_opt.first_subsampling, 13)

        transform = T.Compose(
            [T.FixedPoints(dataset_opt.num_points), T.RandomTranslate(0.01), T.RandomRotate(180, axis=2),]
        )

        train_dataset = S3DIS_With_Weights(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=True,
            pre_transform=pre_transform,
            transform=transform,
            class_weight_method=dataset_opt.class_weight_method,
        )
        test_dataset = S3DIS_With_Weights(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=False,
            pre_transform=pre_transform,
            transform=T.FixedPoints(dataset_opt.num_points),
        )
```

This part creates some transform and train / test dataset.

```python
self._create_dataloaders(train_dataset, test_dataset, val_dataset=None)
```

This line is important. It is going to wrap your datasets directly within the correct dataloader. Don't forget to call this function. Also, we can observe it is possible to provide a ```val_dataset```.

<h4> Associate a ```metric tracker``` to your dataset </h4>

```python
    @staticmethod
    def get_tracker(model, task: str, dataset, wandb_opt: bool, tensorboard_opt: bool):
        """Factory method for the tracker

        Arguments:
            task {str} -- task description
            dataset {[type]}
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(dataset, wandb_log=wandb_opt.log, use_tensorboard=tensorboard_opt.log)
```

Finally, one needs to implement the ```@staticmethod get_tracker``` method with ```model, task: str, dataset, wandb_opt: bool, tensorboard_opt: bool``` as parameters.

### What about the segmentation tracker?

```python
class SegmentationTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False):
        """ This is a generic tracker for segmentation tasks.
        It uses a confusion matrix in the back-end to track results.
        Use the tracker to track an epoch.
        You can use the reset function before you start a new epoch

        Arguments:
            dataset  -- dataset to track (used for the number of classes)

        Keyword Arguments:
            stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
            wandb_log {str} --  Log using weight and biases
        """
        super(SegmentationTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self._num_classes = dataset.num_classes
        self.reset(stage)

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._confusion_matrix = ConfusionMatrix(self._num_classes)

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.confusion_matrix

    def track(self, model: BaseModel):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        outputs = self._convert(model.get_output())
        targets = self._convert(model.get_labels())
        assert outputs.shape[0] == len(targets)
        self._confusion_matrix.count_predicted_batch(targets, np.argmax(outputs, 1))

        self._acc = 100 * self._confusion_matrix.get_overall_accuracy()
        self._macc = 100 * self._confusion_matrix.get_mean_class_accuracy()
        self._miou = 100 * self._confusion_matrix.get_average_intersection_union()

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_acc".format(self._stage)] = self._acc
        metrics["{}_macc".format(self._stage)] = self._macc
        metrics["{}_miou".format(self._stage)] = self._miou

        return metrics
```

The tracker needs to inherit from the ```BaseTracker``` and implements the following methods:

* ```reset```: The tracker need to be reset when switching to a new stage ```["train", "test", "val"]```

* ```track```: This function is responsible to implement your metrics

* ```get_metrics```: This function is responsible to return a dictionnary with all the tracked metrics for your dataset.