# Adding a new metric

Within the file ```src/metrics/model_checkpoint.py```,
It contains a mapping dictionnary between a sub ```metric_name``` and an ```optimization function```.

Currently, we support the following metrics.

```python
DEFAULT_METRICS_FUNC = {
    "iou": max,
    "acc": max,
    "loss": min,
    "mer": min,
}  # Those map subsentences to their optimization functions
```

# Model Saving

Our custom ```Checkpoint``` class keeps track of the models for ```every metric```, the stats for ```"train", "test", "val"```, ```optimizer``` and ```its learning params```.

```python
        self._objects = {}
        self._objects["models"] = {}
        self._objects["stats"] = {"train": [], "test": [], "val": []}
        self._objects["optimizer"] = None
        self._objects["lr_params"] = None
```

# Model Loading

In training.yaml and eval.yaml, you can find the followings parameters:

* weight_name
* checkpoint_dir
* resume

As the model is saved for every metric + the latest epoch.
It is possible by loading any of them using ```weight_name```.

Example: ```weight_name: "miou"```

If the checkpoint contains weight with the key "miou", it will set the model state to them. If not, it will try the latest if it exists. If None are found, the model will be randonmly initialized.

