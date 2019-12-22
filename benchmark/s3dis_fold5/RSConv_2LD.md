
```
# Relation-Shape Convolutional Neural Network for Point Cloud Analysis (https://arxiv.org/abs/1904.07601)
RSConv:
    type: RSConv
    down_conv:
        module_name: RSConv
        ratios: [0.2, 0.25]
        radius: [0.1, 0.2]
        local_nn: [[10, 8, 3], [10, 32, 64, 64]]
        down_conv_nn: [[6, 16, 32, 64], [64, 64, 128]]
    innermost:
        module_name: GlobalBaseModule
        aggr: max
        nn: [131, 128] #[3  + 128]
    up_conv:
        module_name: FPModule
        ratios: [1, 0.25, 0.2]
        radius: [0.2, 0.2, 0.1]
        up_conv_nn: [[256, 64], [128, 64], [64, 64]] #[128 + 128, ...], [64+64, ...]
        up_k: [1, 3, 3]
        skip: True
    mlp_cls:
        nn: [64, 64, 64, 64]
        dropout: 0.5
```

```
CLASS WEIGHT : {'ceiling': 0.0249, 'floor': 0.026, 'wall': 0.0301, 'column': 0.0805, 'beam': 0.1004, 'window': 0.1216, 'door': 0.0584, 'table': 0.0679, 'chair': 0.0542, 'bookcase': 0.179, 'sofa': 0.069, 'board': 0.1509, 'clutter': 0.0371}
SegmentationModel(
  (model): UnetSkipConnectionBlock(
    (down): RSConv(
      (_conv): Convolution(
        (local_nn): Sequential(
          (0): Sequential(
            (0): Linear(in_features=10, out_features=8, bias=True)
            (1): ReLU()
            (2): BatchNorm1d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): Sequential(
            (0): Linear(in_features=8, out_features=6, bias=True)
            (1): ReLU()
            (2): BatchNorm1d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (activation): ReLU()
        (global_nn): Sequential(
          (0): Sequential(
            (0): Linear(in_features=6, out_features=16, bias=True)
            (1): ReLU()
            (2): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): Sequential(
            (0): Linear(in_features=16, out_features=32, bias=True)
            (1): ReLU()
            (2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (2): Sequential(
            (0): Linear(in_features=32, out_features=64, bias=True)
            (1): ReLU()
            (2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
    )
    (submodule): UnetSkipConnectionBlock(
      (down): RSConv(
        (_conv): Convolution(
          (local_nn): Sequential(
            (0): Sequential(
              (0): Linear(in_features=10, out_features=32, bias=True)
              (1): ReLU()
              (2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Sequential(
              (0): Linear(in_features=32, out_features=64, bias=True)
              (1): ReLU()
              (2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (2): Sequential(
              (0): Linear(in_features=64, out_features=64, bias=True)
              (1): ReLU()
              (2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (activation): ReLU()
          (global_nn): Sequential(
            (0): Sequential(
              (0): Linear(in_features=64, out_features=64, bias=True)
              (1): ReLU()
              (2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Sequential(
              (0): Linear(in_features=64, out_features=128, bias=True)
              (1): ReLU()
              (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
      )
      (submodule): UnetSkipConnectionBlock(
        (inner): GlobalBaseModule(
          (nn): Sequential(
            (0): Sequential(
              (0): Linear(in_features=131, out_features=128, bias=True)
              (1): ReLU()
              (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
        (up): FPModule(
          (nn): Sequential(
            (0): Sequential(
              (0): Linear(in_features=256, out_features=64, bias=True)
              (1): ReLU()
              (2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
      )
      (up): FPModule(
        (nn): Sequential(
          (0): Sequential(
            (0): Linear(in_features=128, out_features=64, bias=True)
            (1): ReLU()
            (2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
    )
    (up): FPModule(
      (nn): Sequential(
        (0): Sequential(
          (0): Linear(in_features=70, out_features=64, bias=True)
          (1): ReLU()
          (2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
  )
)
Model size = 78919
```

EPOCH 105 / 350
```
100%|███████████████████████████████████████████████████| 1395/1395 [08:40<00:00,  2.68it/s, data_loading=0.002, iteration=0.161, train_acc=92.19, train_loss_seg=0.278, train_macc=88.40, train_miou=60.00]
```
```
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 571/571 [01:53<00:00,  5.01it/s, test_acc=84.50, test_loss_seg=0.492, test_macc=74.96, test_miou=50.16]
```
