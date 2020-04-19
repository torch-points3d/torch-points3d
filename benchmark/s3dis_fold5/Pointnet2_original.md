```
(superpoint-graph-job-py3.6) ➜  deeppointcloud-benchmarks git:(pn2) ✗ poetry run python train.py experiment.model_name=pointnet2_kc experiment.dataset=s3dis experiment.experiment_name=15-59-58
CLASS WEIGHT : {'ceiling': 0.0249, 'floor': 0.026, 'wall': 0.0301, 'column': 0.0805, 'beam': 0.1004, 'window': 0.1216, 'door': 0.0584, 'table': 0.0679, 'chair': 0.0542, 'bookcase': 0.179, 'sofa': 0.069, 'board': 0.1509, 'clutter': 0.0371}
SegmentationModel(
  (SA_modules): ModuleList(
    (0): PointnetSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(9, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer1): Conv2d(
            (conv): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer2): Conv2d(
            (conv): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
        )
        (1): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(9, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer1): Conv2d(
            (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer2): Conv2d(
            (conv): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
        )
      )
    )
    (1): PointnetSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(99, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer1): Conv2d(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer2): Conv2d(
            (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
        )
        (1): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(99, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer1): Conv2d(
            (conv): Conv2d(64, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer2): Conv2d(
            (conv): Conv2d(96, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
        )
      )
    )
    (2): PointnetSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(259, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer1): Conv2d(
            (conv): Conv2d(128, 196, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(196, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer2): Conv2d(
            (conv): Conv2d(196, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
        )
        (1): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(259, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer1): Conv2d(
            (conv): Conv2d(128, 196, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(196, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer2): Conv2d(
            (conv): Conv2d(196, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
        )
      )
    )
    (3): PointnetSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(515, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer1): Conv2d(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer2): Conv2d(
            (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
        )
        (1): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(515, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer1): Conv2d(
            (conv): Conv2d(256, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
          (layer2): Conv2d(
            (conv): Conv2d(384, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace)
          )
        )
      )
    )
  )
  (FP_modules): ModuleList(
    (0): PointnetFPModule(
      (mlp): SharedMLP(
        (layer0): Conv2d(
          (conv): Conv2d(262, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normlayer): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
        )
        (layer1): Conv2d(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normlayer): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
        )
      )
    )
    (1): PointnetFPModule(
      (mlp): SharedMLP(
        (layer0): Conv2d(
          (conv): Conv2d(608, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normlayer): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
        )
        (layer1): Conv2d(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normlayer): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
        )
      )
    )
    (2): PointnetFPModule(
      (mlp): SharedMLP(
        (layer0): Conv2d(
          (conv): Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normlayer): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
        )
        (layer1): Conv2d(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normlayer): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
        )
      )
    )
    (3): PointnetFPModule(
      (mlp): SharedMLP(
        (layer0): Conv2d(
          (conv): Conv2d(1536, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normlayer): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
        )
        (layer1): Conv2d(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normlayer): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
        )
      )
    )
  )
  (FC_layer): Seq(
    (0): Conv1d(
      (conv): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
      (normlayer): BatchNorm1d(
        (bn): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (activation): ReLU(inplace)
    )
    (1): Dropout(p=0.5)
    (2): Conv1d(
      (conv): Conv1d(128, 13, kernel_size=(1,), stride=(1,))
    )
  )
)
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.01
    weight_decay: 0
)
Model size = 3026829
Access tensorboard with the following command <tensorboard --logdir=/home/thomas/HELIX/research/deeppointcloud-benchmarks/outputs/2020-01-07/15-59-58/tensorboard>
EPOCH 1 / 350
  0%|                                                                                                                                                                               | 0/523 [00:00<?, ?it/s]THCudaCheck FAIL file=/pytorch/aten/torch_points3d/THC/THCGeneral.cpp line=383 error=11 : invalid argument
100%|█████████████████████████████████████████████████████| 523/523 [04:29<00:00,  1.94it/s, data_loading=0.320, iteration=0.470, train_acc=73.18, train_loss_seg=1.082, train_macc=53.47, train_miou=35.38]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.20it/s, test_acc=75.69, test_loss_seg=0.998, test_macc=61.33, test_miou=34.12]

EPOCH 2 / 350
100%|█████████████████████████████████████████████████████| 523/523 [05:09<00:00,  1.69it/s, data_loading=0.327, iteration=0.455, train_acc=80.34, train_loss_seg=0.783, train_macc=66.40, train_miou=45.55]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.18it/s, test_acc=73.41, test_loss_seg=0.911, test_macc=61.37, test_miou=33.20]
loss_seg: 0.9982370138168335 -> 0.9110346436500549, macc: 61.33574242365333 -> 61.377196530312936

EPOCH 3 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:41<00:00,  1.86it/s, data_loading=0.331, iteration=0.476, train_acc=82.36, train_loss_seg=0.677, train_macc=71.94, train_miou=49.87]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:11<00:00,  3.02it/s, test_acc=80.76, test_loss_seg=0.563, test_macc=67.97, test_miou=38.89]
loss_seg: 0.9110346436500549 -> 0.563483715057373, acc: 75.69741537404613 -> 80.76140647710756, macc: 61.377196530312936 -> 67.97780615141481, miou: 34.126744523142705 -> 38.895046430138535

EPOCH 4 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:46<00:00,  1.82it/s, data_loading=0.332, iteration=0.466, train_acc=83.81, train_loss_seg=0.615, train_macc=74.71, train_miou=52.82]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:09<00:00,  3.09it/s, test_acc=80.82, test_loss_seg=0.529, test_macc=69.60, test_miou=36.70]
loss_seg: 0.563483715057373 -> 0.5292543172836304, acc: 80.76140647710756 -> 80.82145513490195, macc: 67.97780615141481 -> 69.60467642834878

EPOCH 5 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:48<00:00,  1.82it/s, data_loading=0.336, iteration=0.464, train_acc=84.80, train_loss_seg=0.572, train_macc=76.60, train_miou=54.62]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.13it/s, test_acc=80.43, test_loss_seg=0.839, test_macc=70.67, test_miou=39.47]
macc: 69.60467642834878 -> 70.67034371324998, miou: 38.895046430138535 -> 39.47518720735008

EPOCH 6 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:43<00:00,  1.84it/s, data_loading=0.326, iteration=0.459, train_acc=85.68, train_loss_seg=0.537, train_macc=77.89, train_miou=56.21]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.13it/s, test_acc=82.38, test_loss_seg=0.535, test_macc=68.75, test_miou=38.79]
acc: 80.82145513490195 -> 82.38928861396256

EPOCH 7 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:39<00:00,  1.87it/s, data_loading=0.325, iteration=0.457, train_acc=86.40, train_loss_seg=0.501, train_macc=79.63, train_miou=57.89]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.16it/s, test_acc=76.86, test_loss_seg=0.630, test_macc=66.25, test_miou=35.61]

EPOCH 8 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:38<00:00,  1.88it/s, data_loading=0.338, iteration=0.466, train_acc=87.15, train_loss_seg=0.473, train_macc=80.78, train_miou=59.18]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.18it/s, test_acc=79.99, test_loss_seg=0.512, test_macc=69.68, test_miou=38.44]
loss_seg: 0.5292543172836304 -> 0.51248699426651

EPOCH 9 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:38<00:00,  1.88it/s, data_loading=0.331, iteration=0.465, train_acc=87.69, train_loss_seg=0.452, train_macc=81.48, train_miou=60.15]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.15it/s, test_acc=82.55, test_loss_seg=0.545, test_macc=70.58, test_miou=40.43]
acc: 82.38928861396256 -> 82.55865052688948, miou: 39.47518720735008 -> 40.43524252361075

EPOCH 10 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:37<00:00,  1.88it/s, data_loading=0.324, iteration=0.476, train_acc=88.20, train_loss_seg=0.424, train_macc=82.79, train_miou=61.47]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.20it/s, test_acc=81.87, test_loss_seg=0.385, test_macc=69.04, test_miou=40.39]
loss_seg: 0.51248699426651 -> 0.3858228325843811

EPOCH 11 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:33<00:00,  1.91it/s, data_loading=0.327, iteration=0.460, train_acc=88.64, train_loss_seg=0.408, train_macc=83.38, train_miou=61.98]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.21it/s, test_acc=82.11, test_loss_seg=0.393, test_macc=70.55, test_miou=39.37]

EPOCH 12 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.329, iteration=0.458, train_acc=88.96, train_loss_seg=0.391, train_macc=84.36, train_miou=63.03]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.23it/s, test_acc=83.40, test_loss_seg=0.233, test_macc=70.19, test_miou=41.02]
loss_seg: 0.3858228325843811 -> 0.2336183786392212, acc: 82.55865052688948 -> 83.40356871139171, miou: 40.43524252361075 -> 41.021470293663725

EPOCH 13 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.331, iteration=0.465, train_acc=89.41, train_loss_seg=0.375, train_macc=84.59, train_miou=63.92]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.20it/s, test_acc=82.00, test_loss_seg=0.339, test_macc=70.33, test_miou=41.84]
miou: 41.021470293663725 -> 41.84289022643496

EPOCH 14 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.330, iteration=0.452, train_acc=89.90, train_loss_seg=0.354, train_macc=85.73, train_miou=64.91]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.16it/s, test_acc=83.10, test_loss_seg=0.311, test_macc=69.94, test_miou=41.03]

EPOCH 15 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:37<00:00,  1.89it/s, data_loading=0.344, iteration=0.452, train_acc=90.04, train_loss_seg=0.347, train_macc=85.96, train_miou=65.52]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.16it/s, test_acc=81.96, test_loss_seg=0.295, test_macc=69.19, test_miou=40.05]

EPOCH 16 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.333, iteration=0.461, train_acc=90.43, train_loss_seg=0.328, train_macc=86.81, train_miou=66.43]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.20it/s, test_acc=83.43, test_loss_seg=0.434, test_macc=71.08, test_miou=41.53]
acc: 83.40356871139171 -> 83.43692868254911, macc: 70.67034371324998 -> 71.08029098060784

EPOCH 17 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.325, iteration=0.461, train_acc=90.69, train_loss_seg=0.32 , train_macc=87.06, train_miou=66.71]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.18it/s, test_acc=83.33, test_loss_seg=0.269, test_macc=71.73, test_miou=41.16]
macc: 71.08029098060784 -> 71.73158063103956

EPOCH 18 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:37<00:00,  1.89it/s, data_loading=0.323, iteration=0.462, train_acc=91.20, train_loss_seg=0.294, train_macc=87.99, train_miou=68.31]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.13it/s, test_acc=82.44, test_loss_seg=0.307, test_macc=70.42, test_miou=40.09]

EPOCH 19 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.323, iteration=0.468, train_acc=91.32, train_loss_seg=0.291, train_macc=88.27, train_miou=68.26]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.17it/s, test_acc=84.06, test_loss_seg=0.212, test_macc=71.29, test_miou=43.01]
loss_seg: 0.2336183786392212 -> 0.21208004653453827, acc: 83.43692868254911 -> 84.06896546829583, miou: 41.84289022643496 -> 43.01301018966881

EPOCH 20 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:37<00:00,  1.88it/s, data_loading=0.333, iteration=0.460, train_acc=91.62, train_loss_seg=0.281, train_macc=88.50, train_miou=69.17]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.14it/s, test_acc=83.58, test_loss_seg=0.254, test_macc=70.92, test_miou=41.98]

EPOCH 21 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:37<00:00,  1.88it/s, data_loading=0.339, iteration=0.464, train_acc=91.81, train_loss_seg=0.272, train_macc=88.65, train_miou=69.02]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.13it/s, test_acc=83.31, test_loss_seg=0.318, test_macc=70.13, test_miou=42.24]

EPOCH 22 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:37<00:00,  1.89it/s, data_loading=0.334, iteration=0.462, train_acc=92.04, train_loss_seg=0.261, train_macc=89.32, train_miou=70.17]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.17it/s, test_acc=84.69, test_loss_seg=0.224, test_macc=72.45, test_miou=43.03]
acc: 84.06896546829583 -> 84.69125082326487, macc: 71.73158063103956 -> 72.45769781266885, miou: 43.01301018966881 -> 43.03650049598167

EPOCH 23 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:38<00:00,  1.88it/s, data_loading=0.336, iteration=0.459, train_acc=92.09, train_loss_seg=0.261, train_macc=89.61, train_miou=70.26]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.14it/s, test_acc=83.80, test_loss_seg=0.197, test_macc=71.36, test_miou=42.44]
loss_seg: 0.21208004653453827 -> 0.19700782001018524

EPOCH 24 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:37<00:00,  1.89it/s, data_loading=0.314, iteration=0.456, train_acc=92.48, train_loss_seg=0.246, train_macc=89.72, train_miou=71.14]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.13it/s, test_acc=84.98, test_loss_seg=0.151, test_macc=71.98, test_miou=44.43]
loss_seg: 0.19700782001018524 -> 0.15112227201461792, acc: 84.69125082326487 -> 84.98886463253999, miou: 43.03650049598167 -> 44.439731484668506

EPOCH 25 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.90it/s, data_loading=0.324, iteration=0.452, train_acc=92.77, train_loss_seg=0.234, train_macc=90.51, train_miou=71.85]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.17it/s, test_acc=84.12, test_loss_seg=0.221, test_macc=70.77, test_miou=42.35]

EPOCH 26 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.91it/s, data_loading=0.315, iteration=0.449, train_acc=92.93, train_loss_seg=0.227, train_macc=90.85, train_miou=72.26]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.12it/s, test_acc=83.00, test_loss_seg=0.288, test_macc=69.50, test_miou=42.85]

EPOCH 27 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.322, iteration=0.454, train_acc=93.00, train_loss_seg=0.225, train_macc=90.99, train_miou=72.58]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.20it/s, test_acc=83.31, test_loss_seg=0.235, test_macc=70.93, test_miou=42.57]

EPOCH 28 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.319, iteration=0.460, train_acc=93.36, train_loss_seg=0.213, train_macc=91.15, train_miou=73.26]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.17it/s, test_acc=83.88, test_loss_seg=0.183, test_macc=70.78, test_miou=42.46]

EPOCH 29 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.334, iteration=0.449, train_acc=93.19, train_loss_seg=0.217, train_macc=91.11, train_miou=73.31]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.18it/s, test_acc=82.47, test_loss_seg=0.242, test_macc=71.12, test_miou=40.42]

EPOCH 30 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.328, iteration=0.461, train_acc=93.53, train_loss_seg=0.204, train_macc=91.65, train_miou=73.97]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.18it/s, test_acc=82.15, test_loss_seg=0.203, test_macc=71.51, test_miou=39.71]

EPOCH 31 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.324, iteration=0.453, train_acc=93.54, train_loss_seg=0.204, train_macc=91.61, train_miou=73.81]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:09<00:00,  3.08it/s, test_acc=83.94, test_loss_seg=0.235, test_macc=70.03, test_miou=43.26]

EPOCH 32 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.325, iteration=0.456, train_acc=93.69, train_loss_seg=0.200, train_macc=91.95, train_miou=74.49]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.18it/s, test_acc=84.11, test_loss_seg=0.166, test_macc=71.31, test_miou=41.01]

EPOCH 33 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.90it/s, data_loading=0.327, iteration=0.469, train_acc=93.82, train_loss_seg=0.193, train_macc=92.10, train_miou=74.67]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.16it/s, test_acc=84.75, test_loss_seg=0.185, test_macc=73.11, test_miou=44.23]
macc: 72.45769781266885 -> 73.11160574504926

EPOCH 34 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.329, iteration=0.459, train_acc=93.74, train_loss_seg=0.199, train_macc=91.91, train_miou=74.15]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.18it/s, test_acc=83.20, test_loss_seg=0.150, test_macc=71.73, test_miou=42.17]
loss_seg: 0.15112227201461792 -> 0.15026850998401642

EPOCH 35 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.329, iteration=0.451, train_acc=94.19, train_loss_seg=0.179, train_macc=92.64, train_miou=75.81]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.19it/s, test_acc=84.23, test_loss_seg=0.190, test_macc=71.86, test_miou=43.93]

EPOCH 36 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.332, iteration=0.456, train_acc=94.30, train_loss_seg=0.176, train_macc=92.89, train_miou=76.17]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.14it/s, test_acc=82.42, test_loss_seg=0.261, test_macc=68.87, test_miou=38.35]

EPOCH 37 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.325, iteration=0.460, train_acc=94.05, train_loss_seg=0.186, train_macc=92.43, train_miou=75.26]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.16it/s, test_acc=84.07, test_loss_seg=0.165, test_macc=70.83, test_miou=42.26]

EPOCH 38 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.331, iteration=0.464, train_acc=94.14, train_loss_seg=0.182, train_macc=92.75, train_miou=75.89]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.20it/s, test_acc=83.08, test_loss_seg=0.171, test_macc=70.10, test_miou=41.86]

EPOCH 39 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:33<00:00,  1.91it/s, data_loading=0.319, iteration=0.451, train_acc=94.48, train_loss_seg=0.169, train_macc=93.20, train_miou=76.81]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.22it/s, test_acc=84.71, test_loss_seg=0.132, test_macc=71.43, test_miou=44.33]
loss_seg: 0.15026850998401642 -> 0.13270394504070282

EPOCH 40 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.327, iteration=0.462, train_acc=94.69, train_loss_seg=0.163, train_macc=93.58, train_miou=77.13]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.17it/s, test_acc=84.79, test_loss_seg=0.129, test_macc=71.15, test_miou=42.89]
loss_seg: 0.13270394504070282 -> 0.12961801886558533

EPOCH 41 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.91it/s, data_loading=0.328, iteration=0.457, train_acc=94.54, train_loss_seg=0.167, train_macc=93.25, train_miou=76.86]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.20it/s, test_acc=84.38, test_loss_seg=0.196, test_macc=71.61, test_miou=41.87]

EPOCH 42 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.340, iteration=0.455, train_acc=94.73, train_loss_seg=0.161, train_macc=93.53, train_miou=77.72]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.25it/s, test_acc=84.59, test_loss_seg=0.167, test_macc=71.84, test_miou=44.35]

EPOCH 43 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.91it/s, data_loading=0.325, iteration=0.463, train_acc=94.70, train_loss_seg=0.163, train_macc=93.26, train_miou=77.40]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.20it/s, test_acc=83.36, test_loss_seg=0.166, test_macc=70.89, test_miou=41.52]

EPOCH 44 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.90it/s, data_loading=0.325, iteration=0.452, train_acc=94.84, train_loss_seg=0.156, train_macc=93.64, train_miou=77.39]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.25it/s, test_acc=83.43, test_loss_seg=0.158, test_macc=71.14, test_miou=41.82]

EPOCH 45 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:32<00:00,  1.92it/s, data_loading=0.323, iteration=0.461, train_acc=94.50, train_loss_seg=0.176, train_macc=93.02, train_miou=76.19]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.25it/s, test_acc=84.02, test_loss_seg=0.161, test_macc=71.02, test_miou=42.06]

EPOCH 46 / 350
100%|█████████████████████████████████████████████████████| 523/523 [05:06<00:00,  1.70it/s, data_loading=0.338, iteration=0.461, train_acc=94.97, train_loss_seg=0.153, train_macc=93.70, train_miou=77.66]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.18it/s, test_acc=84.55, test_loss_seg=0.115, test_macc=71.61, test_miou=42.24]
loss_seg: 0.12961801886558533 -> 0.11495383083820343

EPOCH 47 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.333, iteration=0.463, train_acc=95.36, train_loss_seg=0.139, train_macc=94.37, train_miou=79.42]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.14it/s, test_acc=84.62, test_loss_seg=0.154, test_macc=72.11, test_miou=43.54]

EPOCH 48 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.328, iteration=0.457, train_acc=95.34, train_loss_seg=0.138, train_macc=94.43, train_miou=79.75]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.24it/s, test_acc=84.35, test_loss_seg=0.110, test_macc=70.68, test_miou=42.43]
loss_seg: 0.11495383083820343 -> 0.11047439277172089

EPOCH 49 / 350
100%|█████████████████████████████████████████████████████| 523/523 [05:05<00:00,  1.71it/s, data_loading=0.332, iteration=0.469, train_acc=94.97, train_loss_seg=0.153, train_macc=93.95, train_miou=78.14]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.13it/s, test_acc=84.87, test_loss_seg=0.195, test_macc=71.97, test_miou=44.80]
miou: 44.439731484668506 -> 44.80838627244144

EPOCH 50 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:38<00:00,  1.88it/s, data_loading=0.333, iteration=0.462, train_acc=95.29, train_loss_seg=0.139, train_macc=94.41, train_miou=79.09]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.17it/s, test_acc=84.60, test_loss_seg=0.158, test_macc=71.63, test_miou=43.43]

EPOCH 51 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:37<00:00,  1.88it/s, data_loading=0.322, iteration=0.461, train_acc=95.36, train_loss_seg=0.138, train_macc=94.42, train_miou=79.55]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:09<00:00,  3.10it/s, test_acc=84.55, test_loss_seg=0.098, test_macc=70.97, test_miou=43.26]
loss_seg: 0.11047439277172089 -> 0.09837505966424942

EPOCH 52 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:38<00:00,  1.88it/s, data_loading=0.335, iteration=0.470, train_acc=95.31, train_loss_seg=0.140, train_macc=94.25, train_miou=78.97]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.18it/s, test_acc=84.49, test_loss_seg=0.204, test_macc=71.37, test_miou=42.48]

EPOCH 53 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.90it/s, data_loading=0.334, iteration=0.450, train_acc=95.43, train_loss_seg=0.136, train_macc=94.42, train_miou=79.70]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.22it/s, test_acc=84.37, test_loss_seg=0.130, test_macc=71.28, test_miou=41.90]

EPOCH 54 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:31<00:00,  1.92it/s, data_loading=0.323, iteration=0.453, train_acc=95.47, train_loss_seg=0.134, train_macc=94.46, train_miou=79.92]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.20it/s, test_acc=84.68, test_loss_seg=0.106, test_macc=71.52, test_miou=43.73]

EPOCH 55 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.324, iteration=0.453, train_acc=95.67, train_loss_seg=0.126, train_macc=94.63, train_miou=80.24]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.21it/s, test_acc=84.83, test_loss_seg=0.124, test_macc=71.88, test_miou=44.05]

EPOCH 56 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.91it/s, data_loading=0.324, iteration=0.450, train_acc=95.53, train_loss_seg=0.133, train_macc=94.66, train_miou=80.05]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:09<00:00,  3.12it/s, test_acc=84.29, test_loss_seg=0.177, test_macc=70.81, test_miou=43.80]

EPOCH 57 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.91it/s, data_loading=0.322, iteration=0.458, train_acc=95.39, train_loss_seg=0.138, train_macc=94.43, train_miou=79.38]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.18it/s, test_acc=84.19, test_loss_seg=0.208, test_macc=70.90, test_miou=42.96]

EPOCH 58 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.324, iteration=0.464, train_acc=95.61, train_loss_seg=0.130, train_macc=94.69, train_miou=80.01]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.15it/s, test_acc=84.55, test_loss_seg=0.112, test_macc=70.99, test_miou=41.65]

EPOCH 59 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.325, iteration=0.453, train_acc=95.80, train_loss_seg=0.122, train_macc=95.03, train_miou=81.05]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.24it/s, test_acc=84.23, test_loss_seg=0.138, test_macc=72.23, test_miou=42.83]

EPOCH 60 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:37<00:00,  1.89it/s, data_loading=0.326, iteration=0.456, train_acc=95.75, train_loss_seg=0.125, train_macc=94.90, train_miou=80.58]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.14it/s, test_acc=84.16, test_loss_seg=0.100, test_macc=70.87, test_miou=42.73]

EPOCH 61 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.323, iteration=0.459, train_acc=95.80, train_loss_seg=0.123, train_macc=94.99, train_miou=80.98]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.17it/s, test_acc=84.52, test_loss_seg=0.168, test_macc=71.90, test_miou=43.64]

EPOCH 62 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:37<00:00,  1.89it/s, data_loading=0.331, iteration=0.464, train_acc=95.94, train_loss_seg=0.118, train_macc=95.10, train_miou=80.99]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.14it/s, test_acc=83.74, test_loss_seg=0.127, test_macc=70.53, test_miou=41.60]

EPOCH 63 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.344, iteration=0.451, train_acc=95.67, train_loss_seg=0.128, train_macc=94.89, train_miou=80.30]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.16it/s, test_acc=84.29, test_loss_seg=0.140, test_macc=71.06, test_miou=41.74]

EPOCH 64 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.328, iteration=0.456, train_acc=96.12, train_loss_seg=0.111, train_macc=95.37, train_miou=82.25]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.17it/s, test_acc=83.05, test_loss_seg=0.139, test_macc=71.48, test_miou=42.18]

EPOCH 65 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.325, iteration=0.461, train_acc=95.73, train_loss_seg=0.128, train_macc=94.82, train_miou=80.18]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.16it/s, test_acc=85.08, test_loss_seg=0.077, test_macc=70.91, test_miou=43.16]
loss_seg: 0.09837505966424942 -> 0.07739892601966858, acc: 84.98886463253999 -> 85.0884193597838

EPOCH 66 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:37<00:00,  1.88it/s, data_loading=0.324, iteration=0.479, train_acc=95.70, train_loss_seg=0.128, train_macc=94.81, train_miou=80.21]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.20it/s, test_acc=84.17, test_loss_seg=0.123, test_macc=71.74, test_miou=41.97]

EPOCH 67 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.91it/s, data_loading=0.320, iteration=0.457, train_acc=96.04, train_loss_seg=0.114, train_macc=95.37, train_miou=82.08]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.22it/s, test_acc=84.49, test_loss_seg=0.119, test_macc=71.48, test_miou=43.67]

EPOCH 68 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.91it/s, data_loading=0.332, iteration=0.454, train_acc=96.13, train_loss_seg=0.110, train_macc=95.51, train_miou=82.13]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.21it/s, test_acc=84.97, test_loss_seg=0.127, test_macc=72.48, test_miou=44.31]

EPOCH 69 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.91it/s, data_loading=0.331, iteration=0.454, train_acc=96.16, train_loss_seg=0.110, train_macc=95.54, train_miou=82.26]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.19it/s, test_acc=84.71, test_loss_seg=0.083, test_macc=70.67, test_miou=43.37]

EPOCH 70 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:33<00:00,  1.91it/s, data_loading=0.330, iteration=0.456, train_acc=96.23, train_loss_seg=0.108, train_macc=95.54, train_miou=82.62]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.16it/s, test_acc=84.86, test_loss_seg=0.094, test_macc=71.52, test_miou=43.10]

EPOCH 71 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.330, iteration=0.451, train_acc=96.21, train_loss_seg=0.108, train_macc=95.59, train_miou=82.09]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.16it/s, test_acc=84.41, test_loss_seg=0.112, test_macc=71.62, test_miou=42.50]

EPOCH 72 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.90it/s, data_loading=0.328, iteration=0.452, train_acc=96.06, train_loss_seg=0.115, train_macc=95.35, train_miou=81.45]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.17it/s, test_acc=84.80, test_loss_seg=0.120, test_macc=71.54, test_miou=43.00]

EPOCH 73 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.90it/s, data_loading=0.336, iteration=0.455, train_acc=95.86, train_loss_seg=0.123, train_macc=95.20, train_miou=81.20]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.18it/s, test_acc=83.71, test_loss_seg=0.129, test_macc=70.73, test_miou=40.08]

EPOCH 74 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.326, iteration=0.452, train_acc=96.15, train_loss_seg=0.112, train_macc=95.38, train_miou=81.31]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.18it/s, test_acc=84.92, test_loss_seg=0.088, test_macc=71.76, test_miou=41.76]

EPOCH 75 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.332, iteration=0.469, train_acc=96.31, train_loss_seg=0.105, train_macc=95.69, train_miou=82.84]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.19it/s, test_acc=84.66, test_loss_seg=0.152, test_macc=71.09, test_miou=42.08]

EPOCH 76 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.331, iteration=0.457, train_acc=96.28, train_loss_seg=0.108, train_macc=95.67, train_miou=82.70]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.18it/s, test_acc=84.69, test_loss_seg=0.104, test_macc=71.21, test_miou=42.81]

EPOCH 77 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:32<00:00,  1.92it/s, data_loading=0.317, iteration=0.458, train_acc=96.45, train_loss_seg=0.101, train_macc=95.90, train_miou=83.25]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.22it/s, test_acc=84.88, test_loss_seg=0.109, test_macc=71.48, test_miou=45.58]
miou: 44.80838627244144 -> 45.583527852040845

EPOCH 78 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.326, iteration=0.459, train_acc=96.18, train_loss_seg=0.111, train_macc=95.49, train_miou=82.25]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.18it/s, test_acc=84.90, test_loss_seg=0.138, test_macc=71.76, test_miou=42.66]

EPOCH 79 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:35<00:00,  1.90it/s, data_loading=0.330, iteration=0.453, train_acc=96.50, train_loss_seg=0.099, train_macc=95.97, train_miou=82.85]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.17it/s, test_acc=85.13, test_loss_seg=0.077, test_macc=71.45, test_miou=44.09]
loss_seg: 0.07739892601966858 -> 0.07701006531715393, acc: 85.0884193597838 -> 85.13295373251269

EPOCH 80 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.90it/s, data_loading=0.329, iteration=0.451, train_acc=96.38, train_loss_seg=0.103, train_macc=95.77, train_miou=82.97]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.22it/s, test_acc=84.03, test_loss_seg=0.068, test_macc=70.81, test_miou=42.61]
loss_seg: 0.07701006531715393 -> 0.06862835586071014

EPOCH 81 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:30<00:00,  1.93it/s, data_loading=0.322, iteration=0.454, train_acc=96.39, train_loss_seg=0.102, train_macc=95.87, train_miou=82.95]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:05<00:00,  3.28it/s, test_acc=84.78, test_loss_seg=0.248, test_macc=70.62, test_miou=44.48]

EPOCH 82 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.313, iteration=0.459, train_acc=96.46, train_loss_seg=0.101, train_macc=96.00, train_miou=83.53]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.19it/s, test_acc=84.42, test_loss_seg=0.083, test_macc=71.10, test_miou=42.54]

EPOCH 83 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:33<00:00,  1.91it/s, data_loading=0.316, iteration=0.456, train_acc=95.87, train_loss_seg=0.127, train_macc=94.91, train_miou=80.11]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.21it/s, test_acc=84.12, test_loss_seg=0.087, test_macc=70.99, test_miou=41.23]

EPOCH 84 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:33<00:00,  1.92it/s, data_loading=0.320, iteration=0.465, train_acc=96.59, train_loss_seg=0.097, train_macc=96.04, train_miou=83.78]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.17it/s, test_acc=84.86, test_loss_seg=0.069, test_macc=71.64, test_miou=44.13]

EPOCH 85 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:31<00:00,  1.93it/s, data_loading=0.320, iteration=0.446, train_acc=96.61, train_loss_seg=0.095, train_macc=96.06, train_miou=83.66]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.24it/s, test_acc=84.61, test_loss_seg=0.079, test_macc=71.61, test_miou=42.14]

EPOCH 86 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:33<00:00,  1.91it/s, data_loading=0.329, iteration=0.453, train_acc=96.54, train_loss_seg=0.098, train_macc=95.90, train_miou=83.48]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.17it/s, test_acc=84.79, test_loss_seg=0.088, test_macc=71.40, test_miou=43.40]

EPOCH 87 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:32<00:00,  1.92it/s, data_loading=0.324, iteration=0.443, train_acc=96.60, train_loss_seg=0.097, train_macc=96.17, train_miou=83.74]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.19it/s, test_acc=84.87, test_loss_seg=0.078, test_macc=72.29, test_miou=41.83]

EPOCH 88 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:33<00:00,  1.91it/s, data_loading=0.326, iteration=0.455, train_acc=96.47, train_loss_seg=0.101, train_macc=95.83, train_miou=82.83]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.21it/s, test_acc=83.82, test_loss_seg=0.130, test_macc=71.20, test_miou=40.69]

EPOCH 89 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.90it/s, data_loading=0.324, iteration=0.461, train_acc=96.44, train_loss_seg=0.104, train_macc=95.80, train_miou=82.77]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.20it/s, test_acc=84.00, test_loss_seg=0.123, test_macc=69.64, test_miou=40.34]

EPOCH 90 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.90it/s, data_loading=0.319, iteration=0.471, train_acc=96.56, train_loss_seg=0.098, train_macc=96.12, train_miou=83.30]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.16it/s, test_acc=84.24, test_loss_seg=0.102, test_macc=70.94, test_miou=42.32]

EPOCH 91 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:34<00:00,  1.91it/s, data_loading=0.320, iteration=0.451, train_acc=96.79, train_loss_seg=0.09 , train_macc=96.35, train_miou=84.52]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.25it/s, test_acc=84.29, test_loss_seg=0.080, test_macc=71.28, test_miou=42.60]

EPOCH 92 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:33<00:00,  1.91it/s, data_loading=0.318, iteration=0.450, train_acc=96.77, train_loss_seg=0.09 , train_macc=96.37, train_miou=84.68]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:13<00:00,  2.94it/s, test_acc=84.82, test_loss_seg=0.061, test_macc=71.26, test_miou=43.51]
loss_seg: 0.06862835586071014 -> 0.0614144541323185

EPOCH 93 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:40<00:00,  1.87it/s, data_loading=0.330, iteration=0.457, train_acc=96.64, train_loss_seg=0.095, train_macc=96.14, train_miou=84.0 ]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.16it/s, test_acc=84.29, test_loss_seg=0.087, test_macc=70.98, test_miou=42.75]

EPOCH 94 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:36<00:00,  1.89it/s, data_loading=0.335, iteration=0.456, train_acc=96.55, train_loss_seg=0.099, train_macc=95.96, train_miou=83.15]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:07<00:00,  3.20it/s, test_acc=83.66, test_loss_seg=0.154, test_macc=70.37, test_miou=39.60]

EPOCH 95 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:32<00:00,  1.92it/s, data_loading=0.320, iteration=0.453, train_acc=96.62, train_loss_seg=0.096, train_macc=96.02, train_miou=83.26]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.21it/s, test_acc=84.72, test_loss_seg=0.076, test_macc=71.05, test_miou=42.13]

EPOCH 96 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:32<00:00,  1.92it/s, data_loading=0.332, iteration=0.455, train_acc=96.75, train_loss_seg=0.091, train_macc=96.31, train_miou=84.42]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.25it/s, test_acc=85.26, test_loss_seg=0.078, test_macc=71.66, test_miou=44.13]
acc: 85.13295373251269 -> 85.26667395303411

EPOCH 97 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:32<00:00,  1.92it/s, data_loading=0.326, iteration=0.455, train_acc=96.69, train_loss_seg=0.092, train_macc=96.27, train_miou=84.20]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:06<00:00,  3.21it/s, test_acc=84.62, test_loss_seg=0.061, test_macc=71.09, test_miou=41.48]

EPOCH 98 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:31<00:00,  1.93it/s, data_loading=0.324, iteration=0.445, train_acc=96.75, train_loss_seg=0.092, train_macc=96.14, train_miou=84.26]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:05<00:00,  3.26it/s, test_acc=84.99, test_loss_seg=0.099, test_macc=70.84, test_miou=43.99]

EPOCH 99 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:32<00:00,  1.92it/s, data_loading=0.326, iteration=0.454, train_acc=96.59, train_loss_seg=0.099, train_macc=96.08, train_miou=83.40]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:08<00:00,  3.15it/s, test_acc=84.42, test_loss_seg=0.075, test_macc=71.49, test_miou=43.09]

EPOCH 100 / 350
100%|█████████████████████████████████████████████████████| 523/523 [04:32<00:00,  1.92it/s, data_loading=0.323, iteration=0.450, train_acc=96.90, train_loss_seg=0.085, train_macc=96.43, train_miou=85.00]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:12<00:00,  2.97it/s, test_acc=84.83, test_loss_seg=0.051, test_macc=71.33, test_miou=42.57]
loss_seg: 0.0614144541323185 -> 0.051259834319353104

BEST: 
* loss_seg: 0.051259834319353104
* acc: 85.26667395303411
* miou: 45.583527852040845
* macc: 73.11160574504926
```
