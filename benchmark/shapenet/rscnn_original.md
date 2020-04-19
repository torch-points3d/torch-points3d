(superpoint-graph-job-py3.6) ➜  deeppointcloud-benchmarks git:(RSConv_debug) ✗ poetry run python train.py experiment.model_name=RSConv_MSN experiment.dataset=shapenet wandb.log=False training.enable_cudnn=True training.batch_size=12 data.shapenet.normal=False
The down_conv_nn has a different size as radii. Make sure of have sharedMLP
Using category information for the predictions with 16 categories
SegmentationModel(
  (down_modules): ModuleList(
    (0): RSConvOriginalMSGDown: 2800 (ModuleList(
      (0): OriginalRSConv(ModuleList(
        (0): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      ))
      (1): OriginalRSConv(ModuleList(
        (0): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      ))
      (2): OriginalRSConv(ModuleList(
        (0): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      ))
    ), shared:  ModuleList(
      (0): Conv2d(10, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
      (2): Conv1d(16, 64, kernel_size=(1,), stride=(1,))
      (3): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
    ) ))
    (1): RSConvOriginalMSGDown: 34005 (ModuleList(
      (0): OriginalRSConv(ModuleList(
        (0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(195, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      ))
      (1): OriginalRSConv(ModuleList(
        (0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(195, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      ))
      (2): OriginalRSConv(ModuleList(
        (0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(195, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      ))
    ), shared:  ModuleList(
      (0): Conv2d(10, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): Conv2d(32, 195, kernel_size=(1, 1), stride=(1, 1))
      (2): Conv1d(195, 128, kernel_size=(1,), stride=(1,))
    ) ))
    (2): RSConvOriginalMSGDown: 129429 (ModuleList(
      (0): OriginalRSConv(ModuleList(
        (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(387, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      ))
      (1): OriginalRSConv(ModuleList(
        (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(387, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      ))
      (2): OriginalRSConv(ModuleList(
        (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(387, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      ))
    ), shared:  ModuleList(
      (0): Conv2d(10, 64, kernel_size=(1, 1), stride=(1, 1))
      (1): Conv2d(64, 387, kernel_size=(1, 1), stride=(1, 1))
      (2): Conv1d(387, 256, kernel_size=(1,), stride=(1,))
    ) ))
    (3): RSConvOriginalMSGDown: 504597 (ModuleList(
      (0): OriginalRSConv(ModuleList(
        (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(771, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      ))
      (1): OriginalRSConv(ModuleList(
        (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(771, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      ))
      (2): OriginalRSConv(ModuleList(
        (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(771, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      ))
    ), shared:  ModuleList(
      (0): Conv2d(10, 128, kernel_size=(1, 1), stride=(1, 1))
      (1): Conv2d(128, 771, kernel_size=(1, 1), stride=(1, 1))
      (2): Conv1d(771, 512, kernel_size=(1,), stride=(1,))
    ) ))
  )
  (inner_modules): ModuleList(
    (0): GlobalDenseBaseModule: 197376 (aggr=mean, SharedMLP(
      (layer0): Conv2d(
        (conv): Conv2d(1539, 128, kernel_size=(1, 1), stride=(1, 1))
        (normlayer): BatchNorm2d(
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): ReLU(inplace)
      )
    ))
    (1): GlobalDenseBaseModule: 99072 (aggr=mean, SharedMLP(
      (layer0): Conv2d(
        (conv): Conv2d(771, 128, kernel_size=(1, 1), stride=(1, 1))
        (normlayer): BatchNorm2d(
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): ReLU(inplace)
      )
    ))
  )
  (up_modules): ModuleList(
    (0): DenseFPModule: 1443840 (SharedMLP(
      (layer0): Conv2d(
        (conv): Conv2d(2304, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (normlayer): BatchNorm2d(
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): ReLU(inplace)
      )
      (layer1): Conv2d(
        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (normlayer): BatchNorm2d(
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): ReLU(inplace)
      )
    ))
    (1): DenseFPModule: 722944 (SharedMLP(
      (layer0): Conv2d(
        (conv): Conv2d(896, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (normlayer): BatchNorm2d(
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): ReLU(inplace)
      )
      (layer1): Conv2d(
        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (normlayer): BatchNorm2d(
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): ReLU(inplace)
      )
    ))
    (2): DenseFPModule: 246784 (SharedMLP(
      (layer0): Conv2d(
        (conv): Conv2d(704, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (normlayer): BatchNorm2d(
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): ReLU(inplace)
      )
      (layer1): Conv2d(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (normlayer): BatchNorm2d(
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): ReLU(inplace)
      )
    ))
    (3): DenseFPModule: 49664 (SharedMLP(
      (layer0): Conv2d(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (normlayer): BatchNorm2d(
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): ReLU(inplace)
      )
      (layer1): Conv2d(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (normlayer): BatchNorm2d(
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): ReLU(inplace)
      )
    ))
    (4): DenseFPModule: 0 (SharedMLP())
  )
  (FC_layer): Seq(
    (0): Conv1d(
      (conv): Conv1d(400, 128, kernel_size=(1,), stride=(1,), bias=False)
      (normlayer): BatchNorm1d(
        (bn): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (activation): ReLU(inplace)
    )
    (1): Dropout(p=0.5)
    (2): Conv1d(
      (conv): Conv1d(128, 50, kernel_size=(1,), stride=(1,))
    )
  )
)
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    initial_lr: 0.001
    lr: 0.001
    weight_decay: 0
)
Model size = 3488417
Access tensorboard with the following command <tensorboard --logdir=/home/thomas/HELIX/research/deeppointcloud-benchmarks/outputs/2020-01-19/22-39-50/tensorboard>
EPOCH 1 / 100
  0%|                                                                                                                                                                              | 0/1168 [00:00<?, ?it/s]THCudaCheck FAIL file=/pytorch/aten/torch_points3d/THC/THCGeneral.cpp line=383 error=11 : invalid argument
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:40<00:00,  3.43it/s, data_loading=0.006, iteration=0.133, train_Cmiou=56.47, train_Imiou=71.21, train_loss_seg=0.573]
Learning rate = 0.001000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:24<00:00,  9.92it/s, test_Cmiou=65.98, test_Imiou=77.52, test_loss_seg=0.267]
==================================================
    test_loss_seg = 0.26787325739860535
    test_Cmiou = 65.98094982236391
    test_Imiou = 77.52860172866566
    test_Imiou_per_class = {'Airplane': 0.7507927080254545, 'Bag': 0.4478062220982143, 'Cap': 0.765709929593222, 'Car': 0.5466032733722264, 'Chair': 0.8723628096946204, 'Earphone': 0.6261891998786752, 'Guitar': 0.8729472573841379, 'Knife': 0.5566930868522879, 'Lamp': 0.790361182717579, 'Laptop': 0.9467861984445894, 'Motorbike': 0.26363823782534224, 'Mug': 0.7473244359636818, 'Pistol': 0.7362160575938994, 'Rocket': 0.23672508548518303, 'Skateboard': 0.6136300257531923, 'Table': 0.7831662608959201}
==================================================
EPOCH 2 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:46<00:00,  3.37it/s, data_loading=0.006, iteration=0.134, train_Cmiou=69.28, train_Imiou=78.30, train_loss_seg=0.257]
Learning rate = 0.001000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:24<00:00,  9.93it/s, test_Cmiou=71.88, test_Imiou=80.26, test_loss_seg=1.058]
==================================================
    test_loss_seg = 1.0588456392288208
    test_Cmiou = 71.88677117909052
    test_Imiou = 80.26959477294649
    test_Imiou_per_class = {'Airplane': 0.768896690751059, 'Bag': 0.4832810325916728, 'Cap': 0.7834859882953452, 'Car': 0.6568410246290428, 'Chair': 0.8810223608776417, 'Earphone': 0.66536593550776, 'Guitar': 0.8680206749166361, 'Knife': 0.7876564133026259, 'Lamp': 0.7951810575208798, 'Laptop': 0.9472765942149636, 'Motorbike': 0.3423461218697208, 'Mug': 0.8329662531893036, 'Pistol': 0.7422060726848511, 'Rocket': 0.46069221081431927, 'Skateboard': 0.6842250340356498, 'Table': 0.802419923453011}
==================================================
Cmiou: 65.98094982236391 -> 71.88677117909052, Imiou: 77.52860172866566 -> 80.26959477294649
EPOCH 3 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:39<00:00,  3.44it/s, data_loading=0.006, iteration=0.133, train_Cmiou=71.45, train_Imiou=80.33, train_loss_seg=0.218]
Learning rate = 0.001000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.12it/s, test_Cmiou=74.38, test_Imiou=80.84, test_loss_seg=2.044]
==================================================
    test_loss_seg = 2.0445027351379395
    test_Cmiou = 74.38502038921175
    test_Imiou = 80.84875734591189
    test_Imiou_per_class = {'Airplane': 0.7674508273467864, 'Bag': 0.5676537065008816, 'Cap': 0.7502212984227693, 'Car': 0.6730542422843496, 'Chair': 0.8725531961226138, 'Earphone': 0.6870072048777072, 'Guitar': 0.8786945310789914, 'Knife': 0.8284336832073766, 'Lamp': 0.8026789801449955, 'Laptop': 0.9511344857405444, 'Motorbike': 0.37458383869284667, 'Mug': 0.8976668330787444, 'Pistol': 0.7663134051553214, 'Rocket': 0.5511253044913195, 'Skateboard': 0.7253365250344729, 'Table': 0.8076952000941581}
==================================================
Cmiou: 71.88677117909052 -> 74.38502038921175, Imiou: 80.26959477294649 -> 80.84875734591189
EPOCH 4 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:42<00:00,  3.41it/s, data_loading=0.006, iteration=0.135, train_Cmiou=71.53, train_Imiou=81.38, train_loss_seg=0.201]
Learning rate = 0.001000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:24<00:00,  9.96it/s, test_Cmiou=74.26, test_Imiou=81.53, test_loss_seg=0.183]
==================================================
    test_loss_seg = 0.18360912799835205
    test_Cmiou = 74.26034444971782
    test_Imiou = 81.53296936116313
    test_Imiou_per_class = {'Airplane': 0.7514842414283773, 'Bag': 0.6409829213673703, 'Cap': 0.7821761482191179, 'Car': 0.7253725420040257, 'Chair': 0.8874418708600028, 'Earphone': 0.6991693064276119, 'Guitar': 0.8800044434651371, 'Knife': 0.8362970408843464, 'Lamp': 0.8208426496845715, 'Laptop': 0.9516511458289447, 'Motorbike': 0.39183747394113033, 'Mug': 0.9254703575880555, 'Pistol': 0.6754748116718252, 'Rocket': 0.3956599418735582, 'Skateboard': 0.7062565565119785, 'Table': 0.8115336601987986}
==================================================
loss_seg: 0.26787325739860535 -> 0.18360912799835205, Imiou: 80.84875734591189 -> 81.53296936116313
EPOCH 5 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:39<00:00,  3.44it/s, data_loading=0.006, iteration=0.132, train_Cmiou=74.54, train_Imiou=80.89, train_loss_seg=0.211]
Learning rate = 0.001000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:24<00:00,  9.95it/s, test_Cmiou=75.99, test_Imiou=80.81, test_loss_seg=2.657]
==================================================
    test_loss_seg = 2.6575050354003906
    test_Cmiou = 75.99827049024069
    test_Imiou = 80.81525574609294
    test_Imiou_per_class = {'Airplane': 0.7763293515561746, 'Bag': 0.7762284597670567, 'Cap': 0.7173052283333056, 'Car': 0.7236475366578089, 'Chair': 0.881958540330812, 'Earphone': 0.7388659937480995, 'Guitar': 0.8889777264722523, 'Knife': 0.8616837235217852, 'Lamp': 0.7421000652707677, 'Laptop': 0.9501962007073005, 'Motorbike': 0.4488839218285674, 'Mug': 0.8306115696392516, 'Pistol': 0.7893753705044979, 'Rocket': 0.5193188016267178, 'Skateboard': 0.7188746287734852, 'Table': 0.7953661597006253}
==================================================
Cmiou: 74.38502038921175 -> 75.99827049024069
EPOCH 6 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:41<00:00,  3.42it/s, data_loading=0.006, iteration=0.134, train_Cmiou=78.51, train_Imiou=81.80, train_loss_seg=0.193]
Learning rate = 0.001000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.12it/s, test_Cmiou=78.24, test_Imiou=82.91, test_loss_seg=0.543]
==================================================
    test_loss_seg = 0.5429779887199402
    test_Cmiou = 78.24885796696306
    test_Imiou = 82.9124972269167
    test_Imiou_per_class = {'Airplane': 0.786495198009139, 'Bag': 0.7570709632091378, 'Cap': 0.7819285681093746, 'Car': 0.7183936212684267, 'Chair': 0.8854132563656214, 'Earphone': 0.7357636212300427, 'Guitar': 0.8987665451027259, 'Knife': 0.857413096670798, 'Lamp': 0.8182716917612152, 'Laptop': 0.9514762098052532, 'Motorbike': 0.5631626676477176, 'Mug': 0.9184284573342963, 'Pistol': 0.7787380200388226, 'Rocket': 0.517346652356757, 'Skateboard': 0.7289977792122357, 'Table': 0.8221509265925264}
==================================================
Cmiou: 75.99827049024069 -> 78.24885796696306, Imiou: 81.53296936116313 -> 82.9124972269167
EPOCH 7 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:38<00:00,  3.45it/s, data_loading=0.006, iteration=0.133, train_Cmiou=76.09, train_Imiou=82.03, train_loss_seg=0.190]
Learning rate = 0.001000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.12it/s, test_Cmiou=78.21, test_Imiou=82.35, test_loss_seg=0.120]
==================================================
    test_loss_seg = 0.12039124220609665
    test_Cmiou = 78.21156296514759
    test_Imiou = 82.3568613750875
    test_Imiou_per_class = {'Airplane': 0.7797232533051404, 'Bag': 0.7407793969824407, 'Cap': 0.7965692730027004, 'Car': 0.7492881277601655, 'Chair': 0.8902414541990832, 'Earphone': 0.7651565548544763, 'Guitar': 0.889803380650947, 'Knife': 0.8075506232246399, 'Lamp': 0.764575965019347, 'Laptop': 0.9479549211695949, 'Motorbike': 0.6098580692028543, 'Mug': 0.9165535050688336, 'Pistol': 0.7582048706115967, 'Rocket': 0.5349234317899472, 'Skateboard': 0.744428145314425, 'Table': 0.8182391022674242}
==================================================
loss_seg: 0.18360912799835205 -> 0.12039124220609665
EPOCH 8 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:38<00:00,  3.45it/s, data_loading=0.006, iteration=0.132, train_Cmiou=77.24, train_Imiou=83.24, train_loss_seg=0.181]
Learning rate = 0.001000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.12it/s, test_Cmiou=78.36, test_Imiou=83.18, test_loss_seg=0.426]
==================================================
    test_loss_seg = 0.4265753924846649
    test_Cmiou = 78.3621485930258
    test_Imiou = 83.18212531154978
    test_Imiou_per_class = {'Airplane': 0.7889784433994564, 'Bag': 0.7280971146701337, 'Cap': 0.8011821259429799, 'Car': 0.7409191906010333, 'Chair': 0.8935525882502223, 'Earphone': 0.7469496429188613, 'Guitar': 0.8961279765749962, 'Knife': 0.8496235901511952, 'Lamp': 0.8206871698408128, 'Laptop': 0.9515201799170039, 'Motorbike': 0.5366096189059334, 'Mug': 0.9212907928014193, 'Pistol': 0.7996045645545198, 'Rocket': 0.5307769631358669, 'Skateboard': 0.7113934879731946, 'Table': 0.8206303252464969}
==================================================
Cmiou: 78.24885796696306 -> 78.3621485930258, Imiou: 82.9124972269167 -> 83.18212531154978
EPOCH 9 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:38<00:00,  3.45it/s, data_loading=0.006, iteration=0.132, train_Cmiou=78.44, train_Imiou=82.88, train_loss_seg=0.179]
Learning rate = 0.001000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.15it/s, test_Cmiou=78.14, test_Imiou=83.61, test_loss_seg=0.114]
==================================================
    test_loss_seg = 0.11409256607294083
    test_Cmiou = 78.14567102029041
    test_Imiou = 83.61323591851735
    test_Imiou_per_class = {'Airplane': 0.7965467379130337, 'Bag': 0.7243093143282097, 'Cap': 0.6671315385627674, 'Car': 0.7424826673487234, 'Chair': 0.892319508743801, 'Earphone': 0.7636338251491451, 'Guitar': 0.8958823725300923, 'Knife': 0.8646175237581384, 'Lamp': 0.8322834504677235, 'Laptop': 0.95262569932145, 'Motorbike': 0.5947551480452967, 'Mug': 0.9049798927412476, 'Pistol': 0.7909445710627434, 'Rocket': 0.5101860261313231, 'Skateboard': 0.7447865735936754, 'Table': 0.8258225135490956}
==================================================
loss_seg: 0.12039124220609665 -> 0.11409256607294083, Imiou: 83.18212531154978 -> 83.61323591851735
EPOCH 10 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:37<00:00,  3.46it/s, data_loading=0.006, iteration=0.130, train_Cmiou=78.23, train_Imiou=82.83, train_loss_seg=0.175]
Learning rate = 0.001000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.14it/s, test_Cmiou=79.62, test_Imiou=83.61, test_loss_seg=0.376]
==================================================
    test_loss_seg = 0.3768831193447113
    test_Cmiou = 79.62705928648487
    test_Imiou = 83.61512871845541
    test_Imiou_per_class = {'Airplane': 0.8037846293340755, 'Bag': 0.8022576623545082, 'Cap': 0.7193409163141844, 'Car': 0.7498310034749296, 'Chair': 0.8900789325734144, 'Earphone': 0.7846210442714144, 'Guitar': 0.8847172585689239, 'Knife': 0.8518276988584231, 'Lamp': 0.8272541208260421, 'Laptop': 0.9470698776446872, 'Motorbike': 0.6591794498862853, 'Mug': 0.9355348017672703, 'Pistol': 0.7435836651476063, 'Rocket': 0.5614485249072299, 'Skateboard': 0.7570639777338524, 'Table': 0.8227359221747287}
==================================================
Cmiou: 78.3621485930258 -> 79.62705928648487, Imiou: 83.61323591851735 -> 83.61512871845541
EPOCH 11 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:37<00:00,  3.46it/s, data_loading=0.006, iteration=0.130, train_Cmiou=81.41, train_Imiou=83.92, train_loss_seg=0.174]
Learning rate = 0.001000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.14it/s, test_Cmiou=77.89, test_Imiou=83.35, test_loss_seg=0.179]
==================================================
    test_loss_seg = 0.17958565056324005
    test_Cmiou = 77.89891977758248
    test_Imiou = 83.35557425087717
    test_Imiou_per_class = {'Airplane': 0.7939147829566964, 'Bag': 0.7259618622103458, 'Cap': 0.7358026964665918, 'Car': 0.7069893215027807, 'Chair': 0.890769117843113, 'Earphone': 0.7427333116183322, 'Guitar': 0.9022920605225498, 'Knife': 0.8501091276046167, 'Lamp': 0.835305779740743, 'Laptop': 0.9521186552277612, 'Motorbike': 0.5708998478606424, 'Mug': 0.926004399400188, 'Pistol': 0.7880252172734185, 'Rocket': 0.4891996942033587, 'Skateboard': 0.7274538547200936, 'Table': 0.8262474352619658}
==================================================
EPOCH 12 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:38<00:00,  3.45it/s, data_loading=0.006, iteration=0.132, train_Cmiou=78.59, train_Imiou=83.28, train_loss_seg=0.168]
Learning rate = 0.001000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.15it/s, test_Cmiou=79.69, test_Imiou=83.69, test_loss_seg=0.119]
==================================================
    test_loss_seg = 0.11908375471830368
    test_Cmiou = 79.69552925222621
    test_Imiou = 83.69810362091756
    test_Imiou_per_class = {'Airplane': 0.8087143050218527, 'Bag': 0.753909075514775, 'Cap': 0.8176454142814644, 'Car': 0.7488680195224408, 'Chair': 0.8878817165554534, 'Earphone': 0.7600604163577055, 'Guitar': 0.9029211733638155, 'Knife': 0.7307670480588075, 'Lamp': 0.8414834295586219, 'Laptop': 0.9484973071223448, 'Motorbike': 0.6638914643152091, 'Mug': 0.9253253904107973, 'Pistol': 0.8014697077020526, 'Rocket': 0.591284681958504, 'Skateboard': 0.7427264118506718, 'Table': 0.8258391187616775}
==================================================
Cmiou: 79.62705928648487 -> 79.69552925222621, Imiou: 83.61512871845541 -> 83.69810362091756
EPOCH 13 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:37<00:00,  3.46it/s, data_loading=0.006, iteration=0.133, train_Cmiou=77.01, train_Imiou=84.15, train_loss_seg=0.159]
Learning rate = 0.001000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.15it/s, test_Cmiou=78.93, test_Imiou=83.62, test_loss_seg=1.539]
==================================================
    test_loss_seg = 1.5393120050430298
    test_Cmiou = 78.93332007633876
    test_Imiou = 83.62831870658638
    test_Imiou_per_class = {'Airplane': 0.8084913564279467, 'Bag': 0.6911929204422939, 'Cap': 0.7696524447540068, 'Car': 0.7481061814360007, 'Chair': 0.8939065368319224, 'Earphone': 0.7331740661532764, 'Guitar': 0.8836214550978481, 'Knife': 0.8709380248049854, 'Lamp': 0.8353999481616642, 'Laptop': 0.952522017893786, 'Motorbike': 0.6018332982449491, 'Mug': 0.9279313648725869, 'Pistol': 0.8056688652504956, 'Rocket': 0.5388942025059527, 'Skateboard': 0.7513149763455211, 'Table': 0.8166835529909637}
==================================================
EPOCH 14 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:49<00:00,  3.35it/s, data_loading=0.006, iteration=0.135, train_Cmiou=79.13, train_Imiou=83.37, train_loss_seg=0.171]
Learning rate = 0.001000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.06it/s, test_Cmiou=80.22, test_Imiou=83.91, test_loss_seg=0.295]
==================================================
    test_loss_seg = 0.2954549491405487
    test_Cmiou = 80.22990447260217
    test_Imiou = 83.91855145904098
    test_Imiou_per_class = {'Airplane': 0.8012859059118967, 'Bag': 0.7955366839754271, 'Cap': 0.8354432742108215, 'Car': 0.7578535590093275, 'Chair': 0.8950772688306533, 'Earphone': 0.7499732541298363, 'Guitar': 0.9000480626239846, 'Knife': 0.815373706154593, 'Lamp': 0.828750132255234, 'Laptop': 0.9459906139867795, 'Motorbike': 0.6060891803732622, 'Mug': 0.9318436609668541, 'Pistol': 0.7999163123627582, 'Rocket': 0.5937044396727758, 'Skateboard': 0.7520126559871194, 'Table': 0.8278860051650256}
==================================================
Cmiou: 79.69552925222621 -> 80.22990447260217, Imiou: 83.69810362091756 -> 83.91855145904098
EPOCH 15 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:37<00:00,  3.46it/s, data_loading=0.006, iteration=0.132, train_Cmiou=81.41, train_Imiou=84.57, train_loss_seg=0.15 ]
Learning rate = 0.000500
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.16it/s, test_Cmiou=81.12, test_Imiou=84.63, test_loss_seg=0.244]
==================================================
    test_loss_seg = 0.24409419298171997
    test_Cmiou = 81.12462360060302
    test_Imiou = 84.63663479350241
    test_Imiou_per_class = {'Airplane': 0.8180254957356968, 'Bag': 0.7953288864011787, 'Cap': 0.7934103152503394, 'Car': 0.7629001003165631, 'Chair': 0.8984597240784905, 'Earphone': 0.7586289253096344, 'Guitar': 0.9075626672947285, 'Knife': 0.8621869579825054, 'Lamp': 0.8457027112901959, 'Laptop': 0.9515286913065066, 'Motorbike': 0.6779115577419915, 'Mug': 0.9244603006202886, 'Pistol': 0.8111456283105553, 'Rocket': 0.5983855390620869, 'Skateboard': 0.748756691037802, 'Table': 0.8255455843579186}
==================================================
Cmiou: 80.22990447260217 -> 81.12462360060302, Imiou: 83.91855145904098 -> 84.63663479350241
EPOCH 16 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:37<00:00,  3.46it/s, data_loading=0.006, iteration=0.134, train_Cmiou=80.18, train_Imiou=84.95, train_loss_seg=0.151]
Learning rate = 0.000500
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.15it/s, test_Cmiou=81.03, test_Imiou=84.41, test_loss_seg=0.246]
==================================================
    test_loss_seg = 0.246812641620636
    test_Cmiou = 81.03889564443665
    test_Imiou = 84.41292209889264
    test_Imiou_per_class = {'Airplane': 0.801713142212039, 'Bag': 0.8632003111265174, 'Cap': 0.8187601987892907, 'Car': 0.7740617595543855, 'Chair': 0.9021615939393197, 'Earphone': 0.7351419708608992, 'Guitar': 0.9027901636871706, 'Knife': 0.8208805500730229, 'Lamp': 0.8408113377776257, 'Laptop': 0.9512863567956421, 'Motorbike': 0.6249248165655994, 'Mug': 0.9329285020937848, 'Pistol': 0.8152124859732722, 'Rocket': 0.6062447978222891, 'Skateboard': 0.7488499490974646, 'Table': 0.82725536674154}
==================================================
EPOCH 17 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:49<00:00,  3.34it/s, data_loading=0.005, iteration=0.360, train_Cmiou=80.76, train_Imiou=84.73, train_loss_seg=0.147]
Learning rate = 0.000500
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:25<00:00,  9.42it/s, test_Cmiou=81.52, test_Imiou=84.65, test_loss_seg=0.520]
==================================================
    test_loss_seg = 0.5203151106834412
    test_Cmiou = 81.52569012393795
    test_Imiou = 84.65452131646114
    test_Imiou_per_class = {'Airplane': 0.8199124718201742, 'Bag': 0.8214982670652088, 'Cap': 0.8241900922123911, 'Car': 0.772135429292009, 'Chair': 0.8976009221489408, 'Earphone': 0.7569344800035914, 'Guitar': 0.8986558041202389, 'Knife': 0.8563460061443866, 'Lamp': 0.8475917712714084, 'Laptop': 0.9530679985165784, 'Motorbike': 0.6892968355699574, 'Mug': 0.9283949328716289, 'Pistol': 0.8177333832633606, 'Rocket': 0.5787753628981419, 'Skateboard': 0.7582332677485698, 'Table': 0.8237433948834835}
==================================================
Cmiou: 81.12462360060302 -> 81.52569012393795, Imiou: 84.63663479350241 -> 84.65452131646114
EPOCH 18 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:39<00:00,  3.44it/s, data_loading=0.006, iteration=0.131, train_Cmiou=81.69, train_Imiou=85.68, train_loss_seg=0.138]
Learning rate = 0.000500
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.16it/s, test_Cmiou=81.55, test_Imiou=84.90, test_loss_seg=0.324]
==================================================
    test_loss_seg = 0.3242870271205902
    test_Cmiou = 81.55274699976489
    test_Imiou = 84.90127959663297
    test_Imiou_per_class = {'Airplane': 0.8191590619261371, 'Bag': 0.8027306620751807, 'Cap': 0.8310465526983961, 'Car': 0.7736352054520587, 'Chair': 0.9021862104081422, 'Earphone': 0.7274188174620096, 'Guitar': 0.9082631340913178, 'Knife': 0.8379035302150246, 'Lamp': 0.8391340340742076, 'Laptop': 0.954799813537434, 'Motorbike': 0.7053785571090592, 'Mug': 0.9373601679446255, 'Pistol': 0.8067955333682982, 'Rocket': 0.6177992894657252, 'Skateboard': 0.7543891069138643, 'Table': 0.8304398432209026}
==================================================
Cmiou: 81.52569012393795 -> 81.55274699976489, Imiou: 84.65452131646114 -> 84.90127959663297
EPOCH 19 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=82.87, train_Imiou=85.68, train_loss_seg=0.145]
Learning rate = 0.000500
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.16it/s, test_Cmiou=80.62, test_Imiou=84.70, test_loss_seg=0.252]
==================================================
    test_loss_seg = 0.2520694434642792
    test_Cmiou = 80.62645306031288
    test_Imiou = 84.70454955887809
    test_Imiou_per_class = {'Airplane': 0.82251276999973, 'Bag': 0.8295466913570223, 'Cap': 0.8074211146691804, 'Car': 0.76824051873163, 'Chair': 0.9017092392928461, 'Earphone': 0.7191874392464864, 'Guitar': 0.9079677017037555, 'Knife': 0.8504417001862776, 'Lamp': 0.8404673517690898, 'Laptop': 0.9512243265677787, 'Motorbike': 0.6658829841980075, 'Mug': 0.9361588391107638, 'Pistol': 0.8222072585004238, 'Rocket': 0.49578291971545174, 'Skateboard': 0.7555713680407491, 'Table': 0.82591026656087}
==================================================
EPOCH 20 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:48<00:00,  3.35it/s, data_loading=0.006, iteration=0.131, train_Cmiou=82.24, train_Imiou=85.14, train_loss_seg=0.142]
Learning rate = 0.000500
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.15it/s, test_Cmiou=81.41, test_Imiou=84.80, test_loss_seg=0.106]
==================================================
    test_loss_seg = 0.10631060600280762
    test_Cmiou = 81.41646482159814
    test_Imiou = 84.80702058353035
    test_Imiou_per_class = {'Airplane': 0.8169236238248554, 'Bag': 0.8190462503490394, 'Cap': 0.8201504821100809, 'Car': 0.7643175387367237, 'Chair': 0.9017009586079845, 'Earphone': 0.7258679421914123, 'Guitar': 0.9023168842245136, 'Knife': 0.8559856621211516, 'Lamp': 0.838865246677382, 'Laptop': 0.9517645613135591, 'Motorbike': 0.6840154296499834, 'Mug': 0.9343892658566947, 'Pistol': 0.8051269593621657, 'Rocket': 0.6215775001047001, 'Skateboard': 0.7531107959667613, 'Table': 0.831475270358696}
==================================================
loss_seg: 0.11409256607294083 -> 0.10631060600280762
EPOCH 21 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:37<00:00,  3.46it/s, data_loading=0.006, iteration=0.129, train_Cmiou=82.94, train_Imiou=85.64, train_loss_seg=0.134]
Learning rate = 0.000500
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.16it/s, test_Cmiou=81.06, test_Imiou=84.86, test_loss_seg=3.372]
==================================================
    test_loss_seg = 3.37199330329895
    test_Cmiou = 81.06493114825042
    test_Imiou = 84.86660942808463
    test_Imiou_per_class = {'Airplane': 0.8210149226994407, 'Bag': 0.7793822575116863, 'Cap': 0.8372754946764384, 'Car': 0.7750803734376505, 'Chair': 0.9040886277319987, 'Earphone': 0.7418449105340124, 'Guitar': 0.9075132412223345, 'Knife': 0.8016628136476387, 'Lamp': 0.848226331894293, 'Laptop': 0.9537725505595704, 'Motorbike': 0.6940234557006898, 'Mug': 0.931679454191615, 'Pistol': 0.8210191592681181, 'Rocket': 0.5681850480534116, 'Skateboard': 0.7575026316573491, 'Table': 0.8281177109338205}
==================================================
EPOCH 22 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:54<00:00,  3.30it/s, data_loading=0.006, iteration=0.142, train_Cmiou=80.86, train_Imiou=85.67, train_loss_seg=0.145]
Learning rate = 0.000500
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.03it/s, test_Cmiou=81.45, test_Imiou=84.68, test_loss_seg=0.164]
==================================================
    test_loss_seg = 0.16402961313724518
    test_Cmiou = 81.4517951406267
    test_Imiou = 84.68718974333045
    test_Imiou_per_class = {'Airplane': 0.8161181348224613, 'Bag': 0.8205034528360088, 'Cap': 0.8300398828499524, 'Car': 0.7731112609478349, 'Chair': 0.9005092816092944, 'Earphone': 0.7557930381535956, 'Guitar': 0.9007886110129848, 'Knife': 0.8496466727364815, 'Lamp': 0.8272106983662908, 'Laptop': 0.9539579989753718, 'Motorbike': 0.6939101286304499, 'Mug': 0.9319765473535121, 'Pistol': 0.7849565768422433, 'Rocket': 0.6093968640009856, 'Skateboard': 0.7525735052106012, 'Table': 0.8317945681522042}
==================================================
EPOCH 23 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:38<00:00,  3.45it/s, data_loading=0.006, iteration=0.132, train_Cmiou=82.47, train_Imiou=85.31, train_loss_seg=0.138]
Learning rate = 0.000500
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.16it/s, test_Cmiou=81.74, test_Imiou=84.74, test_loss_seg=0.241]
==================================================
    test_loss_seg = 0.2414415031671524
    test_Cmiou = 81.74535774435638
    test_Imiou = 84.74932390045373
    test_Imiou_per_class = {'Airplane': 0.8065867278952064, 'Bag': 0.8411455948990716, 'Cap': 0.8342588752115269, 'Car': 0.761299705308712, 'Chair': 0.9013748491366765, 'Earphone': 0.7539591137461847, 'Guitar': 0.9065585017207215, 'Knife': 0.837951404705424, 'Lamp': 0.850145406546514, 'Laptop': 0.9525918058692788, 'Motorbike': 0.6923560443052812, 'Mug': 0.9332980899627784, 'Pistol': 0.8245337595237665, 'Rocket': 0.595101968578932, 'Skateboard': 0.7588724275064169, 'Table': 0.8292229641805291}
==================================================
Cmiou: 81.55274699976489 -> 81.74535774435638
EPOCH 24 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:37<00:00,  3.47it/s, data_loading=0.006, iteration=0.134, train_Cmiou=83.45, train_Imiou=85.69, train_loss_seg=0.138]
Learning rate = 0.000500
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.16it/s, test_Cmiou=80.28, test_Imiou=84.59, test_loss_seg=0.164]
==================================================
    test_loss_seg = 0.16421888768672943
    test_Cmiou = 80.28762852820694
    test_Imiou = 84.59977015515243
    test_Imiou_per_class = {'Airplane': 0.8227829210473409, 'Bag': 0.7945347531353929, 'Cap': 0.8284817132461195, 'Car': 0.7531754461960676, 'Chair': 0.9016993787731026, 'Earphone': 0.7589071206752723, 'Guitar': 0.9091306855515366, 'Knife': 0.8242475355199389, 'Lamp': 0.8352806869568299, 'Laptop': 0.9477460667215378, 'Motorbike': 0.7047905391086572, 'Mug': 0.937308000235622, 'Pistol': 0.8116368378424272, 'Rocket': 0.4308026960539803, 'Skateboard': 0.7574300736724557, 'Table': 0.82806610977683}
==================================================
EPOCH 25 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.132, train_Cmiou=83.01, train_Imiou=85.67, train_loss_seg=0.136]
Learning rate = 0.000500
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.16it/s, test_Cmiou=80.38, test_Imiou=84.82, test_loss_seg=0.142]
==================================================
    test_loss_seg = 0.14251427352428436
    test_Cmiou = 80.38086888503997
    test_Imiou = 84.82596033289583
    test_Imiou_per_class = {'Airplane': 0.8182821823709368, 'Bag': 0.8624006123685607, 'Cap': 0.6624350892263049, 'Car': 0.769166015327608, 'Chair': 0.904106522683711, 'Earphone': 0.7247271329119711, 'Guitar': 0.9098088292115941, 'Knife': 0.8346803006508715, 'Lamp': 0.8422780496927316, 'Laptop': 0.9443256544663488, 'Motorbike': 0.7030246619361273, 'Mug': 0.9336738162318771, 'Pistol': 0.8133055716613165, 'Rocket': 0.5457313524140787, 'Skateboard': 0.7636387319930656, 'Table': 0.8293544984592904}
==================================================
EPOCH 26 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:40<00:00,  3.43it/s, data_loading=0.006, iteration=0.132, train_Cmiou=82.93, train_Imiou=85.57, train_loss_seg=0.137]
Learning rate = 0.000500
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.15it/s, test_Cmiou=80.66, test_Imiou=84.53, test_loss_seg=1.716]
==================================================
    test_loss_seg = 1.7164469957351685
    test_Cmiou = 80.66582721881734
    test_Imiou = 84.53151857047575
    test_Imiou_per_class = {'Airplane': 0.8097597166174304, 'Bag': 0.8124394240251451, 'Cap': 0.8644536227940539, 'Car': 0.7696071232392052, 'Chair': 0.9011034421091026, 'Earphone': 0.6626663040051403, 'Guitar': 0.9086065910015245, 'Knife': 0.8264444103015209, 'Lamp': 0.8424672776107792, 'Laptop': 0.9510937056602533, 'Motorbike': 0.7093503764307544, 'Mug': 0.9293923290712685, 'Pistol': 0.7867845149888162, 'Rocket': 0.5500977762824478, 'Skateboard': 0.7561477063755675, 'Table': 0.8261180344977642}
==================================================
EPOCH 27 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:37<00:00,  3.46it/s, data_loading=0.006, iteration=0.131, train_Cmiou=83.56, train_Imiou=85.84, train_loss_seg=0.133]
Learning rate = 0.000500
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.13it/s, test_Cmiou=81.97, test_Imiou=84.85, test_loss_seg=0.148]
==================================================
    test_loss_seg = 0.14824117720127106
    test_Cmiou = 81.9728507730735
    test_Imiou = 84.85913797345867
    test_Imiou_per_class = {'Airplane': 0.8199677796928738, 'Bag': 0.8251891941382109, 'Cap': 0.8467728452280882, 'Car': 0.7732311442241536, 'Chair': 0.9019476462038655, 'Earphone': 0.7488266533679369, 'Guitar': 0.9048766984809814, 'Knife': 0.8342971534644953, 'Lamp': 0.8441494694435537, 'Laptop': 0.9522864309377328, 'Motorbike': 0.702323334856813, 'Mug': 0.9347396238386579, 'Pistol': 0.8067301932813981, 'Rocket': 0.6286228201593531, 'Skateboard': 0.7643465575699347, 'Table': 0.8273485788037117}
==================================================
Cmiou: 81.74535774435638 -> 81.9728507730735
EPOCH 28 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:38<00:00,  3.45it/s, data_loading=0.006, iteration=0.132, train_Cmiou=84.03, train_Imiou=85.97, train_loss_seg=0.133]
Learning rate = 0.000500
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.14it/s, test_Cmiou=81.68, test_Imiou=84.85, test_loss_seg=0.233]
==================================================
    test_loss_seg = 0.23347754776477814
    test_Cmiou = 81.68711006218525
    test_Imiou = 84.85890134329176
    test_Imiou_per_class = {'Airplane': 0.8232339720691833, 'Bag': 0.8105322110678245, 'Cap': 0.8447219934215862, 'Car': 0.7718240042917669, 'Chair': 0.9034505735940347, 'Earphone': 0.757383036312956, 'Guitar': 0.9034120871095215, 'Knife': 0.835188290454186, 'Lamp': 0.8326072595410603, 'Laptop': 0.9502889532635976, 'Motorbike': 0.7044695995888535, 'Mug': 0.9344167208089713, 'Pistol': 0.8068394610825842, 'Rocket': 0.6132366503089146, 'Skateboard': 0.7481972204791664, 'Table': 0.8301355765554336}
==================================================
EPOCH 29 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=81.79, train_Imiou=86.39, train_loss_seg=0.126]
Learning rate = 0.000250
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.16it/s, test_Cmiou=82.12, test_Imiou=85.06, test_loss_seg=0.183]
==================================================
    test_loss_seg = 0.18340271711349487
    test_Cmiou = 82.12534276132405
    test_Imiou = 85.06652596505626
    test_Imiou_per_class = {'Airplane': 0.8181968448804698, 'Bag': 0.8377829519060542, 'Cap': 0.8646582716402151, 'Car': 0.7785576678631543, 'Chair': 0.9042519162181292, 'Earphone': 0.7498865273078371, 'Guitar': 0.9110073985688287, 'Knife': 0.838920151501795, 'Lamp': 0.8455467720843652, 'Laptop': 0.9541229319926708, 'Motorbike': 0.6982469309753787, 'Mug': 0.9361693514850593, 'Pistol': 0.829446158531794, 'Rocket': 0.5811715293263314, 'Skateboard': 0.7628716915034786, 'Table': 0.8292177460262871}
==================================================
Cmiou: 81.9728507730735 -> 82.12534276132405, Imiou: 84.90127959663297 -> 85.06652596505626
EPOCH 30 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.133, train_Cmiou=82.29, train_Imiou=86.23, train_loss_seg=0.122]
Learning rate = 0.000250
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=81.93, test_Imiou=85.01, test_loss_seg=0.136]
==================================================
    test_loss_seg = 0.13666513562202454
    test_Cmiou = 81.93604168019515
    test_Imiou = 85.01444670682645
    test_Imiou_per_class = {'Airplane': 0.8183684760430658, 'Bag': 0.8219496503624624, 'Cap': 0.8524042250906182, 'Car': 0.7761288327524044, 'Chair': 0.9035306850264159, 'Earphone': 0.7559016685216049, 'Guitar': 0.9109108746129099, 'Knife': 0.8221725468272281, 'Lamp': 0.8477281793448156, 'Laptop': 0.9533107905748996, 'Motorbike': 0.7098978512978164, 'Mug': 0.9364863250595531, 'Pistol': 0.8093673992902236, 'Rocket': 0.6052850572625542, 'Skateboard': 0.756404115093152, 'Table': 0.8299199916714995}
==================================================
EPOCH 31 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:54<00:00,  3.29it/s, data_loading=0.006, iteration=0.132, train_Cmiou=83.60, train_Imiou=86.23, train_loss_seg=0.125]
Learning rate = 0.000250
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.16it/s, test_Cmiou=82.10, test_Imiou=85.18, test_loss_seg=0.197]
==================================================
    test_loss_seg = 0.19716638326644897
    test_Cmiou = 82.10717626603528
    test_Imiou = 85.18960342396188
    test_Imiou_per_class = {'Airplane': 0.8247793412574551, 'Bag': 0.819976704384295, 'Cap': 0.8591871299557128, 'Car': 0.7771357319746545, 'Chair': 0.9049916301543083, 'Earphone': 0.76615468906727, 'Guitar': 0.9102510808992149, 'Knife': 0.8307105594478553, 'Lamp': 0.8484775525980444, 'Laptop': 0.9528356020982153, 'Motorbike': 0.6985597638214299, 'Mug': 0.9335534193260542, 'Pistol': 0.826380103779244, 'Rocket': 0.5738177914099213, 'Skateboard': 0.7800608134632374, 'Table': 0.8302762889287312}
==================================================
Imiou: 85.06652596505626 -> 85.18960342396188
EPOCH 32 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.132, train_Cmiou=85.30, train_Imiou=86.89, train_loss_seg=0.12 ]
Learning rate = 0.000250
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=81.72, test_Imiou=84.97, test_loss_seg=0.403]
==================================================
    test_loss_seg = 0.40336596965789795
    test_Cmiou = 81.72828454476465
    test_Imiou = 84.97845519287398
    test_Imiou_per_class = {'Airplane': 0.823198432648196, 'Bag': 0.8170507393880972, 'Cap': 0.8666634544442163, 'Car': 0.7788847830973966, 'Chair': 0.9030668650483306, 'Earphone': 0.754931798996123, 'Guitar': 0.9059398193263983, 'Knife': 0.8297050931249974, 'Lamp': 0.8403626373810258, 'Laptop': 0.9529277762615483, 'Motorbike': 0.70240265745866, 'Mug': 0.9351969093149608, 'Pistol': 0.7981401665563683, 'Rocket': 0.5794709159249636, 'Skateboard': 0.7578971479841548, 'Table': 0.8306863302069064}
==================================================
EPOCH 33 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:57<00:00,  3.27it/s, data_loading=0.006, iteration=0.132, train_Cmiou=84.40, train_Imiou=86.73, train_loss_seg=0.125]
Learning rate = 0.000250
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.12it/s, test_Cmiou=81.98, test_Imiou=84.98, test_loss_seg=0.248]
==================================================
    test_loss_seg = 0.24841059744358063
    test_Cmiou = 81.98512634355582
    test_Imiou = 84.98163435148734
    test_Imiou_per_class = {'Airplane': 0.8251345377382595, 'Bag': 0.8516226124136296, 'Cap': 0.8492051937242735, 'Car': 0.7775206492748361, 'Chair': 0.9034314891933367, 'Earphone': 0.7438383457567179, 'Guitar': 0.9089025374879787, 'Knife': 0.8131981952386245, 'Lamp': 0.841220667401948, 'Laptop': 0.9533039138078535, 'Motorbike': 0.7076064236294481, 'Mug': 0.9362931156350316, 'Pistol': 0.812775828544723, 'Rocket': 0.6026695176478108, 'Skateboard': 0.7620161494221411, 'Table': 0.8288810380523186}
==================================================
EPOCH 34 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=84.47, train_Imiou=86.15, train_loss_seg=0.127]
Learning rate = 0.000250
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.03, test_Imiou=85.06, test_loss_seg=0.161]
==================================================
    test_loss_seg = 0.16102126240730286
    test_Cmiou = 82.03951947792422
    test_Imiou = 85.0657051785214
    test_Imiou_per_class = {'Airplane': 0.8236145529994575, 'Bag': 0.8509105289103001, 'Cap': 0.8393329050219677, 'Car': 0.7792342425081389, 'Chair': 0.9046599317553159, 'Earphone': 0.7580353348751675, 'Guitar': 0.9093695931451502, 'Knife': 0.8436195002295142, 'Lamp': 0.8383758000129177, 'Laptop': 0.9536237393382005, 'Motorbike': 0.7170924418132019, 'Mug': 0.9381170922513219, 'Pistol': 0.7954719917711842, 'Rocket': 0.5841280331700537, 'Skateboard': 0.761325684384739, 'Table': 0.8294117442812442}
==================================================
EPOCH 35 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:48<00:00,  3.35it/s, data_loading=0.006, iteration=0.14 , train_Cmiou=84.94, train_Imiou=86.32, train_loss_seg=0.124]
Learning rate = 0.000250
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.06it/s, test_Cmiou=81.89, test_Imiou=85.00, test_loss_seg=0.073]
==================================================
    test_loss_seg = 0.07321659475564957
    test_Cmiou = 81.89040765596434
    test_Imiou = 85.00873125828522
    test_Imiou_per_class = {'Airplane': 0.8205993187629219, 'Bag': 0.847981002697627, 'Cap': 0.8430485160621347, 'Car': 0.7809964304464027, 'Chair': 0.905068789181107, 'Earphone': 0.7553425649438535, 'Guitar': 0.9038681373645654, 'Knife': 0.837776087102967, 'Lamp': 0.83841102790515, 'Laptop': 0.95330690720915, 'Motorbike': 0.7003005442255339, 'Mug': 0.9358635567510944, 'Pistol': 0.8138375698852134, 'Rocket': 0.5799814705006886, 'Skateboard': 0.755999902446561, 'Table': 0.8300833994693227}
==================================================
loss_seg: 0.10631060600280762 -> 0.07321659475564957
EPOCH 36 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:38<00:00,  3.45it/s, data_loading=0.006, iteration=0.130, train_Cmiou=84.58, train_Imiou=86.61, train_loss_seg=0.121]
Learning rate = 0.000250
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.16it/s, test_Cmiou=81.76, test_Imiou=85.13, test_loss_seg=0.208]
==================================================
    test_loss_seg = 0.2082234025001526
    test_Cmiou = 81.76138029467674
    test_Imiou = 85.13142836791715
    test_Imiou_per_class = {'Airplane': 0.8227898965649466, 'Bag': 0.8485881233941104, 'Cap': 0.8325403604855709, 'Car': 0.7772894199831996, 'Chair': 0.9064460840650616, 'Earphone': 0.6921795678619509, 'Guitar': 0.9087188426178205, 'Knife': 0.8341656245243749, 'Lamp': 0.8461556707152667, 'Laptop': 0.9538122625715864, 'Motorbike': 0.7175781506770537, 'Mug': 0.9347886362896223, 'Pistol': 0.8141552620446743, 'Rocket': 0.6061126903164281, 'Skateboard': 0.7570696525352757, 'Table': 0.8294306025013377}
==================================================
EPOCH 37 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.132, train_Cmiou=84.33, train_Imiou=86.43, train_loss_seg=0.122]
Learning rate = 0.000250
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.16it/s, test_Cmiou=82.16, test_Imiou=85.12, test_loss_seg=0.080]
==================================================
    test_loss_seg = 0.08080736547708511
    test_Cmiou = 82.1623866600803
    test_Imiou = 85.12253957328583
    test_Imiou_per_class = {'Airplane': 0.8185618059493257, 'Bag': 0.8400060433575975, 'Cap': 0.8382520810412276, 'Car': 0.7761994326873569, 'Chair': 0.904645168627015, 'Earphone': 0.7589472540949673, 'Guitar': 0.910264271718976, 'Knife': 0.8338491510603545, 'Lamp': 0.8400952546017002, 'Laptop': 0.9538767405943006, 'Motorbike': 0.7143192814337027, 'Mug': 0.9349819298808121, 'Pistol': 0.8163859532132087, 'Rocket': 0.6142860024469409, 'Skateboard': 0.7581233029852329, 'Table': 0.8331881919201294}
==================================================
Cmiou: 82.12534276132405 -> 82.1623866600803
EPOCH 38 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.134, train_Cmiou=85.39, train_Imiou=86.71, train_loss_seg=0.116]
Learning rate = 0.000250
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.13, test_Imiou=85.02, test_loss_seg=0.135]
==================================================
    test_loss_seg = 0.13505111634731293
    test_Cmiou = 82.13082672139885
    test_Imiou = 85.02903124118843
    test_Imiou_per_class = {'Airplane': 0.8208631004955532, 'Bag': 0.8493747621437409, 'Cap': 0.8414630944235792, 'Car': 0.7815434659312678, 'Chair': 0.905783049824814, 'Earphone': 0.7624064733380882, 'Guitar': 0.9112153770905912, 'Knife': 0.820976782061223, 'Lamp': 0.8436368771485049, 'Laptop': 0.9523079675755194, 'Motorbike': 0.6902181283925438, 'Mug': 0.9351922105762368, 'Pistol': 0.8217306194841677, 'Rocket': 0.6058728031965, 'Skateboard': 0.7706279126319983, 'Table': 0.8277196511094875}
==================================================
EPOCH 39 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.132, train_Cmiou=84.80, train_Imiou=86.66, train_loss_seg=0.118]
Learning rate = 0.000250
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=81.98, test_Imiou=85.08, test_loss_seg=0.071]
==================================================
    test_loss_seg = 0.071554996073246
    test_Cmiou = 81.9810687521441
    test_Imiou = 85.08168080592581
    test_Imiou_per_class = {'Airplane': 0.8122543476135131, 'Bag': 0.8458496107550068, 'Cap': 0.8425592403433271, 'Car': 0.7824509593417582, 'Chair': 0.9038929680757598, 'Earphone': 0.7600032886360953, 'Guitar': 0.9118769927844025, 'Knife': 0.832160559278412, 'Lamp': 0.8492425615648793, 'Laptop': 0.9543858040414142, 'Motorbike': 0.7027842914657167, 'Mug': 0.9205308695997773, 'Pistol': 0.8229403703431811, 'Rocket': 0.5871055014720928, 'Skateboard': 0.7571608778143017, 'Table': 0.8317727572134174}
==================================================
loss_seg: 0.07321659475564957 -> 0.071554996073246
EPOCH 40 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.132, train_Cmiou=83.71, train_Imiou=86.35, train_loss_seg=0.120]
Learning rate = 0.000250
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.16it/s, test_Cmiou=82.54, test_Imiou=85.11, test_loss_seg=0.170]
==================================================
    test_loss_seg = 0.17091111838817596
    test_Cmiou = 82.54043226503444
    test_Imiou = 85.10998151443786
    test_Imiou_per_class = {'Airplane': 0.8210442250269581, 'Bag': 0.8431229287414572, 'Cap': 0.8550897816924674, 'Car': 0.7839122981287232, 'Chair': 0.9030064449407746, 'Earphone': 0.7634198972899283, 'Guitar': 0.9119751179251371, 'Knife': 0.8346864170770312, 'Lamp': 0.8423750302663572, 'Laptop': 0.9528285292713342, 'Motorbike': 0.7152608438019269, 'Mug': 0.9405351309456697, 'Pistol': 0.8232917627396173, 'Rocket': 0.6178805798653135, 'Skateboard': 0.7688697354775696, 'Table': 0.8291704392152452}
==================================================
Cmiou: 82.1623866600803 -> 82.54043226503444
EPOCH 41 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:54<00:00,  3.29it/s, data_loading=0.006, iteration=0.131, train_Cmiou=84.32, train_Imiou=86.56, train_loss_seg=0.116]
Learning rate = 0.000250
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.18, test_Imiou=85.14, test_loss_seg=5.551]
==================================================
    test_loss_seg = 5.551235198974609
    test_Cmiou = 82.1800768594926
    test_Imiou = 85.14962678667084
    test_Imiou_per_class = {'Airplane': 0.823148690231068, 'Bag': 0.8424739515861053, 'Cap': 0.8375201689353602, 'Car': 0.7801370666742162, 'Chair': 0.902408458789382, 'Earphone': 0.7565244488578413, 'Guitar': 0.9115204607936775, 'Knife': 0.842081624608394, 'Lamp': 0.8396902170272479, 'Laptop': 0.9539574358679748, 'Motorbike': 0.703548285270217, 'Mug': 0.9390142999544654, 'Pistol': 0.81199014365022, 'Rocket': 0.5985620721541904, 'Skateboard': 0.7733642194936238, 'Table': 0.8328707536248303}
==================================================
EPOCH 42 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.130, train_Cmiou=84.68, train_Imiou=86.85, train_loss_seg=0.120]
Learning rate = 0.000250
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.06, test_Imiou=84.97, test_loss_seg=0.637]
==================================================
    test_loss_seg = 0.637665331363678
    test_Cmiou = 82.06282151388848
    test_Imiou = 84.97374531901059
    test_Imiou_per_class = {'Airplane': 0.8243768481699044, 'Bag': 0.8426317513894028, 'Cap': 0.8636980450124359, 'Car': 0.7816550469699467, 'Chair': 0.9016161209763364, 'Earphone': 0.7535023672581452, 'Guitar': 0.9123025870097206, 'Knife': 0.8377068243742165, 'Lamp': 0.8384515716406048, 'Laptop': 0.9530337914824356, 'Motorbike': 0.6913227073552874, 'Mug': 0.9396203559301288, 'Pistol': 0.8101870649164629, 'Rocket': 0.5931542594960163, 'Skateboard': 0.7580823666393027, 'Table': 0.82870973360181}
==================================================
EPOCH 43 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:55<00:00,  3.29it/s, data_loading=0.006, iteration=0.132, train_Cmiou=84.09, train_Imiou=85.81, train_loss_seg=0.120]
Learning rate = 0.000125
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.15it/s, test_Cmiou=82.36, test_Imiou=85.25, test_loss_seg=0.126]
==================================================
    test_loss_seg = 0.12619845569133759
    test_Cmiou = 82.36413788764033
    test_Imiou = 85.25959281022749
    test_Imiou_per_class = {'Airplane': 0.8276812015018369, 'Bag': 0.8300272190693331, 'Cap': 0.8610504513963018, 'Car': 0.7800997254468185, 'Chair': 0.9057435202873051, 'Earphone': 0.7516309237578718, 'Guitar': 0.9124385161155064, 'Knife': 0.8353431151908366, 'Lamp': 0.8419442502196651, 'Laptop': 0.9539753011315014, 'Motorbike': 0.7211736977534381, 'Mug': 0.940527641373321, 'Pistol': 0.8104011656627231, 'Rocket': 0.6157980782019976, 'Skateboard': 0.7595143015919501, 'Table': 0.8309129533220453}
==================================================
Imiou: 85.18960342396188 -> 85.25959281022749
EPOCH 44 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=84.95, train_Imiou=87.51, train_loss_seg=0.116]
Learning rate = 0.000125
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.19it/s, test_Cmiou=82.64, test_Imiou=85.27, test_loss_seg=0.162]
==================================================
    test_loss_seg = 0.162679985165596
    test_Cmiou = 82.64493595689592
    test_Imiou = 85.27760985360428
    test_Imiou_per_class = {'Airplane': 0.8269475951239537, 'Bag': 0.845311691194821, 'Cap': 0.8650740713031646, 'Car': 0.7812131337421516, 'Chair': 0.9053091115027251, 'Earphone': 0.7672628801644051, 'Guitar': 0.9133320414388001, 'Knife': 0.8353014197111971, 'Lamp': 0.8363567917163747, 'Laptop': 0.9541709851017998, 'Motorbike': 0.7132141679984769, 'Mug': 0.9425096100364724, 'Pistol': 0.8195363737774436, 'Rocket': 0.6276261208426223, 'Skateboard': 0.7570752303250291, 'Table': 0.8329485291239096}
==================================================
Cmiou: 82.54043226503444 -> 82.64493595689592, Imiou: 85.25959281022749 -> 85.27760985360428
EPOCH 45 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:35<00:00,  3.48it/s, data_loading=0.006, iteration=0.130, train_Cmiou=84.07, train_Imiou=87.19, train_loss_seg=0.111]
Learning rate = 0.000125
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.19it/s, test_Cmiou=82.52, test_Imiou=85.21, test_loss_seg=0.101]
==================================================
    test_loss_seg = 0.10165390372276306
    test_Cmiou = 82.52356808557289
    test_Imiou = 85.21555739598313
    test_Imiou_per_class = {'Airplane': 0.8247102351954111, 'Bag': 0.8450456728102148, 'Cap': 0.8588548651932754, 'Car': 0.7836878848811009, 'Chair': 0.9046634886417153, 'Earphone': 0.7627830455036183, 'Guitar': 0.9125108612866619, 'Knife': 0.8365844368526341, 'Lamp': 0.837235966122656, 'Laptop': 0.9543541002596324, 'Motorbike': 0.715438229182684, 'Mug': 0.9414212104485439, 'Pistol': 0.8142055916644754, 'Rocket': 0.6121139841727821, 'Skateboard': 0.7684691391622244, 'Table': 0.8316921823140315}
==================================================
EPOCH 46 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:35<00:00,  3.48it/s, data_loading=0.006, iteration=0.130, train_Cmiou=86.47, train_Imiou=87.69, train_loss_seg=0.111]
Learning rate = 0.000125
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.43, test_Imiou=85.19, test_loss_seg=0.161]
==================================================
    test_loss_seg = 0.1614244133234024
    test_Cmiou = 82.43155416219783
    test_Imiou = 85.19369532950064
    test_Imiou_per_class = {'Airplane': 0.8248818555509242, 'Bag': 0.8456148693494543, 'Cap': 0.8655530858329491, 'Car': 0.780494708887844, 'Chair': 0.9056165863139894, 'Earphone': 0.7642468006905551, 'Guitar': 0.9130370862286461, 'Knife': 0.8399035945800808, 'Lamp': 0.8387716945532692, 'Laptop': 0.9536920977289508, 'Motorbike': 0.7102945303401067, 'Mug': 0.9348899825941919, 'Pistol': 0.8136519011574698, 'Rocket': 0.597819033083916, 'Skateboard': 0.7701071408246234, 'Table': 0.8304736982346819}
==================================================
EPOCH 47 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.130, train_Cmiou=85.68, train_Imiou=87.23, train_loss_seg=0.111]
Learning rate = 0.000125
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.30, test_Imiou=85.08, test_loss_seg=0.048]
==================================================
    test_loss_seg = 0.048263195902109146
    test_Cmiou = 82.30096758900982
    test_Imiou = 85.08604095531621
    test_Imiou_per_class = {'Airplane': 0.8228261853797931, 'Bag': 0.8421310298793402, 'Cap': 0.8639239351432786, 'Car': 0.7783147793641617, 'Chair': 0.9045455409962474, 'Earphone': 0.7541021176646762, 'Guitar': 0.912542816299337, 'Knife': 0.8321926341095403, 'Lamp': 0.8365054947796016, 'Laptop': 0.9548181550653159, 'Motorbike': 0.7041920072520442, 'Mug': 0.9392349194134993, 'Pistol': 0.8046539859110046, 'Rocket': 0.6164765492819693, 'Skateboard': 0.7706727301195627, 'Table': 0.8310219335821974}
==================================================
loss_seg: 0.071554996073246 -> 0.048263195902109146
EPOCH 48 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=84.67, train_Imiou=86.90, train_loss_seg=0.116]
Learning rate = 0.000125
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.53, test_Imiou=85.23, test_loss_seg=0.160]
==================================================
    test_loss_seg = 0.16045145690441132
    test_Cmiou = 82.53630596304734
    test_Imiou = 85.230172342764
    test_Imiou_per_class = {'Airplane': 0.8230240145579925, 'Bag': 0.8487441862690731, 'Cap': 0.8657298492492841, 'Car': 0.7824151029527522, 'Chair': 0.904830657469504, 'Earphone': 0.7652588085060394, 'Guitar': 0.9125412742517963, 'Knife': 0.8419221826044812, 'Lamp': 0.8388643020950115, 'Laptop': 0.9543653356792525, 'Motorbike': 0.7114332050450111, 'Mug': 0.940084214284794, 'Pistol': 0.8214759212466533, 'Rocket': 0.5930244508719169, 'Skateboard': 0.7702537258294723, 'Table': 0.8318417231745405}
==================================================
EPOCH 49 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:35<00:00,  3.48it/s, data_loading=0.006, iteration=0.131, train_Cmiou=84.92, train_Imiou=87.12, train_loss_seg=0.112]
Learning rate = 0.000125
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.53, test_Imiou=85.25, test_loss_seg=0.147]
==================================================
    test_loss_seg = 0.1477843075990677
    test_Cmiou = 82.53501161642123
    test_Imiou = 85.25364462845859
    test_Imiou_per_class = {'Airplane': 0.8254727649058997, 'Bag': 0.8435894331769196, 'Cap': 0.8616647026797012, 'Car': 0.7843231569397856, 'Chair': 0.9042148910723499, 'Earphone': 0.7561257331928202, 'Guitar': 0.9136652803680992, 'Knife': 0.840438366257905, 'Lamp': 0.8376190367718969, 'Laptop': 0.9544502565980656, 'Motorbike': 0.7107524038457899, 'Mug': 0.937004390243141, 'Pistol': 0.8156307006776377, 'Rocket': 0.6293074876724837, 'Skateboard': 0.7585072910215722, 'Table': 0.8328359632033289}
==================================================
EPOCH 50 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:35<00:00,  3.48it/s, data_loading=0.006, iteration=0.132, train_Cmiou=85.17, train_Imiou=87.72, train_loss_seg=0.107]
Learning rate = 0.000125
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.26, test_Imiou=84.84, test_loss_seg=0.834]
==================================================
    test_loss_seg = 0.8342617154121399
    test_Cmiou = 82.26705540959274
    test_Imiou = 84.84432457760656
    test_Imiou_per_class = {'Airplane': 0.8238082800475283, 'Bag': 0.8473452230984623, 'Cap': 0.8623071626418686, 'Car': 0.7771567772822713, 'Chair': 0.9047509290665823, 'Earphone': 0.752602723203809, 'Guitar': 0.9117119870884245, 'Knife': 0.803026988694868, 'Lamp': 0.8392573460922411, 'Laptop': 0.9531578876661705, 'Motorbike': 0.711418103993145, 'Mug': 0.9402374732828037, 'Pistol': 0.820891849742832, 'Rocket': 0.6232567596207611, 'Skateboard': 0.768558653247855, 'Table': 0.8232407207652168}
==================================================
EPOCH 51 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=86.32, train_Imiou=87.10, train_loss_seg=0.112]
Learning rate = 0.000125
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.56, test_Imiou=85.09, test_loss_seg=0.112]
==================================================
    test_loss_seg = 0.1127411425113678
    test_Cmiou = 82.569356756156
    test_Imiou = 85.09045484110958
    test_Imiou_per_class = {'Airplane': 0.8207337716063577, 'Bag': 0.8418766827411363, 'Cap': 0.8697097578179335, 'Car': 0.7820315300986355, 'Chair': 0.9055519951380065, 'Earphone': 0.7616132822035361, 'Guitar': 0.9103532403169509, 'Knife': 0.8390426703881626, 'Lamp': 0.8370375275716607, 'Laptop': 0.9541049020106422, 'Motorbike': 0.708174024659518, 'Mug': 0.9377563352368155, 'Pistol': 0.8035750246853219, 'Rocket': 0.6374743717966975, 'Skateboard': 0.7726021681487626, 'Table': 0.8294597965648237}
==================================================
EPOCH 52 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.133, train_Cmiou=85.26, train_Imiou=87.30, train_loss_seg=0.113]
Learning rate = 0.000125
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.45, test_Imiou=85.12, test_loss_seg=0.234]
==================================================
    test_loss_seg = 0.2345266342163086
    test_Cmiou = 82.45846173875232
    test_Imiou = 85.1233579676652
    test_Imiou_per_class = {'Airplane': 0.8261006466035566, 'Bag': 0.8471064582479207, 'Cap': 0.8598101564534038, 'Car': 0.7826311734583656, 'Chair': 0.9029271826739182, 'Earphone': 0.763189599631006, 'Guitar': 0.9125247194519808, 'Knife': 0.8333412033154713, 'Lamp': 0.838101463190862, 'Laptop': 0.9542651880028612, 'Motorbike': 0.7041384473917217, 'Mug': 0.9385538266686348, 'Pistol': 0.8120484564433522, 'Rocket': 0.6174246123008748, 'Skateboard': 0.7708180803677358, 'Table': 0.8303726639987057}
==================================================
EPOCH 53 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:39<00:00,  3.44it/s, data_loading=0.006, iteration=0.132, train_Cmiou=86.95, train_Imiou=87.14, train_loss_seg=0.115]
Learning rate = 0.000125
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.58, test_Imiou=85.14, test_loss_seg=0.042]
==================================================
    test_loss_seg = 0.04246433079242706
    test_Cmiou = 82.5801561149453
    test_Imiou = 85.14826070324307
    test_Imiou_per_class = {'Airplane': 0.8248227498147556, 'Bag': 0.8475490218514172, 'Cap': 0.872431296445079, 'Car': 0.7798639322941646, 'Chair': 0.90494076865624, 'Earphone': 0.7603444196953207, 'Guitar': 0.912694111347077, 'Knife': 0.8430382313736262, 'Lamp': 0.837746346571032, 'Laptop': 0.9543778300104985, 'Motorbike': 0.7113918914710661, 'Mug': 0.940782957091326, 'Pistol': 0.8100853394480351, 'Rocket': 0.6091131707635179, 'Skateboard': 0.7744808788244056, 'Table': 0.8291620327336844}
==================================================
loss_seg: 0.048263195902109146 -> 0.04246433079242706
EPOCH 54 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.132, train_Cmiou=85.72, train_Imiou=87.45, train_loss_seg=0.108]
Learning rate = 0.000125
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.58, test_Imiou=84.95, test_loss_seg=0.078]
==================================================
    test_loss_seg = 0.07859033346176147
    test_Cmiou = 82.58082787612588
    test_Imiou = 84.95796955564524
    test_Imiou_per_class = {'Airplane': 0.8207852452008159, 'Bag': 0.8376088447491757, 'Cap': 0.8689341267231612, 'Car': 0.7819205275896863, 'Chair': 0.9036590942339177, 'Earphone': 0.7661542666601951, 'Guitar': 0.9131816961549476, 'Knife': 0.8415011179090879, 'Lamp': 0.833790219068121, 'Laptop': 0.9534081472715126, 'Motorbike': 0.71083258303592, 'Mug': 0.9408688670245331, 'Pistol': 0.8199920223180525, 'Rocket': 0.6240355760025299, 'Skateboard': 0.7701849790234713, 'Table': 0.8260751472150155}
==================================================
EPOCH 55 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.130, train_Cmiou=85.03, train_Imiou=86.84, train_loss_seg=0.115]
Learning rate = 0.000125
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.48, test_Imiou=85.15, test_loss_seg=0.195]
==================================================
    test_loss_seg = 0.19502870738506317
    test_Cmiou = 82.48612100145785
    test_Imiou = 85.15519367004927
    test_Imiou_per_class = {'Airplane': 0.8268755743610414, 'Bag': 0.8445790531281839, 'Cap': 0.8673201949620739, 'Car': 0.7789737321931949, 'Chair': 0.9036520699738859, 'Earphone': 0.7652682248746233, 'Guitar': 0.9123182024639879, 'Knife': 0.8323842872767131, 'Lamp': 0.8406160920509824, 'Laptop': 0.9540243851112498, 'Motorbike': 0.7079483984406795, 'Mug': 0.9380336715092037, 'Pistol': 0.8170989588710204, 'Rocket': 0.62123989790104, 'Skateboard': 0.7570303198851758, 'Table': 0.8304162972302005}
==================================================
EPOCH 56 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:52<00:00,  3.31it/s, data_loading=0.006, iteration=0.145, train_Cmiou=85.10, train_Imiou=86.88, train_loss_seg=0.116]
Learning rate = 0.000125
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:24<00:00, 10.00it/s, test_Cmiou=82.23, test_Imiou=85.03, test_loss_seg=0.458]
==================================================
    test_loss_seg = 0.45887503027915955
    test_Cmiou = 82.23739340843235
    test_Imiou = 85.03281422288293
    test_Imiou_per_class = {'Airplane': 0.8269324097698488, 'Bag': 0.8500618575400327, 'Cap': 0.8656164854450036, 'Car': 0.7819963152516711, 'Chair': 0.9039874961585884, 'Earphone': 0.7675269567995772, 'Guitar': 0.9113657947248368, 'Knife': 0.828993678613385, 'Lamp': 0.8380181806455695, 'Laptop': 0.9550194208337126, 'Motorbike': 0.7109433048071855, 'Mug': 0.9399147846385588, 'Pistol': 0.8160439346589665, 'Rocket': 0.5758376359780107, 'Skateboard': 0.7587801344637224, 'Table': 0.8269445550205053}
==================================================
EPOCH 57 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:37<00:00,  3.46it/s, data_loading=0.006, iteration=0.130, train_Cmiou=85.39, train_Imiou=86.80, train_loss_seg=0.109]
Learning rate = 0.000125
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.53, test_Imiou=85.06, test_loss_seg=0.096]
==================================================
    test_loss_seg = 0.09691200405359268
    test_Cmiou = 82.53847681247035
    test_Imiou = 85.06523141262146
    test_Imiou_per_class = {'Airplane': 0.825468480777825, 'Bag': 0.8481777706001787, 'Cap': 0.8735962285462272, 'Car': 0.7822874736403007, 'Chair': 0.9038958382993456, 'Earphone': 0.7684152288524068, 'Guitar': 0.9123835490194436, 'Knife': 0.8361537026352213, 'Lamp': 0.8342955319347284, 'Laptop': 0.9544551152923629, 'Motorbike': 0.7046672285706502, 'Mug': 0.9412833505199032, 'Pistol': 0.8225281288319938, 'Rocket': 0.6134448776660448, 'Skateboard': 0.7565641512281883, 'Table': 0.8285396335804363}
==================================================
EPOCH 58 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.130, train_Cmiou=85.67, train_Imiou=87.68, train_loss_seg=0.106]
Learning rate = 0.000063
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.70, test_Imiou=85.12, test_loss_seg=0.295]
==================================================
    test_loss_seg = 0.29586726427078247
    test_Cmiou = 82.70750182461093
    test_Imiou = 85.12714920766857
    test_Imiou_per_class = {'Airplane': 0.8274670066993983, 'Bag': 0.8460735334614952, 'Cap': 0.8730895532296974, 'Car': 0.7848469049366611, 'Chair': 0.9055662007063241, 'Earphone': 0.7727800290944159, 'Guitar': 0.911263115931843, 'Knife': 0.8321923280155004, 'Lamp': 0.8352484454435718, 'Laptop': 0.9548260114744519, 'Motorbike': 0.7155025725930507, 'Mug': 0.9401620424798585, 'Pistol': 0.8055940972484973, 'Rocket': 0.6304405424984665, 'Skateboard': 0.7704526438351091, 'Table': 0.8276952642894079}
==================================================
Cmiou: 82.64493595689592 -> 82.70750182461093
EPOCH 59 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:58<00:00,  3.26it/s, data_loading=0.006, iteration=0.131, train_Cmiou=86.40, train_Imiou=87.54, train_loss_seg=0.105]
Learning rate = 0.000063
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.39, test_Imiou=85.11, test_loss_seg=0.086]
==================================================
    test_loss_seg = 0.0867200568318367
    test_Cmiou = 82.39362602158235
    test_Imiou = 85.1178248801256
    test_Imiou_per_class = {'Airplane': 0.8270082768992354, 'Bag': 0.8479755981540691, 'Cap': 0.869413510636576, 'Car': 0.7796827939922274, 'Chair': 0.9038827490105592, 'Earphone': 0.7754018780427813, 'Guitar': 0.9114471648383506, 'Knife': 0.8339723078390439, 'Lamp': 0.8414684285436868, 'Laptop': 0.954670587169825, 'Motorbike': 0.7152441040788752, 'Mug': 0.9406780891462337, 'Pistol': 0.8109414997019214, 'Rocket': 0.5761706208783236, 'Skateboard': 0.7667944123721097, 'Table': 0.8282281421493586}
==================================================
EPOCH 60 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.134, train_Cmiou=85.75, train_Imiou=86.54, train_loss_seg=0.112]
Learning rate = 0.000063
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.47, test_Imiou=84.98, test_loss_seg=0.452]
==================================================
    test_loss_seg = 0.45247364044189453
    test_Cmiou = 82.47344890206038
    test_Imiou = 84.98537617206765
    test_Imiou_per_class = {'Airplane': 0.8238958254740856, 'Bag': 0.8449547918321721, 'Cap': 0.8745177961536972, 'Car': 0.7825583953173448, 'Chair': 0.9056150783333855, 'Earphone': 0.772312392942772, 'Guitar': 0.913415073321381, 'Knife': 0.8337690520603876, 'Lamp': 0.8375127992762949, 'Laptop': 0.9548052169042042, 'Motorbike': 0.7110981130904338, 'Mug': 0.9425138411400326, 'Pistol': 0.821470522119992, 'Rocket': 0.5941530679936134, 'Skateboard': 0.759504390562551, 'Table': 0.8236554678073131}
==================================================
EPOCH 61 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:35<00:00,  3.48it/s, data_loading=0.006, iteration=0.132, train_Cmiou=85.75, train_Imiou=87.32, train_loss_seg=0.109]
Learning rate = 0.000063
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.78, test_Imiou=85.11, test_loss_seg=0.133]
==================================================
    test_loss_seg = 0.13344667851924896
    test_Cmiou = 82.78002739702927
    test_Imiou = 85.1133523225192
    test_Imiou_per_class = {'Airplane': 0.8232188299718632, 'Bag': 0.8436591729871242, 'Cap': 0.8671774859245496, 'Car': 0.7823451702693675, 'Chair': 0.9051276731324536, 'Earphone': 0.7727810338300648, 'Guitar': 0.9131914973015712, 'Knife': 0.8384633199709256, 'Lamp': 0.8333939551690774, 'Laptop': 0.9543396165115755, 'Motorbike': 0.7154859680751051, 'Mug': 0.9407750152270188, 'Pistol': 0.8168719519525588, 'Rocket': 0.6309439887947459, 'Skateboard': 0.7783340273633256, 'Table': 0.8286956770433583}
==================================================
Cmiou: 82.70750182461093 -> 82.78002739702927
EPOCH 62 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:55<00:00,  3.29it/s, data_loading=0.006, iteration=0.131, train_Cmiou=85.05, train_Imiou=87.05, train_loss_seg=0.112]
Learning rate = 0.000063
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.16it/s, test_Cmiou=82.64, test_Imiou=85.12, test_loss_seg=0.240]
==================================================
    test_loss_seg = 0.24039560556411743
    test_Cmiou = 82.64398327415414
    test_Imiou = 85.12059281873935
    test_Imiou_per_class = {'Airplane': 0.8248124676499672, 'Bag': 0.8466828678248076, 'Cap': 0.8643899222591386, 'Car': 0.7825616399672509, 'Chair': 0.9044200301612755, 'Earphone': 0.7603397677132678, 'Guitar': 0.9123895262853672, 'Knife': 0.839614141519316, 'Lamp': 0.8364191498254285, 'Laptop': 0.9543224352603756, 'Motorbike': 0.7179690709416061, 'Mug': 0.9407985850092166, 'Pistol': 0.8217414112788951, 'Rocket': 0.6181121720353259, 'Skateboard': 0.770330831917541, 'Table': 0.8281333042158839}
==================================================
EPOCH 63 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.130, train_Cmiou=86.95, train_Imiou=87.39, train_loss_seg=0.108]
Learning rate = 0.000063
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.58, test_Imiou=85.01, test_loss_seg=0.089]
==================================================
    test_loss_seg = 0.08992281556129456
    test_Cmiou = 82.58960137298311
    test_Imiou = 85.01671727136589
    test_Imiou_per_class = {'Airplane': 0.828818127818797, 'Bag': 0.8522062299484382, 'Cap': 0.8655506294936072, 'Car': 0.7810083406183701, 'Chair': 0.9023940921901428, 'Earphone': 0.7647884777042727, 'Guitar': 0.9135229106911235, 'Knife': 0.8326229739951799, 'Lamp': 0.8386818417993923, 'Laptop': 0.9554001033834169, 'Motorbike': 0.7190919956814515, 'Mug': 0.9405200275924585, 'Pistol': 0.8128589269039, 'Rocket': 0.6057154624199874, 'Skateboard': 0.7764255218319339, 'Table': 0.824730557604827}
==================================================
EPOCH 64 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.48it/s, data_loading=0.006, iteration=0.132, train_Cmiou=85.96, train_Imiou=87.54, train_loss_seg=0.107]
Learning rate = 0.000063
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.69, test_Imiou=85.10, test_loss_seg=0.070]
==================================================
    test_loss_seg = 0.07091207057237625
    test_Cmiou = 82.69729293641873
    test_Imiou = 85.1052857402083
    test_Imiou_per_class = {'Airplane': 0.8261941298091113, 'Bag': 0.8463004737014447, 'Cap': 0.8628275245667184, 'Car': 0.7831585129893975, 'Chair': 0.9047103538044753, 'Earphone': 0.7664798496285581, 'Guitar': 0.9120544654748601, 'Knife': 0.8365901878710353, 'Lamp': 0.8350976619797179, 'Laptop': 0.9551118455342194, 'Motorbike': 0.7157348493311629, 'Mug': 0.9396947198775045, 'Pistol': 0.8175930512738688, 'Rocket': 0.629557973584186, 'Skateboard': 0.7729722900547066, 'Table': 0.827488980346031}
==================================================
EPOCH 65 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.130, train_Cmiou=84.86, train_Imiou=87.05, train_loss_seg=0.107]
Learning rate = 0.000063
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.63, test_Imiou=84.96, test_loss_seg=0.090]
==================================================
    test_loss_seg = 0.09053408354520798
    test_Cmiou = 82.63967126568727
    test_Imiou = 84.96929271246388
    test_Imiou_per_class = {'Airplane': 0.8241217161381905, 'Bag': 0.85064469148448, 'Cap': 0.8674653842199628, 'Car': 0.7816437445112694, 'Chair': 0.9029020491201142, 'Earphone': 0.774994567528407, 'Guitar': 0.9127525976848138, 'Knife': 0.831074125861463, 'Lamp': 0.8399659210204411, 'Laptop': 0.9544977515135628, 'Motorbike': 0.7087279385636823, 'Mug': 0.9418834384796249, 'Pistol': 0.813307804611479, 'Rocket': 0.6219955894327281, 'Skateboard': 0.7716365133785822, 'Table': 0.8247335689611632}
==================================================
EPOCH 66 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.130, train_Cmiou=84.62, train_Imiou=87.32, train_loss_seg=0.106]
Learning rate = 0.000063
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.19it/s, test_Cmiou=82.56, test_Imiou=85.14, test_loss_seg=0.276]
==================================================
    test_loss_seg = 0.276111900806427
    test_Cmiou = 82.5678743323288
    test_Imiou = 85.14280202763318
    test_Imiou_per_class = {'Airplane': 0.828133226669318, 'Bag': 0.8458053721277341, 'Cap': 0.8690677373148499, 'Car': 0.7820734662427504, 'Chair': 0.9048302063390442, 'Earphone': 0.764980929807357, 'Guitar': 0.9130096718178214, 'Knife': 0.8426661038336627, 'Lamp': 0.8363448154485463, 'Laptop': 0.9539399776891817, 'Motorbike': 0.7049017363665693, 'Mug': 0.9396189461726007, 'Pistol': 0.8151291246861827, 'Rocket': 0.6236678783637557, 'Skateboard': 0.7583115130518262, 'Table': 0.8283791872414072}
==================================================
EPOCH 67 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=85.66, train_Imiou=87.05, train_loss_seg=0.110]
Learning rate = 0.000063
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.15it/s, test_Cmiou=82.63, test_Imiou=84.94, test_loss_seg=0.158]
==================================================
    test_loss_seg = 0.15830697119235992
    test_Cmiou = 82.63261634079821
    test_Imiou = 84.94210537633981
    test_Imiou_per_class = {'Airplane': 0.8250653180787804, 'Bag': 0.8521547864949437, 'Cap': 0.8705755334475295, 'Car': 0.7797041730473515, 'Chair': 0.9030963471711911, 'Earphone': 0.7711205016294367, 'Guitar': 0.9116596574334028, 'Knife': 0.8330237876026695, 'Lamp': 0.8358104488558995, 'Laptop': 0.955219233360935, 'Motorbike': 0.7155950011304446, 'Mug': 0.9398340842427558, 'Pistol': 0.8150751748902887, 'Rocket': 0.6193059884095159, 'Skateboard': 0.769284045853823, 'Table': 0.8246945328787448}
==================================================
EPOCH 68 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=84.17, train_Imiou=88.19, train_loss_seg=0.107]
Learning rate = 0.000063
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.20it/s, test_Cmiou=82.67, test_Imiou=84.98, test_loss_seg=0.179]
==================================================
    test_loss_seg = 0.17899662256240845
    test_Cmiou = 82.67545460045656
    test_Imiou = 84.98757136110535
    test_Imiou_per_class = {'Airplane': 0.8209005466254431, 'Bag': 0.8469564979293546, 'Cap': 0.8704453994275357, 'Car': 0.7829942460145092, 'Chair': 0.9041351512834257, 'Earphone': 0.7675422631117559, 'Guitar': 0.9130913095765736, 'Knife': 0.8376908038602362, 'Lamp': 0.8345337017469099, 'Laptop': 0.9553269234446007, 'Motorbike': 0.7214532094088036, 'Mug': 0.93920709951812, 'Pistol': 0.8180871455495136, 'Rocket': 0.616959829644847, 'Skateboard': 0.7730405348537609, 'Table': 0.8257080740776606}
==================================================
EPOCH 69 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:38<00:00,  3.45it/s, data_loading=0.006, iteration=0.138, train_Cmiou=86.14, train_Imiou=88.01, train_loss_seg=0.103]
Learning rate = 0.000063
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.09it/s, test_Cmiou=82.49, test_Imiou=84.98, test_loss_seg=0.358]
==================================================
    test_loss_seg = 0.35848501324653625
    test_Cmiou = 82.4991653796212
    test_Imiou = 84.9878925307918
    test_Imiou_per_class = {'Airplane': 0.8231311376963037, 'Bag': 0.8457461837643366, 'Cap': 0.8726625745874382, 'Car': 0.7839492294307807, 'Chair': 0.9028706370233592, 'Earphone': 0.7698377515781974, 'Guitar': 0.9126910145823824, 'Knife': 0.8277418463231072, 'Lamp': 0.8357454884481328, 'Laptop': 0.9549737737977909, 'Motorbike': 0.716034151954668, 'Mug': 0.9399359890681566, 'Pistol': 0.8060125039320719, 'Rocket': 0.610697925611496, 'Skateboard': 0.7704463718528989, 'Table': 0.8273898810882713}
==================================================
EPOCH 70 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.007, iteration=0.132, train_Cmiou=84.48, train_Imiou=87.26, train_loss_seg=0.110]
Learning rate = 0.000063
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.70, test_Imiou=85.18, test_loss_seg=0.126]
==================================================
    test_loss_seg = 0.12671466171741486
    test_Cmiou = 82.70833060691028
    test_Imiou = 85.18233270590699
    test_Imiou_per_class = {'Airplane': 0.8263184206150198, 'Bag': 0.8490273863672915, 'Cap': 0.8652524481079822, 'Car': 0.781245390927671, 'Chair': 0.9041682343463474, 'Earphone': 0.7774749802976116, 'Guitar': 0.9126731892589747, 'Knife': 0.8377477734974054, 'Lamp': 0.8346492937354345, 'Laptop': 0.9551088170009182, 'Motorbike': 0.7268443481410057, 'Mug': 0.9407298728118206, 'Pistol': 0.8036525923996244, 'Rocket': 0.615902665053166, 'Skateboard': 0.7717663120256666, 'Table': 0.8307711725197057}
==================================================
EPOCH 71 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:35<00:00,  3.48it/s, data_loading=0.006, iteration=0.131, train_Cmiou=86.15, train_Imiou=87.56, train_loss_seg=0.105]
Learning rate = 0.000063
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.19it/s, test_Cmiou=82.60, test_Imiou=85.17, test_loss_seg=1.663]
==================================================
    test_loss_seg = 1.6638153791427612
    test_Cmiou = 82.60024096801494
    test_Imiou = 85.17808315805743
    test_Imiou_per_class = {'Airplane': 0.8241124222076709, 'Bag': 0.8490973652181723, 'Cap': 0.8732909883217291, 'Car': 0.782081023192087, 'Chair': 0.9050277666555573, 'Earphone': 0.772414365505756, 'Guitar': 0.912854069473607, 'Knife': 0.8320775974500265, 'Lamp': 0.8351527321930338, 'Laptop': 0.9550676191411759, 'Motorbike': 0.7152519445170048, 'Mug': 0.9403672229311593, 'Pistol': 0.8081925965217732, 'Rocket': 0.6090535917578977, 'Skateboard': 0.770414722101105, 'Table': 0.831582527694634}
==================================================
EPOCH 72 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.132, train_Cmiou=85.20, train_Imiou=87.67, train_loss_seg=0.106]
Learning rate = 0.000031
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.72, test_Imiou=85.02, test_loss_seg=0.163]
==================================================
    test_loss_seg = 0.16368243098258972
    test_Cmiou = 82.72383851950104
    test_Imiou = 85.0204432049671
    test_Imiou_per_class = {'Airplane': 0.8258792886510886, 'Bag': 0.8493383830312313, 'Cap': 0.868710334531454, 'Car': 0.7840353365051917, 'Chair': 0.9044417167030196, 'Earphone': 0.7641429226151298, 'Guitar': 0.912853008052252, 'Knife': 0.8404788920949475, 'Lamp': 0.8355219883500247, 'Laptop': 0.9554519810035792, 'Motorbike': 0.7174699932901594, 'Mug': 0.9407924600513572, 'Pistol': 0.8140247874190135, 'Rocket': 0.6283779250317078, 'Skateboard': 0.7701237927303713, 'Table': 0.8241713530596411}
==================================================
EPOCH 73 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:35<00:00,  3.48it/s, data_loading=0.006, iteration=0.132, train_Cmiou=86.67, train_Imiou=87.55, train_loss_seg=0.103]
Learning rate = 0.000031
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.72, test_Imiou=85.13, test_loss_seg=0.090]
==================================================
    test_loss_seg = 0.09034768491983414
    test_Cmiou = 82.72526925695563
    test_Imiou = 85.13573732544242
    test_Imiou_per_class = {'Airplane': 0.8257875045564645, 'Bag': 0.8460116955700936, 'Cap': 0.8732213521653659, 'Car': 0.7830536249759682, 'Chair': 0.9050482318758342, 'Earphone': 0.7681420716574051, 'Guitar': 0.9132965474225593, 'Knife': 0.8338423704511992, 'Lamp': 0.8374221953657265, 'Laptop': 0.9552910163958849, 'Motorbike': 0.7180077864604186, 'Mug': 0.940121677312979, 'Pistol': 0.812183682868475, 'Rocket': 0.6154477785850322, 'Skateboard': 0.7816680117648109, 'Table': 0.8274975336846819}
==================================================
EPOCH 74 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.48it/s, data_loading=0.006, iteration=0.132, train_Cmiou=85.34, train_Imiou=87.35, train_loss_seg=0.103]
Learning rate = 0.000031
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.57, test_Imiou=85.16, test_loss_seg=0.124]
==================================================
    test_loss_seg = 0.12403357028961182
    test_Cmiou = 82.57580855514742
    test_Imiou = 85.16994279658276
    test_Imiou_per_class = {'Airplane': 0.8252667034679838, 'Bag': 0.847839383876302, 'Cap': 0.8687830368774688, 'Car': 0.7832829052581407, 'Chair': 0.9042756921837025, 'Earphone': 0.7710927705351436, 'Guitar': 0.9124433980438913, 'Knife': 0.8355102158184184, 'Lamp': 0.8381103078734048, 'Laptop': 0.9544851122438799, 'Motorbike': 0.711188354938016, 'Mug': 0.9398597532885408, 'Pistol': 0.8133423182219365, 'Rocket': 0.6048605111609623, 'Skateboard': 0.7716169229616415, 'Table': 0.8301719820741528}
==================================================
EPOCH 75 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:55<00:00,  3.29it/s, data_loading=0.006, iteration=0.131, train_Cmiou=87.17, train_Imiou=87.01, train_loss_seg=0.107]
Learning rate = 0.000031
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.52, test_Imiou=85.10, test_loss_seg=0.412]
==================================================
    test_loss_seg = 0.41287198662757874
    test_Cmiou = 82.52336383307744
    test_Imiou = 85.10262960770295
    test_Imiou_per_class = {'Airplane': 0.8287164012510657, 'Bag': 0.8467686680115805, 'Cap': 0.8712794189215977, 'Car': 0.7831488577597485, 'Chair': 0.9052294618666162, 'Earphone': 0.7731129903838011, 'Guitar': 0.9131495014105261, 'Knife': 0.8349543097997472, 'Lamp': 0.839437635496129, 'Laptop': 0.9550648843431436, 'Motorbike': 0.7145588342620811, 'Mug': 0.939592038630022, 'Pistol': 0.8146080173377285, 'Rocket': 0.5909689201301395, 'Skateboard': 0.7679705923809236, 'Table': 0.8251776813075399}
==================================================
EPOCH 76 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:35<00:00,  3.48it/s, data_loading=0.006, iteration=0.131, train_Cmiou=86.21, train_Imiou=87.17, train_loss_seg=0.108]
Learning rate = 0.000031
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.56, test_Imiou=84.99, test_loss_seg=0.088]
==================================================
    test_loss_seg = 0.08817765861749649
    test_Cmiou = 82.56179279171461
    test_Imiou = 84.99887431020127
    test_Imiou_per_class = {'Airplane': 0.8220708487277121, 'Bag': 0.8486916860181386, 'Cap': 0.8687583724926712, 'Car': 0.783244907608814, 'Chair': 0.9047427229366805, 'Earphone': 0.7699568390710086, 'Guitar': 0.9121072208180487, 'Knife': 0.8341852656290637, 'Lamp': 0.833652071403557, 'Laptop': 0.9552798589077913, 'Motorbike': 0.7195323321216778, 'Mug': 0.9409562175857256, 'Pistol': 0.8107878432081546, 'Rocket': 0.6082308596501873, 'Skateboard': 0.771244757715645, 'Table': 0.8264450427794615}
==================================================
EPOCH 77 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:35<00:00,  3.48it/s, data_loading=0.006, iteration=0.131, train_Cmiou=86.62, train_Imiou=87.67, train_loss_seg=0.108]
Learning rate = 0.000031
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.72, test_Imiou=85.14, test_loss_seg=0.908]
==================================================
    test_loss_seg = 0.9085749983787537
    test_Cmiou = 82.7271387627066
    test_Imiou = 85.14121259286185
    test_Imiou_per_class = {'Airplane': 0.8256999293025822, 'Bag': 0.8474703605285809, 'Cap': 0.8723767428918049, 'Car': 0.7836772336029252, 'Chair': 0.9051650620599786, 'Earphone': 0.7774685538063215, 'Guitar': 0.913594396072443, 'Knife': 0.8384728500662362, 'Lamp': 0.8359200114707925, 'Laptop': 0.9545290088557815, 'Motorbike': 0.7162074589577506, 'Mug': 0.9413654920934128, 'Pistol': 0.8112545634045089, 'Rocket': 0.6132051184411768, 'Skateboard': 0.7720238156980253, 'Table': 0.8279116047807352}
==================================================
EPOCH 78 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.48it/s, data_loading=0.006, iteration=0.132, train_Cmiou=86.52, train_Imiou=87.64, train_loss_seg=0.103]
Learning rate = 0.000031
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.78, test_Imiou=85.13, test_loss_seg=0.046]
==================================================
    test_loss_seg = 0.0462506003677845
    test_Cmiou = 82.78019852222316
    test_Imiou = 85.13713406330208
    test_Imiou_per_class = {'Airplane': 0.8259864639460615, 'Bag': 0.8512609850412537, 'Cap': 0.8723287752096841, 'Car': 0.7835308261605867, 'Chair': 0.9059411321095912, 'Earphone': 0.7675746489454057, 'Guitar': 0.9135948213547299, 'Knife': 0.8333777650442766, 'Lamp': 0.8370519202925176, 'Laptop': 0.9552833106161185, 'Motorbike': 0.7192223854777942, 'Mug': 0.9420391056305477, 'Pistol': 0.81629936372763, 'Rocket': 0.6233936527399382, 'Skateboard': 0.7713727079780051, 'Table': 0.8265738992815643}
==================================================
Cmiou: 82.78002739702927 -> 82.78019852222316
EPOCH 79 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=85.44, train_Imiou=86.80, train_loss_seg=0.110]
Learning rate = 0.000031
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:26<00:00,  8.98it/s, test_Cmiou=82.65, test_Imiou=85.14, test_loss_seg=0.217]
==================================================
    test_loss_seg = 0.21724724769592285
    test_Cmiou = 82.65479208310397
    test_Imiou = 85.14457932293135
    test_Imiou_per_class = {'Airplane': 0.8260990774161929, 'Bag': 0.8468347540605298, 'Cap': 0.8621814377892965, 'Car': 0.7845749340734818, 'Chair': 0.9056449057811226, 'Earphone': 0.7627165600542226, 'Guitar': 0.913182951975813, 'Knife': 0.8356457445076971, 'Lamp': 0.838147006658288, 'Laptop': 0.9544054122414832, 'Motorbike': 0.7171766493769086, 'Mug': 0.9420443091124545, 'Pistol': 0.8145235380019588, 'Rocket': 0.6236182886615947, 'Skateboard': 0.7710505261822138, 'Table': 0.8269206374033763}
==================================================
EPOCH 80 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:49<00:00,  3.35it/s, data_loading=0.006, iteration=0.132, train_Cmiou=87.04, train_Imiou=87.31, train_loss_seg=0.103]
Learning rate = 0.000031
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.64, test_Imiou=85.07, test_loss_seg=0.194]
==================================================
    test_loss_seg = 0.1941176801919937
    test_Cmiou = 82.6434829249035
    test_Imiou = 85.0749505288652
    test_Imiou_per_class = {'Airplane': 0.8251147093200122, 'Bag': 0.8466403861141896, 'Cap': 0.867663034325489, 'Car': 0.7831013565383016, 'Chair': 0.9048333971262249, 'Earphone': 0.7701486257707748, 'Guitar': 0.9126537723874131, 'Knife': 0.8333010493508144, 'Lamp': 0.8363955719572312, 'Laptop': 0.9553172733269962, 'Motorbike': 0.7149809538003369, 'Mug': 0.9404382788797135, 'Pistol': 0.8151894607815944, 'Rocket': 0.6192024253995644, 'Skateboard': 0.7712148418612381, 'Table': 0.8267621310446646}
==================================================
EPOCH 81 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:56<00:00,  3.28it/s, data_loading=0.006, iteration=0.134, train_Cmiou=85.85, train_Imiou=87.95, train_loss_seg=0.103]
Learning rate = 0.000031
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.15it/s, test_Cmiou=82.70, test_Imiou=85.07, test_loss_seg=0.135]
==================================================
    test_loss_seg = 0.13560687005519867
    test_Cmiou = 82.70631954116405
    test_Imiou = 85.07518851532492
    test_Imiou_per_class = {'Airplane': 0.8255980295440515, 'Bag': 0.8451628656820878, 'Cap': 0.874831088931181, 'Car': 0.782979545984898, 'Chair': 0.9046337586285058, 'Earphone': 0.7643845845158133, 'Guitar': 0.9132812591719675, 'Knife': 0.8302286589838838, 'Lamp': 0.8404393151188198, 'Laptop': 0.9553140796028733, 'Motorbike': 0.7197120705772343, 'Mug': 0.9410970885983257, 'Pistol': 0.8101057816539184, 'Rocket': 0.6301260049166929, 'Skateboard': 0.7696656784901414, 'Table': 0.8254513161858537}
==================================================
EPOCH 82 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=86.96, train_Imiou=87.65, train_loss_seg=0.102]
Learning rate = 0.000031
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.59, test_Imiou=85.07, test_loss_seg=0.100]
==================================================
    test_loss_seg = 0.10063064843416214
    test_Cmiou = 82.59410726870632
    test_Imiou = 85.07709915417814
    test_Imiou_per_class = {'Airplane': 0.8250235432306899, 'Bag': 0.8482030008379499, 'Cap': 0.8732145260561847, 'Car': 0.7846307023988695, 'Chair': 0.9038166518609819, 'Earphone': 0.7690891991760245, 'Guitar': 0.9140074559066009, 'Knife': 0.8275618871767462, 'Lamp': 0.8363599673911473, 'Laptop': 0.9549843455739296, 'Motorbike': 0.7139710198379611, 'Mug': 0.9408902553086774, 'Pistol': 0.8164012450692578, 'Rocket': 0.6078830718499711, 'Skateboard': 0.7711992916067782, 'Table': 0.8278209997112402}
==================================================
EPOCH 83 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=86.18, train_Imiou=86.70, train_loss_seg=0.108]
Learning rate = 0.000031
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.75, test_Imiou=85.09, test_loss_seg=0.559]
==================================================
    test_loss_seg = 0.5592495203018188
    test_Cmiou = 82.75552201538142
    test_Imiou = 85.09916871042347
    test_Imiou_per_class = {'Airplane': 0.8265845138296957, 'Bag': 0.8475581760989053, 'Cap': 0.8748224236446671, 'Car': 0.7833417227984101, 'Chair': 0.9032348216712496, 'Earphone': 0.7695328957440143, 'Guitar': 0.9116334591582769, 'Knife': 0.8307682824880308, 'Lamp': 0.8354658054765667, 'Laptop': 0.9551576514685347, 'Motorbike': 0.7201015201380491, 'Mug': 0.9398638895191199, 'Pistol': 0.8134674385308552, 'Rocket': 0.624370028022824, 'Skateboard': 0.776504452643892, 'Table': 0.8284764412279348}
==================================================
EPOCH 84 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=86.95, train_Imiou=87.64, train_loss_seg=0.107]
Learning rate = 0.000031
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.60, test_Imiou=85.08, test_loss_seg=0.953]
==================================================
    test_loss_seg = 0.9530434012413025
    test_Cmiou = 82.6044515077152
    test_Imiou = 85.08433382551357
    test_Imiou_per_class = {'Airplane': 0.8233566902317797, 'Bag': 0.8465981922879398, 'Cap': 0.8723362586270685, 'Car': 0.7825804829356513, 'Chair': 0.9049623166584098, 'Earphone': 0.7727117107979308, 'Guitar': 0.9137654429498838, 'Knife': 0.8342059138729138, 'Lamp': 0.8331491049403285, 'Laptop': 0.9542567260829449, 'Motorbike': 0.7093288875402968, 'Mug': 0.9409862906910007, 'Pistol': 0.8149175333291246, 'Rocket': 0.6195013678349283, 'Skateboard': 0.76492017918288, 'Table': 0.8291351432713491}
==================================================
EPOCH 85 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.48it/s, data_loading=0.006, iteration=0.131, train_Cmiou=85.88, train_Imiou=87.72, train_loss_seg=0.106]
Learning rate = 0.000031
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.66, test_Imiou=85.02, test_loss_seg=0.126]
==================================================
    test_loss_seg = 0.12666727602481842
    test_Cmiou = 82.66305424772878
    test_Imiou = 85.02519514157181
    test_Imiou_per_class = {'Airplane': 0.8247770707477936, 'Bag': 0.8485571502158175, 'Cap': 0.8706191846451543, 'Car': 0.7832073724340496, 'Chair': 0.9046357733520374, 'Earphone': 0.7695734524836119, 'Guitar': 0.9140004057360962, 'Knife': 0.8371331656389656, 'Lamp': 0.83744677510884, 'Laptop': 0.9546432375273709, 'Motorbike': 0.714202570386211, 'Mug': 0.9408070745567411, 'Pistol': 0.8134355545535132, 'Rocket': 0.620572256721676, 'Skateboard': 0.7678602350071115, 'Table': 0.8246174005216138}
==================================================
EPOCH 86 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=86.47, train_Imiou=87.45, train_loss_seg=0.105]
Learning rate = 0.000016
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.64, test_Imiou=85.00, test_loss_seg=0.286]
==================================================
    test_loss_seg = 0.28665098547935486
    test_Cmiou = 82.64772638593001
    test_Imiou = 85.00768100074848
    test_Imiou_per_class = {'Airplane': 0.8196097329610204, 'Bag': 0.8473284995842806, 'Cap': 0.8667655552534558, 'Car': 0.7850182151572064, 'Chair': 0.9037618839883602, 'Earphone': 0.7673156494907348, 'Guitar': 0.9134737226936066, 'Knife': 0.8417614484034726, 'Lamp': 0.8383819461301835, 'Laptop': 0.9543891811035125, 'Motorbike': 0.714047262889869, 'Mug': 0.9409423703210152, 'Pistol': 0.8153958123490301, 'Rocket': 0.6187016752725942, 'Skateboard': 0.770959575180987, 'Table': 0.825783690969474}
==================================================
EPOCH 87 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.132, train_Cmiou=86.89, train_Imiou=87.35, train_loss_seg=0.102]
Learning rate = 0.000016
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.80, test_Imiou=85.10, test_loss_seg=0.309]
==================================================
    test_loss_seg = 0.3096754252910614
    test_Cmiou = 82.80514848345668
    test_Imiou = 85.10895649398199
    test_Imiou_per_class = {'Airplane': 0.8232152489010717, 'Bag': 0.8499359258976871, 'Cap': 0.8723617042772638, 'Car': 0.78544069628007, 'Chair': 0.904022752878463, 'Earphone': 0.7690024460982912, 'Guitar': 0.9135780407586203, 'Knife': 0.8373616144188236, 'Lamp': 0.8350432206678389, 'Laptop': 0.9550457716810986, 'Motorbike': 0.7176093155186886, 'Mug': 0.941836076559161, 'Pistol': 0.8127038222467481, 'Rocket': 0.622146863271899, 'Skateboard': 0.7812760186809471, 'Table': 0.8282442392163964}
==================================================
Cmiou: 82.78019852222316 -> 82.80514848345668
EPOCH 88 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=85.31, train_Imiou=87.80, train_loss_seg=0.105]
Learning rate = 0.000016
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.69, test_Imiou=85.15, test_loss_seg=0.167]
==================================================
    test_loss_seg = 0.1677943468093872
    test_Cmiou = 82.6975254038229
    test_Imiou = 85.15494592307996
    test_Imiou_per_class = {'Airplane': 0.8246374888720013, 'Bag': 0.8487443352496399, 'Cap': 0.8691635491270049, 'Car': 0.7847932333887747, 'Chair': 0.9050007639230039, 'Earphone': 0.7697014837164353, 'Guitar': 0.9137660396457301, 'Knife': 0.8359591578765606, 'Lamp': 0.8379627288577824, 'Laptop': 0.9551952476208807, 'Motorbike': 0.7139543456044924, 'Mug': 0.9420014140657762, 'Pistol': 0.8168877068970001, 'Rocket': 0.6163595910657583, 'Skateboard': 0.7692739332848387, 'Table': 0.8282030454159817}
==================================================
EPOCH 89 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:53<00:00,  3.31it/s, data_loading=0.006, iteration=0.145, train_Cmiou=85.87, train_Imiou=87.61, train_loss_seg=0.102]
Learning rate = 0.000016
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:24<00:00, 10.00it/s, test_Cmiou=82.59, test_Imiou=85.03, test_loss_seg=0.230]
==================================================
    test_loss_seg = 0.230404332280159
    test_Cmiou = 82.59378879617375
    test_Imiou = 85.03064031698713
    test_Imiou_per_class = {'Airplane': 0.8227500039459246, 'Bag': 0.8505853689203363, 'Cap': 0.869011966442876, 'Car': 0.7811712932360583, 'Chair': 0.9040094306677381, 'Earphone': 0.7634695201172885, 'Guitar': 0.9133028029667614, 'Knife': 0.8316931300849539, 'Lamp': 0.8356670754453634, 'Laptop': 0.9549438981797294, 'Motorbike': 0.7180662581433988, 'Mug': 0.941755694933712, 'Pistol': 0.8158591847987936, 'Rocket': 0.6161087554816344, 'Skateboard': 0.7691780991191667, 'Table': 0.8274337249040669}
==================================================
EPOCH 90 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:37<00:00,  3.46it/s, data_loading=0.006, iteration=0.132, train_Cmiou=84.68, train_Imiou=86.81, train_loss_seg=0.106]
Learning rate = 0.000016
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.74, test_Imiou=85.07, test_loss_seg=3.705]
==================================================
    test_loss_seg = 3.7053720951080322
    test_Cmiou = 82.74649594702979
    test_Imiou = 85.07728356236642
    test_Imiou_per_class = {'Airplane': 0.823434592581187, 'Bag': 0.8504392556823656, 'Cap': 0.8719287559068337, 'Car': 0.7839727349780081, 'Chair': 0.9035317816879569, 'Earphone': 0.7716085601854221, 'Guitar': 0.9132468265570481, 'Knife': 0.8359828709239483, 'Lamp': 0.8349761939368867, 'Laptop': 0.9552694906595685, 'Motorbike': 0.7187598596673102, 'Mug': 0.9419296130807101, 'Pistol': 0.8160707470630729, 'Rocket': 0.6200077356511854, 'Skateboard': 0.7701817125317745, 'Table': 0.8280986204314862}
==================================================
EPOCH 91 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:38<00:00,  3.45it/s, data_loading=0.006, iteration=0.130, train_Cmiou=82.98, train_Imiou=86.90, train_loss_seg=0.106]
Learning rate = 0.000016
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.73, test_Imiou=85.12, test_loss_seg=0.295]
==================================================
    test_loss_seg = 0.2952561378479004
    test_Cmiou = 82.73552012603481
    test_Imiou = 85.12324729717999
    test_Imiou_per_class = {'Airplane': 0.8262798972401254, 'Bag': 0.8474915093924315, 'Cap': 0.8709346074709338, 'Car': 0.7840882794361886, 'Chair': 0.9038439472812617, 'Earphone': 0.7702419387372784, 'Guitar': 0.9143639570634343, 'Knife': 0.8340276752689852, 'Lamp': 0.8379844136034185, 'Laptop': 0.9553430670716307, 'Motorbike': 0.7189620357680638, 'Mug': 0.9410787458394531, 'Pistol': 0.8151031568563806, 'Rocket': 0.6204011870138155, 'Skateboard': 0.7701999598463841, 'Table': 0.8273388422757841}
==================================================
EPOCH 92 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=86.45, train_Imiou=87.48, train_loss_seg=0.104]
Learning rate = 0.000016
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.16it/s, test_Cmiou=82.61, test_Imiou=85.08, test_loss_seg=0.118]
==================================================
    test_loss_seg = 0.11881987005472183
    test_Cmiou = 82.6139196134149
    test_Imiou = 85.08869393518103
    test_Imiou_per_class = {'Airplane': 0.8236434957437907, 'Bag': 0.8462932620362235, 'Cap': 0.8745579876629315, 'Car': 0.7833747535106542, 'Chair': 0.9037436433480951, 'Earphone': 0.7659333264710905, 'Guitar': 0.9150121977285747, 'Knife': 0.8309594880855815, 'Lamp': 0.8364023801909087, 'Laptop': 0.9544451142170134, 'Motorbike': 0.7147097392623579, 'Mug': 0.9416451722662413, 'Pistol': 0.8115406774799779, 'Rocket': 0.6202077103982866, 'Skateboard': 0.7669437926717642, 'Table': 0.8288143970728898}
==================================================
EPOCH 93 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:45<00:00,  3.38it/s, data_loading=0.006, iteration=0.133, train_Cmiou=85.80, train_Imiou=88.12, train_loss_seg=0.100]
Learning rate = 0.000016
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:24<00:00,  9.98it/s, test_Cmiou=82.67, test_Imiou=85.05, test_loss_seg=0.211]
==================================================
    test_loss_seg = 0.21106493473052979
    test_Cmiou = 82.67050378943969
    test_Imiou = 85.05382458412521
    test_Imiou_per_class = {'Airplane': 0.8264957474664263, 'Bag': 0.8477764018503745, 'Cap': 0.8758259832481847, 'Car': 0.7828318036098725, 'Chair': 0.9047980688050952, 'Earphone': 0.7587258564757986, 'Guitar': 0.9136860985165969, 'Knife': 0.834687817336323, 'Lamp': 0.8351338755724254, 'Laptop': 0.955406425010649, 'Motorbike': 0.7155808170386472, 'Mug': 0.9403337601087801, 'Pistol': 0.8139277911796223, 'Rocket': 0.6271001985724561, 'Skateboard': 0.7692490918438831, 'Table': 0.8257208696752163}
==================================================
EPOCH 94 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:56<00:00,  3.28it/s, data_loading=0.005, iteration=0.368, train_Cmiou=87.30, train_Imiou=87.51, train_loss_seg=0.103]
Learning rate = 0.000016
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:28<00:00,  8.45it/s, test_Cmiou=82.63, test_Imiou=84.99, test_loss_seg=0.179]
==================================================
    test_loss_seg = 0.17907877266407013
    test_Cmiou = 82.63737516125934
    test_Imiou = 84.99719726653137
    test_Imiou_per_class = {'Airplane': 0.822379885350051, 'Bag': 0.8455399591234638, 'Cap': 0.8719019523551864, 'Car': 0.783691464266464, 'Chair': 0.9044452376391745, 'Earphone': 0.7696762023278912, 'Guitar': 0.9139397809377187, 'Knife': 0.8318837880778147, 'Lamp': 0.8369953374146943, 'Laptop': 0.9551514992502209, 'Motorbike': 0.711926471502944, 'Mug': 0.9402578473498852, 'Pistol': 0.8133877387122375, 'Rocket': 0.6246312014459745, 'Skateboard': 0.7708327513995146, 'Table': 0.8253389086482584}
==================================================
EPOCH 95 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:40<00:00,  3.43it/s, data_loading=0.006, iteration=0.130, train_Cmiou=86.65, train_Imiou=87.62, train_loss_seg=0.103]
Learning rate = 0.000016
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.74, test_Imiou=85.10, test_loss_seg=0.068]
==================================================
    test_loss_seg = 0.0680479109287262
    test_Cmiou = 82.74686046598158
    test_Imiou = 85.10331101455691
    test_Imiou_per_class = {'Airplane': 0.8253985387869082, 'Bag': 0.8485253127462693, 'Cap': 0.8729444089880797, 'Car': 0.7848942122540937, 'Chair': 0.9043464599078106, 'Earphone': 0.7651351198059506, 'Guitar': 0.9139789971820858, 'Knife': 0.8378481835402003, 'Lamp': 0.8378285385838813, 'Laptop': 0.9549302353677023, 'Motorbike': 0.7139678800588627, 'Mug': 0.9409699026783485, 'Pistol': 0.8130088285869341, 'Rocket': 0.6282079160895542, 'Skateboard': 0.7709406391414054, 'Table': 0.8265725008389678}
==================================================
EPOCH 96 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=85.69, train_Imiou=87.68, train_loss_seg=0.105]
Learning rate = 0.000016
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.18it/s, test_Cmiou=82.81, test_Imiou=85.14, test_loss_seg=0.216]
==================================================
    test_loss_seg = 0.21643976867198944
    test_Cmiou = 82.81135875470305
    test_Imiou = 85.14380953895215
    test_Imiou_per_class = {'Airplane': 0.8270061230080823, 'Bag': 0.8493233084271087, 'Cap': 0.8650182085485778, 'Car': 0.7855429882292723, 'Chair': 0.9047852480620399, 'Earphone': 0.7790459832036333, 'Guitar': 0.9139993705391923, 'Knife': 0.8409024010661648, 'Lamp': 0.8357014645416262, 'Laptop': 0.9550579474624091, 'Motorbike': 0.7110050748636725, 'Mug': 0.9412701351201642, 'Pistol': 0.8192732945852339, 'Rocket': 0.6235927662781297, 'Skateboard': 0.771315421271747, 'Table': 0.8269776655454337}
==================================================
Cmiou: 82.80514848345668 -> 82.81135875470305
EPOCH 97 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:36<00:00,  3.47it/s, data_loading=0.006, iteration=0.131, train_Cmiou=86.53, train_Imiou=86.93, train_loss_seg=0.108]
Learning rate = 0.000016
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:23<00:00, 10.17it/s, test_Cmiou=82.72, test_Imiou=85.07, test_loss_seg=0.689]
==================================================
    test_loss_seg = 0.6891748905181885
    test_Cmiou = 82.72106861984827
    test_Imiou = 85.07638129606217
    test_Imiou_per_class = {'Airplane': 0.8249401147930723, 'Bag': 0.8491882646630724, 'Cap': 0.8716480952963847, 'Car': 0.7801891497555686, 'Chair': 0.904868380252686, 'Earphone': 0.7697938436931862, 'Guitar': 0.9124537583262219, 'Knife': 0.8386233329215178, 'Lamp': 0.8334797850171515, 'Laptop': 0.9549707506372705, 'Motorbike': 0.7132215031731942, 'Mug': 0.9419259227640239, 'Pistol': 0.8178557127398761, 'Rocket': 0.623448011454023, 'Skateboard': 0.7710587157872879, 'Table': 0.8277056379011863}
==================================================
EPOCH 98 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:47<00:00,  3.36it/s, data_loading=0.006, iteration=0.136, train_Cmiou=85.81, train_Imiou=87.96, train_loss_seg=0.102]
Learning rate = 0.000016
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:24<00:00,  9.94it/s, test_Cmiou=82.72, test_Imiou=85.07, test_loss_seg=0.119]
==================================================
    test_loss_seg = 0.11923985928297043
    test_Cmiou = 82.72254609967693
    test_Imiou = 85.07893879176963
    test_Imiou_per_class = {'Airplane': 0.825863430932043, 'Bag': 0.8478520521894607, 'Cap': 0.8696218830384019, 'Car': 0.7836738043621558, 'Chair': 0.9044058765524128, 'Earphone': 0.7677265239339407, 'Guitar': 0.9126360876128935, 'Knife': 0.8367805442080704, 'Lamp': 0.8367005620738912, 'Laptop': 0.9547663185089644, 'Motorbike': 0.7179131534663832, 'Mug': 0.9422900147122635, 'Pistol': 0.8133819556314097, 'Rocket': 0.6259416128374069, 'Skateboard': 0.7697974625108119, 'Table': 0.8262560933777987}
==================================================
EPOCH 99 / 100
100%|██████████████████████████████████████████████████████████████████| 1168/1168 [05:51<00:00,  3.32it/s, data_loading=0.006, iteration=0.135, train_Cmiou=86.11, train_Imiou=87.51, train_loss_seg=0.102]
Learning rate = 0.000016
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:24<00:00,  9.86it/s, test_Cmiou=82.57, test_Imiou=85.13, test_loss_seg=0.512]
==================================================
    test_loss_seg = 0.5127086043357849
    test_Cmiou = 82.57370585001053
    test_Imiou = 85.13352229711745
    test_Imiou_per_class = {'Airplane': 0.8244139668339339, 'Bag': 0.8472346639162536, 'Cap': 0.8561745670579584, 'Car': 0.7847442837550577, 'Chair': 0.9050730803276622, 'Earphone': 0.7696407940832379, 'Guitar': 0.9132956973045946, 'Knife': 0.8336280980142263, 'Lamp': 0.837400636098979, 'Laptop': 0.9552409111879531, 'Motorbike': 0.7175058991993183, 'Mug': 0.941834690137604, 'Pistol': 0.8165710265029233, 'Rocket': 0.6111093354014496, 'Skateboard': 0.769858499543118, 'Table': 0.8280667866374137}
==================================================