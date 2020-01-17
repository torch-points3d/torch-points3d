```
SegmentationModel(
  (model): UnetSkipConnectionBlock(
    (down): PointNetMSGDown(
      (mlps): ModuleList(
        (0): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(6, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer1): Conv2d(
            (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer2): Conv2d(
            (conv): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
        )
        (1): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(6, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer1): Conv2d(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer2): Conv2d(
            (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
        )
        (2): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(6, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer1): Conv2d(
            (conv): Conv2d(64, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer2): Conv2d(
            (conv): Conv2d(96, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
        )
      )
    )
    (submodule): UnetSkipConnectionBlock(
      (down): PointNetMSGDown(
        (mlps): ModuleList(
          (0): SharedMLP(
            (layer0): Conv2d(
              (conv): Conv2d(323, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (activation): ReLU(inplace=True)
            )
            (layer1): Conv2d(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (activation): ReLU(inplace=True)
            )
            (layer2): Conv2d(
              (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (activation): ReLU(inplace=True)
            )
          )
          (1): SharedMLP(
            (layer0): Conv2d(
              (conv): Conv2d(323, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (activation): ReLU(inplace=True)
            )
            (layer1): Conv2d(
              (conv): Conv2d(128, 196, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(196, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (activation): ReLU(inplace=True)
            )
            (layer2): Conv2d(
              (conv): Conv2d(196, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (activation): ReLU(inplace=True)
            )
          )
        )
      )
      (submodule): UnetSkipConnectionBlock(
        (inner): GlobalDenseBaseModule(
          (nn): SharedMLP(
            (layer0): Conv2d(
              (conv): Conv2d(515, 256, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU(inplace=True)
            )
            (layer1): Conv2d(
              (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU(inplace=True)
            )
            (layer2): Conv2d(
              (conv): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU(inplace=True)
            )
          )
        )
        (up): DenseFPModule(
          (nn): SharedMLP(
            (layer0): Conv2d(
              (conv): Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (activation): ReLU(inplace=True)
            )
            (layer1): Conv2d(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (activation): ReLU(inplace=True)
            )
          )
        )
      )
      (up): DenseFPModule(
        (nn): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(576, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer1): Conv2d(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
        )
      )
    )
    (up): DenseFPModule(
      (nn): SharedMLP(
        (layer0): Conv2d(
          (conv): Conv2d(131, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normlayer): BatchNorm2d(
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU(inplace=True)
        )
        (layer1): Conv2d(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normlayer): BatchNorm2d(
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU(inplace=True)
        )
      )
    )
  )
)
    test_acc = 92.85835378686252
    test_macc = 86.64299746256908
    test_miou = 79.99710743728609
    test_acc_per_class = {'Airplane': 91.19217948130498, 'Bag': 95.99958147321429, 'Cap': 92.1475497159091, 'Car': 91.5811659414557, 'Chair': 94.9159795587713, 'Earphone': 90.45061383928571, 'Guitar': 96.23685632861635, 'Knife': 92.17041015625, 'Lamp': 91.73438865821679, 'Laptop': 97.2526826054217, 'Motorbike': 86.58183976715686, 'Mug': 99.4384765625, 'Pistol': 95.16379616477273, 'Rocket': 81.73014322916666, 'Skateboard': 94.17685231854838, 'Table': 94.96114478920991}
    test_macc_per_class = {'Airplane': 89.06243767787745, 'Bag': 84.09410361964764, 'Cap': 86.54321207586608, 'Car': 87.7766907282271, 'Chair': 91.56436731086353, 'Earphone': 70.48676946450136, 'Guitar': 94.36338527343086, 'Knife': 92.20011208213583, 'Lamp': 90.13929039908149, 'Laptop': 97.40733601531446, 'Motorbike': 78.55112059647392, 'Mug': 96.37424104485675, 'Pistol': 85.95718166476736, 'Rocket': 66.45608530138686, 'Skateboard': 85.07896230817131, 'Table': 90.23266383850316}
    test_miou_per_class = {'Airplane': 81.30800228045005, 'Bag': 79.90154219367757, 'Cap': 80.95288325004574, 'Car': 77.86561384816493, 'Chair': 85.53427826230133, 'Earphone': 63.080359997168365, 'Guitar': 90.16618477875281, 'Knife': 85.47124348409487, 'Lamp': 82.68459221203005, 'Laptop': 94.63087773652728, 'Motorbike': 70.0856598757985, 'Mug': 94.52961288041513, 'Pistol': 80.60174154334135, 'Rocket': 57.74179685696749, 'Skateboard': 72.14278482957299, 'Table': 83.2565449672691}
==================================================
EPOCH 27 / 100
100%|█████████████████████████████| 876/876 [15:46<00:00,  1.08s/it, data_loading=0.573, iteration=0.231, train_acc=94.52, train_loss_seg=0.131, train_macc=89.64, train_miou=84.02]
Learning rate = 0.000500
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:30<00:00,  2.00it/s, test_acc=93.22, test_loss_seg=0.153, test_macc=87.85, test_miou=80.96]
==================================================
    test_loss_seg = 0.15340451896190643
    test_acc = 93.22898699137852
    test_macc = 87.85805703177758
    test_miou = 80.96247394197191
    test_acc_per_class = {'Airplane': 90.62757743768329, 'Bag': 96.11118861607143, 'Cap': 94.02521306818183, 'Car': 91.44580696202532, 'Chair': 94.99241222034802, 'Earphone': 92.29561941964286, 'Guitar': 96.17021668632076, 'Knife': 93.56689453125, 'Lamp': 91.45968777316433, 'Laptop': 98.17924039909639, 'Motorbike': 86.97916666666666, 'Mug': 99.46931537828947, 'Pistol': 94.35924183238636, 'Rocket': 81.34765625, 'Skateboard': 95.50308719758065, 'Table': 95.13146742334906}
    test_macc_per_class = {'Airplane': 88.2120060003439, 'Bag': 84.03704470054272, 'Cap': 90.42198460141482, 'Car': 86.68123156303913, 'Chair': 92.22595957067684, 'Earphone': 71.1377068772335, 'Guitar': 93.38804561422519, 'Knife': 93.55317151502673, 'Lamp': 90.28987301451377, 'Laptop': 98.14970383845026, 'Motorbike': 83.86486246068793, 'Mug': 96.19329426862734, 'Pistol': 90.22163671659838, 'Rocket': 71.62019314384735, 'Skateboard': 84.49905116803619, 'Table': 91.23314745517713}
    test_miou_per_class = {'Airplane': 80.65380838732187, 'Bag': 80.24180951813926, 'Cap': 85.52031373418691, 'Car': 77.22299128601095, 'Chair': 85.97673551319005, 'Earphone': 65.24062052051468, 'Guitar': 89.73535384246146, 'Knife': 87.90561125158978, 'Lamp': 82.19385505627528, 'Laptop': 96.39722300937387, 'Motorbike': 71.07819359908385, 'Mug': 94.78214715603237, 'Pistol': 79.61865910766471, 'Rocket': 59.514398968796364, 'Skateboard': 75.03366818846827, 'Table': 84.28419393244099}
==================================================
EPOCH 28 / 100
100%|█████████████████████████████| 876/876 [15:48<00:00,  1.08s/it, data_loading=0.570, iteration=0.230, train_acc=94.65, train_loss_seg=0.131, train_macc=89.04, train_miou=83.46]
Learning rate = 0.000500
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:29<00:00,  2.00it/s, test_acc=92.80, test_loss_seg=0.100, test_macc=86.38, test_miou=79.91]
==================================================
    test_loss_seg = 0.10021613538265228
    test_acc = 92.80427131779552
    test_macc = 86.38394935831555
    test_miou = 79.91970122660882
    test_acc_per_class = {'Airplane': 90.89720605755132, 'Bag': 95.59500558035714, 'Cap': 91.89009232954545, 'Car': 91.27490852452532, 'Chair': 94.72586891867898, 'Earphone': 89.03111049107143, 'Guitar': 96.2758574095912, 'Knife': 90.9991455078125, 'Lamp': 92.11408708479021, 'Laptop': 98.11452842620481, 'Motorbike': 86.36354932598039, 'Mug': 99.45261101973685, 'Pistol': 95.12717507102273, 'Rocket': 82.47884114583334, 'Skateboard': 95.32825100806451, 'Table': 95.20010318396226}
    test_macc_per_class = {'Airplane': 88.94360221404386, 'Bag': 80.90085808199245, 'Cap': 85.90292619013543, 'Car': 84.60633288726973, 'Chair': 92.15843884959686, 'Earphone': 70.9059370489332, 'Guitar': 93.68911824692113, 'Knife': 91.02153014139623, 'Lamp': 91.07645826302205, 'Laptop': 98.09734732110445, 'Motorbike': 79.12492820838155, 'Mug': 96.56731496049736, 'Pistol': 85.61990236075296, 'Rocket': 69.89330283222094, 'Skateboard': 83.6305206742677, 'Table': 90.00467145251254}
    test_miou_per_class = {'Airplane': 80.78157055117343, 'Bag': 77.35969159376795, 'Cap': 80.28253065818791, 'Car': 76.5164429398088, 'Chair': 85.46628344217666, 'Earphone': 62.04618864045758, 'Guitar': 89.93403329233881, 'Knife': 83.48154491992675, 'Lamp': 83.75901032104366, 'Laptop': 96.27251424563674, 'Motorbike': 69.51969575265848, 'Mug': 94.67321682103926, 'Pistol': 80.22168912889191, 'Rocket': 60.24364578545022, 'Skateboard': 74.26175440042991, 'Table': 83.89540713275309}
==================================================
EPOCH 29 / 100
100%|█████████████████████████████| 876/876 [15:48<00:00,  1.08s/it, data_loading=0.574, iteration=0.230, train_acc=94.96, train_loss_seg=0.128, train_macc=90.79, train_miou=85.09]
Learning rate = 0.000250
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:30<00:00,  2.00it/s, test_acc=93.23, test_loss_seg=0.196, test_macc=88.07, test_miou=81.17]
==================================================
    test_loss_seg = 0.1961469203233719
    test_acc = 93.23073759506124
    test_macc = 88.07382094853415
    test_miou = 81.17031880461357
    test_acc_per_class = {'Airplane': 91.41054572947213, 'Bag': 95.96819196428571, 'Cap': 92.74236505681817, 'Car': 91.76628016218355, 'Chair': 95.0377030806108, 'Earphone': 90.80287388392857, 'Guitar': 96.51201356132076, 'Knife': 92.623291015625, 'Lamp': 91.4357858937937, 'Laptop': 98.00687123493977, 'Motorbike': 86.31472120098039, 'Mug': 99.38707853618422, 'Pistol': 95.47230113636364, 'Rocket': 83.26416015625, 'Skateboard': 95.69839969758065, 'Table': 95.24921921064269}
    test_macc_per_class = {'Airplane': 89.42421771795684, 'Bag': 85.40382379748301, 'Cap': 87.76213757274512, 'Car': 86.46499252816187, 'Chair': 92.7078797963302, 'Earphone': 70.64083375743061, 'Guitar': 95.04001600129193, 'Knife': 92.64376171046031, 'Lamp': 91.56684300364158, 'Laptop': 98.01925754951804, 'Motorbike': 83.67225964252239, 'Mug': 97.22880146728664, 'Pistol': 86.7474776569102, 'Rocket': 76.58076191466664, 'Skateboard': 85.29472350133159, 'Table': 89.98334755880923}
    test_miou_per_class = {'Airplane': 81.76235449819579, 'Bag': 80.27460752031888, 'Cap': 82.39836596406276, 'Car': 77.55584305802572, 'Chair': 86.08657819418973, 'Earphone': 63.0715212758375, 'Guitar': 90.80025263236439, 'Knife': 86.25808350744008, 'Lamp': 83.19166709924777, 'Laptop': 96.06615774894739, 'Motorbike': 72.02578195221675, 'Mug': 94.1796161622361, 'Pistol': 81.19552884029052, 'Rocket': 63.86565827529235, 'Skateboard': 75.9725078860307, 'Table': 84.0205762591209}
==================================================
EPOCH 30 / 100
100%|█████████████████████████████| 876/876 [15:48<00:00,  1.08s/it, data_loading=0.574, iteration=0.236, train_acc=94.85, train_loss_seg=0.120, train_macc=90.81, train_miou=85.67]
Learning rate = 0.000250
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:29<00:00,  2.01it/s, test_acc=93.29, test_loss_seg=0.115, test_macc=86.85, test_miou=80.68]
==================================================
    test_loss_seg = 0.11502961814403534
    test_acc = 93.29741751972873
    test_macc = 86.85001433441582
    test_miou = 80.67996523191651
    test_acc_per_class = {'Airplane': 91.1255956744868, 'Bag': 95.99958147321429, 'Cap': 92.23188920454545, 'Car': 91.95757515822784, 'Chair': 95.0192538174716, 'Earphone': 91.83175223214286, 'Guitar': 96.33973319575472, 'Knife': 93.8232421875, 'Lamp': 92.05928348994755, 'Laptop': 98.17100432981928, 'Motorbike': 86.53301164215686, 'Mug': 99.4384765625, 'Pistol': 95.1693448153409, 'Rocket': 83.12174479166666, 'Skateboard': 94.76121471774194, 'Table': 95.17597702314269}
    test_macc_per_class = {'Airplane': 89.03032846867416, 'Bag': 82.52827540676121, 'Cap': 86.64299248428749, 'Car': 87.08500018893011, 'Chair': 93.07248800622054, 'Earphone': 71.17976525384326, 'Guitar': 94.9332697834113, 'Knife': 93.82134556542857, 'Lamp': 90.64099430528012, 'Laptop': 98.11776983399314, 'Motorbike': 79.96403825324565, 'Mug': 96.15382146208256, 'Pistol': 84.93867895644892, 'Rocket': 70.20119249480884, 'Skateboard': 80.90072247759542, 'Table': 90.38954640964167}
    test_miou_per_class = {'Airplane': 81.49019266459204, 'Bag': 79.3004440240375, 'Cap': 81.13064873848018, 'Car': 78.21987851451986, 'Chair': 85.90387250682713, 'Earphone': 64.73536122899971, 'Guitar': 90.42847266275447, 'Knife': 88.36393133007594, 'Lamp': 83.0220137958252, 'Laptop': 96.37950915676103, 'Motorbike': 69.48324923118211, 'Mug': 94.50629335816407, 'Pistol': 79.96895197215981, 'Rocket': 61.28597424238653, 'Skateboard': 72.74061367666374, 'Table': 83.92003660723488}
==================================================
EPOCH 31 / 100
100%|█████████████████████████████| 876/876 [15:47<00:00,  1.08s/it, data_loading=0.572, iteration=0.230, train_acc=95.34, train_loss_seg=0.116, train_macc=91.69, train_miou=86.29]
Learning rate = 0.000250
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:30<00:00,  2.00it/s, test_acc=93.17, test_loss_seg=0.085, test_macc=87.23, test_miou=80.75]
==================================================
    test_loss_seg = 0.08585391193628311
    test_acc = 93.17479283404612
    test_macc = 87.2334939488383
    test_miou = 80.75697033616278
    test_acc_per_class = {'Airplane': 91.37517755681817, 'Bag': 96.2158203125, 'Cap': 92.46271306818183, 'Car': 91.95726611946202, 'Chair': 95.0624639337713, 'Earphone': 88.82882254464286, 'Guitar': 96.4309404481132, 'Knife': 93.54248046875, 'Lamp': 91.9486519340035, 'Laptop': 98.10746893825302, 'Motorbike': 87.12756587009804, 'Mug': 99.46674547697368, 'Pistol': 95.35466974431817, 'Rocket': 82.88167317708334, 'Skateboard': 94.7265625, 'Table': 95.30766325176887}
    test_macc_per_class = {'Airplane': 90.22941763382764, 'Bag': 83.02168606168019, 'Cap': 87.19646833776883, 'Car': 87.15665523863217, 'Chair': 92.14378129209098, 'Earphone': 69.89354449363078, 'Guitar': 94.72227416056761, 'Knife': 93.52940852113849, 'Lamp': 90.50572842703522, 'Laptop': 98.07639702782734, 'Motorbike': 79.33172343682315, 'Mug': 96.36595383327888, 'Pistol': 87.96831779514088, 'Rocket': 69.20816381628526, 'Skateboard': 85.51262390129905, 'Table': 90.87375920438622}
    test_miou_per_class = {'Airplane': 81.94446050056139, 'Bag': 80.20480432836587, 'Cap': 81.72161524217609, 'Car': 78.28383558090579, 'Chair': 86.12458469788338, 'Earphone': 60.602004915178476, 'Guitar': 90.56895862300145, 'Knife': 87.86284281453165, 'Lamp': 83.05045884869833, 'Laptop': 96.2578093922929, 'Motorbike': 70.92770699341807, 'Mug': 94.77691994930714, 'Pistol': 81.44899227239195, 'Rocket': 60.50318661344794, 'Skateboard': 73.46915263398756, 'Table': 84.36419197245635}
==================================================
EPOCH 32 / 100
100%|█████████████████████████████| 876/876 [15:48<00:00,  1.08s/it, data_loading=0.575, iteration=0.227, train_acc=95.31, train_loss_seg=0.117, train_macc=90.99, train_miou=85.60]
Learning rate = 0.000250
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:29<00:00,  2.00it/s, test_acc=93.06, test_loss_seg=0.203, test_macc=87.12, test_miou=80.45]
==================================================
    test_loss_seg = 0.20361241698265076
    test_acc = 93.06406385793046
    test_macc = 87.12048879686581
    test_miou = 80.4596145496511
    test_acc_per_class = {'Airplane': 91.27881002565982, 'Bag': 96.34137834821429, 'Cap': 90.8203125, 'Car': 91.796875, 'Chair': 94.99400745738636, 'Earphone': 89.84723772321429, 'Guitar': 96.33819772012579, 'Knife': 93.7567138671875, 'Lamp': 91.52985686188812, 'Laptop': 97.85097420933735, 'Motorbike': 86.71587775735294, 'Mug': 99.44875616776315, 'Pistol': 95.2425870028409, 'Rocket': 83.24381510416666, 'Skateboard': 94.50447328629032, 'Table': 95.31514869545991}
    test_macc_per_class = {'Airplane': 89.97161136600984, 'Bag': 84.74701273527164, 'Cap': 84.42859388722587, 'Car': 85.38200318467446, 'Chair': 92.20138202431706, 'Earphone': 70.17629599470958, 'Guitar': 94.38570807320879, 'Knife': 93.74572932992015, 'Lamp': 89.80544114828879, 'Laptop': 97.87469517112717, 'Motorbike': 81.34764461850129, 'Mug': 97.34255017239317, 'Pistol': 85.41972277494715, 'Rocket': 73.5497358098793, 'Skateboard': 82.57158204530427, 'Table': 90.97811241407453}
    test_miou_per_class = {'Airplane': 81.64225364827809, 'Bag': 81.26505834264994, 'Cap': 77.98838099428598, 'Car': 77.55484305260184, 'Chair': 85.83102536674897, 'Earphone': 61.874529239911936, 'Guitar': 90.38948533449083, 'Knife': 88.2428112838634, 'Lamp': 82.24483152850999, 'Laptop': 95.76598672132354, 'Motorbike': 70.53932018062464, 'Mug': 94.71810789810641, 'Pistol': 79.851836655024, 'Rocket': 62.87426005966705, 'Skateboard': 72.08270550885838, 'Table': 84.4883969794728}
==================================================
EPOCH 33 / 100
100%|█████████████████████████████| 876/876 [15:46<00:00,  1.08s/it, data_loading=0.565, iteration=0.224, train_acc=95.59, train_loss_seg=0.115, train_macc=91.64, train_miou=86.71]
Learning rate = 0.000250
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:29<00:00,  2.00it/s, test_acc=93.17, test_loss_seg=0.135, test_macc=87.44, test_miou=80.85]
==================================================
    test_loss_seg = 0.135122612118721
    test_acc = 93.17240495206183
    test_macc = 87.44845845717404
    test_miou = 80.85693253475722
    test_acc_per_class = {'Airplane': 91.4917350164956, 'Bag': 96.38323102678571, 'Cap': 91.41956676136364, 'Car': 91.8552833267405, 'Chair': 95.04852294921875, 'Earphone': 88.91950334821429, 'Guitar': 96.3956245086478, 'Knife': 93.9215087890625, 'Lamp': 92.11391635708041, 'Laptop': 98.10629235692771, 'Motorbike': 86.91980698529412, 'Mug': 99.44747121710526, 'Pistol': 95.5777254971591, 'Rocket': 82.4951171875, 'Skateboard': 95.23689516129032, 'Table': 95.42627874410378}
    test_macc_per_class = {'Airplane': 89.48298147911312, 'Bag': 84.1738476036854, 'Cap': 85.02223493058077, 'Car': 87.03453467609717, 'Chair': 92.76949527200652, 'Earphone': 69.77468796055324, 'Guitar': 94.45888975011894, 'Knife': 93.91355275548563, 'Lamp': 89.8771150526542, 'Laptop': 98.09137434421447, 'Motorbike': 83.5750420112817, 'Mug': 96.042556789376, 'Pistol': 87.74932422919667, 'Rocket': 71.40123873509795, 'Skateboard': 84.52749915952757, 'Table': 91.28096056579543}
    test_miou_per_class = {'Airplane': 81.9324423154001, 'Bag': 81.20226529294713, 'Cap': 79.16953074104345, 'Car': 77.89996014803708, 'Chair': 85.98297460872246, 'Earphone': 60.54599968461872, 'Guitar': 90.50485528649543, 'Knife': 88.53658898782679, 'Lamp': 82.8139468456402, 'Laptop': 96.25670685941566, 'Motorbike': 72.20938870646147, 'Mug': 94.57355801790176, 'Pistol': 81.77631088592517, 'Rocket': 60.99132720422915, 'Skateboard': 74.38179379157471, 'Table': 84.93327117987607}
==================================================
EPOCH 34 / 100
100%|█████████████████████████████| 876/876 [15:47<00:00,  1.08s/it, data_loading=0.569, iteration=0.228, train_acc=95.01, train_loss_seg=0.117, train_macc=90.67, train_miou=84.88]
Learning rate = 0.000250
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:30<00:00,  2.00it/s, test_acc=93.13, test_loss_seg=0.156, test_macc=87.16, test_miou=80.51]
==================================================
    test_loss_seg = 0.1566714346408844
    test_acc = 93.13860638801172
    test_macc = 87.16784201264483
    test_miou = 80.51680126555296
    test_acc_per_class = {'Airplane': 91.45751237170087, 'Bag': 95.89494977678571, 'Cap': 90.98455255681817, 'Car': 91.94212321993672, 'Chair': 94.95072798295455, 'Earphone': 89.52287946428571, 'Guitar': 96.19816234276729, 'Knife': 93.7481689453125, 'Lamp': 91.99594350961539, 'Laptop': 98.06923004518072, 'Motorbike': 86.72928155637256, 'Mug': 99.45132606907895, 'Pistol': 95.2914151278409, 'Rocket': 83.67106119791666, 'Skateboard': 95.22271925403226, 'Table': 95.08764878758844}
    test_macc_per_class = {'Airplane': 90.14471800815083, 'Bag': 81.3812998468758, 'Cap': 84.22385075743897, 'Car': 86.83906886715509, 'Chair': 92.17215592272008, 'Earphone': 70.25123566789519, 'Guitar': 93.44238556728321, 'Knife': 93.73012429327252, 'Lamp': 90.59191881486194, 'Laptop': 98.05803094523561, 'Motorbike': 84.07813927344327, 'Mug': 96.05619166319835, 'Pistol': 86.19158470746667, 'Rocket': 72.86109033223265, 'Skateboard': 84.65582525217332, 'Table': 90.00785228291356}
    test_miou_per_class = {'Airplane': 81.95039234496188, 'Bag': 78.49644362880258, 'Cap': 78.15017554826669, 'Car': 78.08285354399462, 'Chair': 85.71278808389387, 'Earphone': 61.493329293520105, 'Guitar': 89.83670826003824, 'Knife': 88.22385983429145, 'Lamp': 83.07187414930586, 'Laptop': 96.18514563773677, 'Motorbike': 70.96490724000326, 'Mug': 94.60897500534438, 'Pistol': 80.54524133723014, 'Rocket': 63.00256329759934, 'Skateboard': 74.35831010058985, 'Table': 83.58525294326803}
==================================================
EPOCH 35 / 100
100%|█████████████████████████████| 876/876 [15:47<00:00,  1.08s/it, data_loading=0.569, iteration=0.227, train_acc=95.22, train_loss_seg=0.113, train_macc=90.60, train_miou=85.71]
Learning rate = 0.000250
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:30<00:00,  2.00it/s, test_acc=93.24, test_loss_seg=0.084, test_macc=87.64, test_miou=80.87]
==================================================
    test_loss_seg = 0.08403374254703522
    test_acc = 93.24986362383835
    test_macc = 87.64508671171639
    test_miou = 80.87228036053683
    test_acc_per_class = {'Airplane': 91.59082317631965, 'Bag': 95.85658482142857, 'Cap': 91.78355823863636, 'Car': 92.08984375, 'Chair': 95.03083662553267, 'Earphone': 90.96330915178571, 'Guitar': 96.43892492138365, 'Knife': 93.7152099609375, 'Lamp': 91.99833369755245, 'Laptop': 98.15806193524097, 'Motorbike': 86.63449754901961, 'Mug': 99.45775082236842, 'Pistol': 95.43235085227273, 'Rocket': 82.470703125, 'Skateboard': 95.22586945564517, 'Table': 95.15115989829009}
    test_macc_per_class = {'Airplane': 89.04640906700385, 'Bag': 83.79066595617574, 'Cap': 86.22623850636663, 'Car': 87.7611983565129, 'Chair': 91.96910312545982, 'Earphone': 71.0458907123327, 'Guitar': 94.6407168942902, 'Knife': 93.67976612004227, 'Lamp': 89.96881579186314, 'Laptop': 98.10887437499886, 'Motorbike': 82.87111967193545, 'Mug': 95.75795460430123, 'Pistol': 85.64950676192126, 'Rocket': 74.67695197728425, 'Skateboard': 86.05785872019786, 'Table': 91.07031674677604}
    test_miou_per_class = {'Airplane': 82.0641558794372, 'Bag': 79.33434867063099, 'Cap': 80.23520924643405, 'Car': 78.78097647308475, 'Chair': 85.85478447578984, 'Earphone': 63.57578984955058, 'Guitar': 90.54023568451215, 'Knife': 88.1509488464281, 'Lamp': 82.64427089377511, 'Laptop': 96.35466668490601, 'Motorbike': 70.65757813369449, 'Mug': 94.63455818076285, 'Pistol': 80.48778520404437, 'Rocket': 61.78889584558051, 'Skateboard': 74.90865715891717, 'Table': 83.94362454104105}
==================================================
EPOCH 36 / 100
100%|█████████████████████████████| 876/876 [15:48<00:00,  1.08s/it, data_loading=0.568, iteration=0.228, train_acc=95.39, train_loss_seg=0.118, train_macc=91.39, train_miou=86.29]
Learning rate = 0.000250
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:29<00:00,  2.01it/s, test_acc=93.18, test_loss_seg=0.166, test_macc=87.03, test_miou=80.70]
==================================================
    test_loss_seg = 0.16629894077777863
    test_acc = 93.1840362976093
    test_macc = 87.03941365956172
    test_miou = 80.7076158768367
    test_acc_per_class = {'Airplane': 91.48328674853371, 'Bag': 96.41810825892857, 'Cap': 91.89897017045455, 'Car': 92.06512064873418, 'Chair': 95.0775146484375, 'Earphone': 89.50544084821429, 'Guitar': 96.51385613207547, 'Knife': 93.85009765625, 'Lamp': 91.84177638767483, 'Laptop': 98.17276920180723, 'Motorbike': 86.75896139705883, 'Mug': 99.47831003289474, 'Pistol': 94.85529119318183, 'Rocket': 82.30387369791666, 'Skateboard': 95.51568800403226, 'Table': 95.20551573555424}
    test_macc_per_class = {'Airplane': 89.17969589436944, 'Bag': 84.46172600921913, 'Cap': 85.95650258994871, 'Car': 87.49709785916299, 'Chair': 92.40937630807616, 'Earphone': 70.07922308306519, 'Guitar': 95.07229655623304, 'Knife': 93.84826070089584, 'Lamp': 90.22855620366164, 'Laptop': 98.1531353313276, 'Motorbike': 78.9069243520677, 'Mug': 96.02402444255915, 'Pistol': 84.5990567717369, 'Rocket': 72.54369432236118, 'Skateboard': 83.6524462674419, 'Table': 90.01860186086071}
    test_miou_per_class = {'Airplane': 81.8470503478444, 'Bag': 81.42550763852611, 'Cap': 80.31809166699216, 'Car': 78.68272618503379, 'Chair': 86.18688976740931, 'Earphone': 61.42939877080785, 'Guitar': 90.86138931326587, 'Knife': 88.41160480368472, 'Lamp': 82.42479495629277, 'Laptop': 96.3853558978698, 'Motorbike': 69.21458138256197, 'Mug': 94.84495762140371, 'Pistol': 79.57461608721532, 'Rocket': 60.93875306467487, 'Skateboard': 74.96616774092381, 'Table': 83.80996878488072}
==================================================
EPOCH 37 / 100
100%|█████████████████████████████| 876/876 [15:47<00:00,  1.08s/it, data_loading=0.571, iteration=0.230, train_acc=94.57, train_loss_seg=0.114, train_macc=91.38, train_miou=85.48]
Learning rate = 0.000250
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:29<00:00,  2.00it/s, test_acc=93.37, test_loss_seg=0.101, test_macc=87.24, test_miou=81.00]
==================================================
    test_loss_seg = 0.10134632140398026
    test_acc = 93.37676534846744
    test_macc = 87.24977929860533
    test_miou = 81.00780575985672
    test_acc_per_class = {'Airplane': 91.59612124266863, 'Bag': 96.20884486607143, 'Cap': 92.71129261363636, 'Car': 92.16463113132912, 'Chair': 95.00774036754261, 'Earphone': 91.09584263392857, 'Guitar': 96.50617875393081, 'Knife': 93.3416748046875, 'Lamp': 91.4238349541084, 'Laptop': 98.06158226656626, 'Motorbike': 87.39181219362744, 'Mug': 99.42819695723685, 'Pistol': 95.09499289772727, 'Rocket': 83.1787109375, 'Skateboard': 95.64012096774194, 'Table': 95.1766679871757}
    test_macc_per_class = {'Airplane': 90.0597957060545, 'Bag': 83.3458742115248, 'Cap': 88.08413443296197, 'Car': 87.63194239611917, 'Chair': 92.42275250927976, 'Earphone': 70.30104043505679, 'Guitar': 94.54966297827542, 'Knife': 93.30280588427044, 'Lamp': 90.50723449496006, 'Laptop': 98.05754982393566, 'Motorbike': 80.43397309420497, 'Mug': 95.86997314421336, 'Pistol': 83.39273313878863, 'Rocket': 72.25731168832506, 'Skateboard': 85.73219001983568, 'Table': 90.04749481987906}
    test_miou_per_class = {'Airplane': 82.24594243284238, 'Bag': 80.30540005312446, 'Cap': 82.45478709589001, 'Car': 78.86113837408469, 'Chair': 85.97162105477182, 'Earphone': 63.43292136351971, 'Guitar': 90.78079183125853, 'Knife': 87.4869836031494, 'Lamp': 81.79734426433151, 'Laptop': 96.17085371303699, 'Motorbike': 71.45695339730031, 'Mug': 94.38558978536085, 'Pistol': 79.0746673224116, 'Rocket': 62.05819285285838, 'Skateboard': 75.8714510037603, 'Table': 83.77025401000672}
==================================================
EPOCH 38 / 100
100%|█████████████████████████████| 876/876 [15:47<00:00,  1.08s/it, data_loading=0.571, iteration=0.232, train_acc=95.06, train_loss_seg=0.118, train_macc=91.29, train_miou=86.00]
Learning rate = 0.000250
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:29<00:00,  2.00it/s, test_acc=93.00, test_loss_seg=0.161, test_macc=87.38, test_miou=80.62]
==================================================
    test_loss_seg = 0.16178269684314728
    test_acc = 93.00293456839157
    test_macc = 87.38331929450625
    test_miou = 80.6263402251304
    test_acc_per_class = {'Airplane': 91.33995257514663, 'Bag': 95.76939174107143, 'Cap': 93.701171875, 'Car': 92.04750543908227, 'Chair': 94.9948397549716, 'Earphone': 84.70284598214286, 'Guitar': 96.293361831761, 'Knife': 93.665771484375, 'Lamp': 91.6208547312063, 'Laptop': 98.20453689759037, 'Motorbike': 87.11224724264706, 'Mug': 99.4371916118421, 'Pistol': 95.24591619318183, 'Rocket': 83.056640625, 'Skateboard': 95.55506552419355, 'Table': 95.29965958505306}
    test_macc_per_class = {'Airplane': 89.68211677057958, 'Bag': 82.53411455506607, 'Cap': 89.49867437864077, 'Car': 87.0224903619836, 'Chair': 92.74302068612421, 'Earphone': 68.31078982976419, 'Guitar': 94.14587514940838, 'Knife': 93.66488092506658, 'Lamp': 90.31278754375185, 'Laptop': 98.15513383518902, 'Motorbike': 82.98680102771505, 'Mug': 96.17634557571053, 'Pistol': 84.77474579068667, 'Rocket': 69.5738838702578, 'Skateboard': 87.70862139224404, 'Table': 90.84282701991184}
    test_miou_per_class = {'Airplane': 81.71433383063473, 'Bag': 78.57341515759214, 'Cap': 84.65787960229167, 'Car': 78.46661489708228, 'Chair': 86.13013926348177, 'Earphone': 55.42526298722535, 'Guitar': 90.01810406221719, 'Knife': 88.08519094073145, 'Lamp': 82.4485407473712, 'Laptop': 96.44495659836733, 'Motorbike': 72.1415046708371, 'Mug': 94.4974304828809, 'Pistol': 79.97964355553825, 'Rocket': 60.847613104665555, 'Skateboard': 76.01911150214819, 'Table': 84.57170219902143}
==================================================
EPOCH 39 / 100
100%|█████████████████████████████| 876/876 [15:47<00:00,  1.08s/it, data_loading=0.571, iteration=0.227, train_acc=94.88, train_loss_seg=0.117, train_macc=89.82, train_miou=84.36]
Learning rate = 0.000250
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:29<00:00,  2.00it/s, test_acc=93.28, test_loss_seg=0.135, test_macc=87.35, test_miou=80.98]
==================================================
    test_loss_seg = 0.1353352814912796
    test_acc = 93.28559608189924
    test_macc = 87.3499715071235
    test_miou = 80.98760638253789
    test_acc_per_class = {'Airplane': 91.5252417063783, 'Bag': 96.46344866071429, 'Cap': 92.1475497159091, 'Car': 91.9826072982595, 'Chair': 95.00995982776989, 'Earphone': 92.79087611607143, 'Guitar': 96.35355247641509, 'Knife': 92.1063232421875, 'Lamp': 92.02052829982517, 'Laptop': 98.11688158885542, 'Motorbike': 85.66272212009804, 'Mug': 99.4397615131579, 'Pistol': 95.19597833806817, 'Rocket': 82.9345703125, 'Skateboard': 95.49678679435483, 'Table': 95.32274929982312}
    test_macc_per_class = {'Airplane': 89.31818138000617, 'Bag': 84.26332218998596, 'Cap': 86.69105041024714, 'Car': 87.86013469048616, 'Chair': 91.56314828305703, 'Earphone': 71.07660724717591, 'Guitar': 94.15074819362438, 'Knife': 92.12645614791424, 'Lamp': 89.41053595614414, 'Laptop': 98.12254724495146, 'Motorbike': 80.31896494544054, 'Mug': 95.95728188836969, 'Pistol': 87.84739425549608, 'Rocket': 72.27492005296084, 'Skateboard': 86.56324142689732, 'Table': 90.05500980121904}
    test_miou_per_class = {'Airplane': 82.02136922130296, 'Bag': 81.5111735414595, 'Cap': 81.0083226097314, 'Car': 78.39118412073181, 'Chair': 85.82107032484896, 'Earphone': 65.84596469719327, 'Guitar': 90.42008311044715, 'Knife': 85.36561553366818, 'Lamp': 81.89250846989042, 'Laptop': 96.27869771653228, 'Motorbike': 70.0872929649071, 'Mug': 94.49662363091429, 'Pistol': 80.92216764950004, 'Rocket': 61.83925293536993, 'Skateboard': 75.65383176318528, 'Table': 84.2465438309234}
==================================================
EPOCH 40 / 100
100%|█████████████████████████████| 876/876 [15:48<00:00,  1.08s/it, data_loading=0.572, iteration=0.231, train_acc=95.39, train_loss_seg=0.114, train_macc=90.83, train_miou=85.88]
Learning rate = 0.000250
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:30<00:00,  2.00it/s, test_acc=93.33, test_loss_seg=0.096, test_macc=87.12, test_miou=80.91]
==================================================
    test_loss_seg = 0.09682118892669678
    test_acc = 93.3399496902154
    test_macc = 87.12618688077049
    test_miou = 80.91506223808629
    test_acc_per_class = {'Airplane': 91.41956676136364, 'Bag': 96.44949776785714, 'Cap': 91.943359375, 'Car': 91.64977254746836, 'Chair': 95.03284801136364, 'Earphone': 92.00962611607143, 'Guitar': 96.39685288915094, 'Knife': 93.4136962890625, 'Lamp': 91.40420126748252, 'Laptop': 98.16217996987952, 'Motorbike': 87.22522212009804, 'Mug': 99.35109991776315, 'Pistol': 95.53666548295455, 'Rocket': 82.71484375, 'Skateboard': 95.51411290322581, 'Table': 95.21564987470519}
    test_macc_per_class = {'Airplane': 89.2397912630305, 'Bag': 83.8827256640641, 'Cap': 85.56967196532752, 'Car': 82.19177111180215, 'Chair': 91.77833917058938, 'Earphone': 71.2689983527564, 'Guitar': 94.30974927023152, 'Knife': 93.39955294177342, 'Lamp': 88.9389246408982, 'Laptop': 98.10478976644113, 'Motorbike': 79.98996220659137, 'Mug': 94.5647839379295, 'Pistol': 85.82554329717772, 'Rocket': 77.46643347373417, 'Skateboard': 87.35256454971484, 'Table': 90.13538848026582}
    test_miou_per_class = {'Airplane': 81.78825678653105, 'Bag': 81.32486201696794, 'Cap': 80.23656064679783, 'Car': 76.05157055126519, 'Chair': 85.95415329016264, 'Earphone': 64.95603355494943, 'Guitar': 90.49116029179586, 'Knife': 87.63518669124092, 'Lamp': 81.55026575138609, 'Laptop': 96.36205793499381, 'Motorbike': 71.07663446633367, 'Mug': 93.55768058954126, 'Pistol': 80.67325713429466, 'Rocket': 62.984758944343135, 'Skateboard': 76.03308199031679, 'Table': 83.96547516846039}
==================================================
EPOCH 41 / 100
100%|█████████████████████████████| 876/876 [15:47<00:00,  1.08s/it, data_loading=0.570, iteration=0.228, train_acc=95.36, train_loss_seg=0.113, train_macc=89.70, train_miou=84.07]
Learning rate = 0.000250
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:30<00:00,  2.00it/s, test_acc=93.32, test_loss_seg=0.097, test_macc=87.75, test_miou=81.07]
==================================================
    test_loss_seg = 0.09709758311510086
    test_acc = 93.32577508455933
    test_macc = 87.75068264437806
    test_miou = 81.07719762310961
    test_acc_per_class = {'Airplane': 91.48729609604106, 'Bag': 95.61941964285714, 'Cap': 92.41832386363636, 'Car': 91.87073526503164, 'Chair': 95.01481489701705, 'Earphone': 91.11328125, 'Guitar': 96.43677525550315, 'Knife': 92.958984375, 'Lamp': 91.3627144340035, 'Laptop': 98.13335372740963, 'Motorbike': 87.36883425245098, 'Mug': 99.40892269736842, 'Pistol': 95.4134854403409, 'Rocket': 83.33333333333334, 'Skateboard': 95.97246723790323, 'Table': 95.29965958505306}
    test_macc_per_class = {'Airplane': 89.23289364683619, 'Bag': 84.4338588969025, 'Cap': 87.002505505893, 'Car': 86.40860508881731, 'Chair': 92.87547784512319, 'Earphone': 71.30466225306495, 'Guitar': 94.44825945048653, 'Knife': 92.9171995383555, 'Lamp': 89.06774598596772, 'Laptop': 98.10984537489334, 'Motorbike': 82.6191137767716, 'Mug': 95.69738949905071, 'Pistol': 85.58028780969055, 'Rocket': 76.5160783446904, 'Skateboard': 87.44749300229898, 'Table': 90.34950629120657}
    test_miou_per_class = {'Airplane': 81.77379497505214, 'Bag': 78.84671148176555, 'Cap': 81.57631593114627, 'Car': 78.01513477862358, 'Chair': 86.0679043835205, 'Earphone': 63.91438086020067, 'Guitar': 90.46108986537763, 'Knife': 86.81164614842572, 'Lamp': 81.29010810495507, 'Laptop': 96.30857523793959, 'Motorbike': 72.19450773127504, 'Mug': 94.19771031442315, 'Pistol': 80.43945216847987, 'Rocket': 63.570413253538185, 'Skateboard': 77.47816375898654, 'Table': 84.28925297604435}
==================================================
EPOCH 42 / 100
100%|█████████████████████████████| 876/876 [15:48<00:00,  1.08s/it, data_loading=0.566, iteration=0.229, train_acc=95.50, train_loss_seg=0.113, train_macc=92.49, train_miou=87.45]
Learning rate = 0.000250
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:29<00:00,  2.00it/s, test_acc=93.38, test_loss_seg=0.068, test_macc=87.61, test_miou=81.09]
==================================================
    test_loss_seg = 0.06812456995248795
    test_acc = 93.38126958173848
    test_macc = 87.61576947546077
    test_miou = 81.09972830747977
    test_acc_per_class = {'Airplane': 91.30630269428153, 'Bag': 96.22279575892857, 'Cap': 91.59712357954545, 'Car': 91.97086382515823, 'Chair': 95.03284801136364, 'Earphone': 91.259765625, 'Guitar': 96.22856476022012, 'Knife': 93.4423828125, 'Lamp': 92.17469542176573, 'Laptop': 98.17512236445783, 'Motorbike': 87.38319546568627, 'Mug': 99.42048725328947, 'Pistol': 95.27809836647727, 'Rocket': 83.63037109375, 'Skateboard': 95.5534904233871, 'Table': 95.42420585200472}
    test_macc_per_class = {'Airplane': 89.56525251471265, 'Bag': 84.24841347507545, 'Cap': 85.77168655551617, 'Car': 88.16955379496929, 'Chair': 92.94691480843983, 'Earphone': 71.45449986754642, 'Guitar': 93.73531381687282, 'Knife': 93.41926308107895, 'Lamp': 89.86408064217846, 'Laptop': 98.11560452319328, 'Motorbike': 80.70022621292581, 'Mug': 96.14433019404916, 'Pistol': 84.87810731741232, 'Rocket': 74.34272151948909, 'Skateboard': 87.8238404710642, 'Table': 90.6725028128485}
    test_miou_per_class = {'Airplane': 81.82798689644471, 'Bag': 80.6876529167239, 'Cap': 79.75716020303656, 'Car': 78.5130480130221, 'Chair': 86.01744813082675, 'Earphone': 64.15821172383389, 'Guitar': 90.1415203459045, 'Knife': 87.67978146700302, 'Lamp': 82.54916998885274, 'Laptop': 96.38704629956634, 'Motorbike': 71.03906111265627, 'Mug': 94.34810555811409, 'Pistol': 79.7168309597929, 'Rocket': 63.3969921969162, 'Skateboard': 76.5020556979478, 'Table': 84.87358140903449}
==================================================
EPOCH 43 / 100
100%|█████████████████████████████| 876/876 [15:44<00:00,  1.08s/it, data_loading=0.572, iteration=0.229, train_acc=95.98, train_loss_seg=0.107, train_macc=90.71, train_miou=85.93]
Learning rate = 0.000125
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:30<00:00,  2.00it/s, test_acc=93.14, test_loss_seg=0.215, test_macc=87.26, test_miou=80.80]
==================================================
    test_loss_seg = 0.21586279571056366
    test_acc = 93.14783191486194
    test_macc = 87.26379019871194
    test_miou = 80.8070511707846
    test_acc_per_class = {'Airplane': 91.50018328445748, 'Bag': 96.59946986607143, 'Cap': 91.62375710227273, 'Car': 91.98971518987342, 'Chair': 94.95010375976562, 'Earphone': 88.14871651785714, 'Guitar': 96.40790831367924, 'Knife': 93.4130859375, 'Lamp': 91.28878933566433, 'Laptop': 98.16570971385542, 'Motorbike': 87.20320159313727, 'Mug': 99.39221833881578, 'Pistol': 95.4134854403409, 'Rocket': 83.80940755208334, 'Skateboard': 95.07623487903226, 'Table': 95.38332381338444}
    test_macc_per_class = {'Airplane': 89.1357133416585, 'Bag': 84.51813176213902, 'Cap': 85.31466271486369, 'Car': 87.018471028356, 'Chair': 91.9578803542641, 'Earphone': 70.29757907822312, 'Guitar': 94.42736969104251, 'Knife': 93.39540712555545, 'Lamp': 89.30940170745959, 'Laptop': 98.13966019773909, 'Motorbike': 82.06195745310765, 'Mug': 95.90899576150818, 'Pistol': 86.77821260659823, 'Rocket': 73.33974071482608, 'Skateboard': 84.01973395883319, 'Table': 90.59772568321664}
    test_miou_per_class = {'Airplane': 81.87544650534, 'Bag': 82.07935485286893, 'Cap': 79.61546702150153, 'Car': 78.27749042373556, 'Chair': 85.79818532013434, 'Earphone': 59.904316780487534, 'Guitar': 90.5473252749178, 'Knife': 87.6320019974454, 'Lamp': 81.43954437431489, 'Laptop': 96.3711948444136, 'Motorbike': 71.62303349588308, 'Mug': 94.07647220802077, 'Pistol': 81.17544453191627, 'Rocket': 63.306316914176705, 'Skateboard': 74.56789622886456, 'Table': 84.62332795853258}
==================================================
EPOCH 44 / 100
100%|█████████████████████████████| 876/876 [15:48<00:00,  1.08s/it, data_loading=0.571, iteration=0.224, train_acc=95.50, train_loss_seg=0.107, train_macc=91.51, train_miou=86.89]
Learning rate = 0.000125
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:30<00:00,  2.00it/s, test_acc=93.45, test_loss_seg=0.071, test_macc=87.29, test_miou=81.13]
==================================================
    test_loss_seg = 0.07122541964054108
    test_acc = 93.45551160924853
    test_macc = 87.29497962189336
    test_miou = 81.13075927440083
    test_acc_per_class = {'Airplane': 91.43803839809385, 'Bag': 96.35532924107143, 'Cap': 91.845703125, 'Car': 92.20727848101265, 'Chair': 95.07286765358664, 'Earphone': 92.5048828125, 'Guitar': 96.52644703223271, 'Knife': 93.34228515625, 'Lamp': 92.0041384396853, 'Laptop': 98.14158979668674, 'Motorbike': 87.02895220588235, 'Mug': 99.38450863486842, 'Pistol': 95.64985795454545, 'Rocket': 83.349609375, 'Skateboard': 95.09828629032258, 'Table': 95.3384111512382}
    test_macc_per_class = {'Airplane': 88.84612049066982, 'Bag': 85.56007610094308, 'Cap': 85.6192043695282, 'Car': 87.51227147505841, 'Chair': 92.66212084371934, 'Earphone': 71.34738144240855, 'Guitar': 94.80739962151885, 'Knife': 93.34133170891712, 'Lamp': 89.01987897521263, 'Laptop': 98.12591991893007, 'Motorbike': 81.51684584553348, 'Mug': 96.32256517941187, 'Pistol': 86.7560937570455, 'Rocket': 71.04640385969375, 'Skateboard': 83.70172373526961, 'Table': 90.53433662643347}
    test_miou_per_class = {'Airplane': 81.79130526454955, 'Bag': 81.59513901981907, 'Cap': 80.09730015038691, 'Car': 78.75500792080774, 'Chair': 86.06041823876313, 'Earphone': 65.52895474695842, 'Guitar': 90.81936886155347, 'Knife': 87.51468764610927, 'Lamp': 81.82879756900351, 'Laptop': 96.32511663375469, 'Motorbike': 71.48599271866013, 'Mug': 94.05710307207306, 'Pistol': 81.40607518022294, 'Rocket': 62.04210710580309, 'Skateboard': 74.37728702455884, 'Table': 84.40748723738962}
==================================================
EPOCH 45 / 100
100%|█████████████████████████████| 876/876 [15:41<00:00,  1.07s/it, data_loading=0.565, iteration=0.221, train_acc=95.64, train_loss_seg=0.110, train_macc=92.22, train_miou=87.43]
Learning rate = 0.000125
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.09it/s, test_acc=93.37, test_loss_seg=0.209, test_macc=87.39, test_miou=81.07]
==================================================
    test_loss_seg = 0.20905061066150665
    test_acc = 93.37981745699196
    test_macc = 87.38998537352931
    test_miou = 81.07736100139486
    test_acc_per_class = {'Airplane': 91.55975073313783, 'Bag': 96.46693638392857, 'Cap': 91.22869318181817, 'Car': 92.18626384493672, 'Chair': 95.14624855735086, 'Earphone': 91.455078125, 'Guitar': 96.51477741745283, 'Knife': 93.7109375, 'Lamp': 91.68231670673077, 'Laptop': 98.18806475903614, 'Motorbike': 87.61584712009804, 'Mug': 99.43590666118422, 'Pistol': 95.53999467329545, 'Rocket': 82.99153645833334, 'Skateboard': 95.00693044354838, 'Table': 95.34779674602005}
    test_macc_per_class = {'Airplane': 89.8724395013389, 'Bag': 84.444219056934, 'Cap': 85.0556889799006, 'Car': 86.83176788035303, 'Chair': 92.51417810825826, 'Earphone': 70.97941450500976, 'Guitar': 94.95362241423999, 'Knife': 93.69337661696815, 'Lamp': 89.13246727336997, 'Laptop': 98.12793451501744, 'Motorbike': 82.4450058466836, 'Mug': 95.42160063429267, 'Pistol': 86.30249847284904, 'Rocket': 73.92765976707364, 'Skateboard': 84.23502047716792, 'Table': 90.30287192701208}
    test_miou_per_class = {'Airplane': 82.12163626041567, 'Bag': 81.58795434626064, 'Cap': 78.88350456437747, 'Car': 78.60826054253049, 'Chair': 86.2963273042356, 'Earphone': 64.13171983856815, 'Guitar': 90.8128715552976, 'Knife': 88.15818703548601, 'Lamp': 81.3154173707219, 'Laptop': 96.41215468404218, 'Motorbike': 72.08748146812057, 'Mug': 94.40449031912016, 'Pistol': 81.43016177513931, 'Rocket': 62.35069927085194, 'Skateboard': 74.2742806872763, 'Table': 84.36262899987369}
==================================================
EPOCH 46 / 100
100%|█████████████████████████████| 876/876 [15:24<00:00,  1.06s/it, data_loading=0.560, iteration=0.226, train_acc=95.47, train_loss_seg=0.107, train_macc=91.31, train_miou=86.16]
Learning rate = 0.000125
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.08it/s, test_acc=93.35, test_loss_seg=0.148, test_macc=87.56, test_miou=81.11]
==================================================
    test_loss_seg = 0.14810775220394135
    test_acc = 93.35569559738524
    test_macc = 87.56439900327129
    test_miou = 81.11513364106325
    test_acc_per_class = {'Airplane': 91.33937981121701, 'Bag': 96.40066964285714, 'Cap': 90.99343039772727, 'Car': 92.03576196598101, 'Chair': 95.06780450994317, 'Earphone': 92.28515625, 'Guitar': 96.55040045204403, 'Knife': 93.0279541015625, 'Lamp': 91.81326486013987, 'Laptop': 98.0968797063253, 'Motorbike': 87.12852328431373, 'Mug': 99.4140625, 'Pistol': 95.4378995028409, 'Rocket': 83.59781901041666, 'Skateboard': 95.166015625, 'Table': 95.33610793779481}
    test_macc_per_class = {'Airplane': 89.97317632746608, 'Bag': 84.03444646316807, 'Cap': 84.49390471831025, 'Car': 85.63759817022752, 'Chair': 92.64630710449849, 'Earphone': 71.27549290478225, 'Guitar': 95.11368247361435, 'Knife': 93.02249782194711, 'Lamp': 89.5527070902113, 'Laptop': 98.01310114356366, 'Motorbike': 84.26902379874343, 'Mug': 96.03653117941487, 'Pistol': 86.51580072165108, 'Rocket': 75.5396930409615, 'Skateboard': 84.37230920679059, 'Table': 90.53411188699036}
    test_miou_per_class = {'Airplane': 81.78549483782712, 'Bag': 81.21147757299278, 'Cap': 78.28122190954106, 'Car': 78.05794431532433, 'Chair': 86.20153478297254, 'Earphone': 65.29606976274373, 'Guitar': 90.85607566475004, 'Knife': 86.96232271941938, 'Lamp': 81.63388865900927, 'Laptop': 96.23322205596902, 'Motorbike': 73.13578689152611, 'Mug': 94.28035004686883, 'Pistol': 80.80890633930018, 'Rocket': 64.04267206827753, 'Skateboard': 74.61627953746688, 'Table': 84.43889109302319}
==================================================
EPOCH 47 / 100
100%|█████████████████████████████| 876/876 [15:26<00:00,  1.06s/it, data_loading=0.564, iteration=0.223, train_acc=95.91, train_loss_seg=0.105, train_macc=92.39, train_miou=87.64]
Learning rate = 0.000125
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.08it/s, test_acc=93.39, test_loss_seg=0.121, test_macc=87.65, test_miou=81.29]
==================================================
    test_loss_seg = 0.12167800217866898
    test_acc = 93.39537527749485
    test_macc = 87.65639224525272
    test_miou = 81.29817841668648
    test_acc_per_class = {'Airplane': 91.48758247800586, 'Bag': 96.49135044642857, 'Cap': 91.99662642045455, 'Car': 92.20326097705697, 'Chair': 95.06454467773438, 'Earphone': 91.58761160714286, 'Guitar': 96.49205237814465, 'Knife': 92.6202392578125, 'Lamp': 91.97084653627621, 'Laptop': 98.07099491716868, 'Motorbike': 87.06629136029412, 'Mug': 99.42691200657895, 'Pistol': 95.54887251420455, 'Rocket': 83.29671223958334, 'Skateboard': 95.59759324596774, 'Table': 95.40451337706368}
    test_macc_per_class = {'Airplane': 89.45416522046395, 'Bag': 85.12890378935542, 'Cap': 86.15512738988768, 'Car': 86.27939102438143, 'Chair': 92.83669345885028, 'Earphone': 71.0501447539857, 'Guitar': 94.5590013554026, 'Knife': 92.61397922715094, 'Lamp': 89.33551122024805, 'Laptop': 98.05460642472656, 'Motorbike': 82.47103684494476, 'Mug': 96.47254879145764, 'Pistol': 87.78608977785458, 'Rocket': 75.10286222058134, 'Skateboard': 84.68741363701618, 'Table': 90.51480078773656}
    test_miou_per_class = {'Airplane': 82.01931989187621, 'Bag': 81.91256762537272, 'Cap': 80.55544632067806, 'Car': 78.53845087247494, 'Chair': 86.22394552877041, 'Earphone': 64.32157828064375, 'Guitar': 90.67221616446103, 'Knife': 86.25202456835301, 'Lamp': 81.94206767325947, 'Laptop': 96.18818435502496, 'Motorbike': 72.42398061838334, 'Mug': 94.43933680634318, 'Pistol': 81.66082283285111, 'Rocket': 63.09944043758991, 'Skateboard': 75.80134340810613, 'Table': 84.7201292827957}
==================================================
EPOCH 48 / 100
100%|█████████████████████████████| 876/876 [15:28<00:00,  1.06s/it, data_loading=0.569, iteration=0.229, train_acc=95.88, train_loss_seg=0.109, train_macc=92.51, train_miou=87.64]
Learning rate = 0.000125
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:28<00:00,  2.04it/s, test_acc=93.22, test_loss_seg=0.158, test_macc=87.94, test_miou=80.96]
==================================================
    test_loss_seg = 0.15843454003334045
    test_acc = 93.22569155887729
    test_macc = 87.94394212582242
    test_miou = 80.96089920703717
    test_acc_per_class = {'Airplane': 91.4672493585044, 'Bag': 96.27511160714286, 'Cap': 91.3662997159091, 'Car': 92.1414532238924, 'Chair': 95.08445046164773, 'Earphone': 91.41671316964286, 'Guitar': 96.37443494496856, 'Knife': 93.031005859375, 'Lamp': 91.69529201267483, 'Laptop': 98.01804875753012, 'Motorbike': 87.39372702205883, 'Mug': 99.44618626644737, 'Pistol': 95.3280362215909, 'Rocket': 82.11263020833334, 'Skateboard': 95.22744455645162, 'Table': 95.23298155586674}
    test_macc_per_class = {'Airplane': 89.49552112839663, 'Bag': 84.82935757984144, 'Cap': 84.84333492556854, 'Car': 88.29431652453803, 'Chair': 92.54062402324074, 'Earphone': 71.61747968051588, 'Guitar': 94.29274816099148, 'Knife': 93.00862915455482, 'Lamp': 89.05009483758131, 'Laptop': 97.99410071490693, 'Motorbike': 85.43968969175896, 'Mug': 96.9235571727561, 'Pistol': 85.29156553868646, 'Rocket': 77.39148674460186, 'Skateboard': 86.00719640173905, 'Table': 90.08337173348004}
    test_miou_per_class = {'Airplane': 81.98618178833947, 'Bag': 81.07112085487826, 'Cap': 79.01033846986377, 'Car': 78.76704623661456, 'Chair': 86.21050181575089, 'Earphone': 64.23632911283535, 'Guitar': 90.46537556877549, 'Knife': 86.95797713740754, 'Lamp': 81.42090450287893, 'Laptop': 96.08507790754814, 'Motorbike': 73.35077301166585, 'Mug': 94.65362679907044, 'Pistol': 79.98599510681265, 'Rocket': 62.04592486399335, 'Skateboard': 75.11498117285788, 'Table': 84.01223296330232}
==================================================
EPOCH 49 / 100
100%|█████████████████████████████| 876/876 [15:24<00:00,  1.06s/it, data_loading=0.560, iteration=0.224, train_acc=95.79, train_loss_seg=0.105, train_macc=91.84, train_miou=87.28]
Learning rate = 0.000125
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.08it/s, test_acc=93.44, test_loss_seg=0.066, test_macc=87.74, test_miou=81.24]
==================================================
    test_loss_seg = 0.06631797552108765
    test_acc = 93.44060345504438
    test_macc = 87.74822844304096
    test_miou = 81.2493094922331
    test_acc_per_class = {'Airplane': 91.44118859970675, 'Bag': 96.14606584821429, 'Cap': 91.03781960227273, 'Car': 92.16895767405063, 'Chair': 95.0223749334162, 'Earphone': 92.15611049107143, 'Guitar': 96.57404677672956, 'Knife': 93.5748291015625, 'Lamp': 91.58824573863636, 'Laptop': 97.99039909638554, 'Motorbike': 87.01267616421569, 'Mug': 99.39221833881578, 'Pistol': 95.57550603693183, 'Rocket': 84.24886067708334, 'Skateboard': 95.79133064516128, 'Table': 95.32902555645637}
    test_macc_per_class = {'Airplane': 89.27457489531396, 'Bag': 83.47490207622381, 'Cap': 84.12291391522987, 'Car': 87.99498492512188, 'Chair': 92.94119896778858, 'Earphone': 71.44912799026463, 'Guitar': 94.79544305543568, 'Knife': 93.56707171605484, 'Lamp': 88.48909361612513, 'Laptop': 98.00822073678128, 'Motorbike': 84.57616217703077, 'Mug': 96.58185554050307, 'Pistol': 86.41053357002706, 'Rocket': 74.4167806770052, 'Skateboard': 87.16826340192826, 'Table': 90.7005278278213}
    test_miou_per_class = {'Airplane': 81.83332091685963, 'Bag': 80.14595717376332, 'Cap': 78.18750126974496, 'Car': 78.7752310650431, 'Chair': 86.02733393513773, 'Earphone': 65.33976332644504, 'Guitar': 90.90925917240838, 'Knife': 87.92236873763463, 'Lamp': 80.84628521585107, 'Laptop': 96.03470136561087, 'Motorbike': 72.83280936897685, 'Mug': 94.15220319743915, 'Pistol': 80.99760292892256, 'Rocket': 64.48105968454922, 'Skateboard': 77.07173123896273, 'Table': 84.43182327838062}
==================================================
EPOCH 50 / 100
100%|█████████████████████████████| 876/876 [15:24<00:00,  1.05s/it, data_loading=0.567, iteration=0.224, train_acc=95.91, train_loss_seg=0.103, train_macc=91.89, train_miou=86.97]
Learning rate = 0.000125
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.08it/s, test_acc=93.27, test_loss_seg=0.142, test_macc=87.83, test_miou=80.99]
==================================================
    test_loss_seg = 0.14202098548412323
    test_acc = 93.27067085336823
    test_macc = 87.8356143238222
    test_miou = 80.99377248218505
    test_acc_per_class = {'Airplane': 91.43832478005865, 'Bag': 96.09375, 'Cap': 90.8114346590909, 'Car': 92.11363973496836, 'Chair': 95.0284090909091, 'Earphone': 91.75153459821429, 'Guitar': 96.54118759827044, 'Knife': 92.89306640625, 'Lamp': 91.9032383631993, 'Laptop': 97.84273814006023, 'Motorbike': 87.1026731004902, 'Mug': 99.4397615131579, 'Pistol': 95.60657848011364, 'Rocket': 83.17464192708334, 'Skateboard': 95.37865423387096, 'Table': 95.21110102815447}
    test_macc_per_class = {'Airplane': 89.87100714397559, 'Bag': 83.86328019848277, 'Cap': 83.82062373348359, 'Car': 88.08155260135965, 'Chair': 92.57802572625698, 'Earphone': 71.65184081628027, 'Guitar': 94.8216935347686, 'Knife': 92.89935047221103, 'Lamp': 88.57623574311047, 'Laptop': 97.85255968680238, 'Motorbike': 82.3668532606009, 'Mug': 96.13129734845458, 'Pistol': 87.62635552536656, 'Rocket': 77.56884215103614, 'Skateboard': 86.3780614540886, 'Table': 91.28224978487728}
    test_miou_per_class = {'Airplane': 81.9494124663798, 'Bag': 80.12027792374614, 'Cap': 77.70703974986984, 'Car': 78.67497208152965, 'Chair': 86.03908367369752, 'Earphone': 64.68606481686045, 'Guitar': 90.85978321040471, 'Knife': 86.72922696336092, 'Lamp': 81.1764651404967, 'Laptop': 95.74903077256387, 'Motorbike': 72.39481733888812, 'Mug': 94.51516788189755, 'Pistol': 81.61505982384524, 'Rocket': 64.01157415210635, 'Skateboard': 75.45989562938695, 'Table': 84.21248808992705}
==================================================
EPOCH 51 / 100
100%|█████████████████████████████| 876/876 [15:24<00:00,  1.06s/it, data_loading=0.559, iteration=0.226, train_acc=95.89, train_loss_seg=0.107, train_macc=92.70, train_miou=88.00]
Learning rate = 0.000125
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.08it/s, test_acc=93.25, test_loss_seg=0.070, test_macc=87.09, test_miou=80.79]
==================================================
    test_loss_seg = 0.07092564553022385
    test_acc = 93.25430206546478
    test_macc = 87.09881206503472
    test_miou = 80.79540499953775
    test_acc_per_class = {'Airplane': 91.51006346224341, 'Bag': 96.240234375, 'Cap': 90.83362926136364, 'Car': 92.11085838607595, 'Chair': 95.01301158558239, 'Earphone': 91.54575892857143, 'Guitar': 96.55377849842768, 'Knife': 92.342529296875, 'Lamp': 91.7045113090035, 'Laptop': 98.13276543674698, 'Motorbike': 86.78959865196079, 'Mug': 99.43076685855263, 'Pistol': 95.67205255681817, 'Rocket': 83.642578125, 'Skateboard': 95.29517389112904, 'Table': 95.25152242408609}
    test_macc_per_class = {'Airplane': 89.09042117852562, 'Bag': 84.48182857572155, 'Cap': 84.12088285919548, 'Car': 86.53671553287067, 'Chair': 92.22069122939186, 'Earphone': 71.7347874298771, 'Guitar': 94.78381712698295, 'Knife': 92.35543456705526, 'Lamp': 88.55395207395534, 'Laptop': 98.08778694918266, 'Motorbike': 82.14802570959759, 'Mug': 95.81332388628508, 'Pistol': 86.78213551332551, 'Rocket': 71.80465422111607, 'Skateboard': 85.18942417129813, 'Table': 89.8771120161749}
    test_miou_per_class = {'Airplane': 81.93912423911743, 'Bag': 80.8303752678565, 'Cap': 77.87446919598364, 'Car': 78.24600804082435, 'Chair': 85.86946023399253, 'Earphone': 64.67612270797927, 'Guitar': 90.89601936001893, 'Knife': 85.77406466809845, 'Lamp': 81.03116646016376, 'Laptop': 96.30586226929863, 'Motorbike': 71.44159534115815, 'Mug': 94.40198410855774, 'Pistol': 81.68131824732355, 'Rocket': 62.51116306129815, 'Skateboard': 75.24470913181963, 'Table': 84.0030376591133}
==================================================
EPOCH 52 / 100
100%|█████████████████████████████| 876/876 [15:24<00:00,  1.06s/it, data_loading=0.560, iteration=0.229, train_acc=95.71, train_loss_seg=0.103, train_macc=92.35, train_miou=87.57]
Learning rate = 0.000125
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.08it/s, test_acc=93.35, test_loss_seg=0.071, test_macc=87.56, test_miou=81.04]
==================================================
    test_loss_seg = 0.07174931466579437
    test_acc = 93.3595435434435
    test_macc = 87.56577536826875
    test_miou = 81.04646629347569
    test_acc_per_class = {'Airplane': 91.51980044904691, 'Bag': 96.04143415178571, 'Cap': 91.41956676136364, 'Car': 92.19460789161393, 'Chair': 95.06787386807528, 'Earphone': 91.53180803571429, 'Guitar': 96.50341489779875, 'Knife': 92.7862548828125, 'Lamp': 91.6958041958042, 'Laptop': 98.17865210843374, 'Motorbike': 87.4444699754902, 'Mug': 99.43076685855263, 'Pistol': 95.6265536221591, 'Rocket': 84.04134114583334, 'Skateboard': 94.90297379032258, 'Table': 95.36737406028891}
    test_macc_per_class = {'Airplane': 89.95087545491901, 'Bag': 84.20692037180213, 'Cap': 85.20175290804349, 'Car': 86.88851862780642, 'Chair': 92.6044415205628, 'Earphone': 71.5230664659375, 'Guitar': 94.74581115238342, 'Knife': 92.77594118168253, 'Lamp': 89.11827066835927, 'Laptop': 98.11895806499338, 'Motorbike': 83.47524322213789, 'Mug': 96.5441888186416, 'Pistol': 86.16019987535263, 'Rocket': 75.66328287413154, 'Skateboard': 83.31229555086193, 'Table': 90.76263913468439}
    test_miou_per_class = {'Airplane': 82.06123078431214, 'Bag': 80.07793531910863, 'Cap': 79.24402761633843, 'Car': 78.60705542080362, 'Chair': 86.0838817054356, 'Earphone': 64.30912128258902, 'Guitar': 90.77909771654843, 'Knife': 86.53870270682711, 'Lamp': 81.62904582302875, 'Laptop': 96.39389265171945, 'Motorbike': 72.81202667312535, 'Mug': 94.48040977161791, 'Pistol': 81.15242788035229, 'Rocket': 64.50610819141362, 'Skateboard': 73.45778693520938, 'Table': 84.61071021718138}
==================================================
EPOCH 53 / 100
100%|█████████████████████████████| 876/876 [15:26<00:00,  1.06s/it, data_loading=0.561, iteration=0.223, train_acc=95.27, train_loss_seg=0.109, train_macc=92.06, train_miou=86.82]
Learning rate = 0.000125
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:28<00:00,  2.04it/s, test_acc=93.40, test_loss_seg=0.083, test_macc=87.83, test_miou=81.23]
==================================================
    test_loss_seg = 0.0837961956858635
    test_acc = 93.40027782393105
    test_macc = 87.83364204876779
    test_miou = 81.23142192067982
    test_acc_per_class = {'Airplane': 91.52123235887096, 'Bag': 96.20186941964286, 'Cap': 91.22869318181817, 'Car': 92.26846815664557, 'Chair': 95.05760886452414, 'Earphone': 92.05496651785714, 'Guitar': 96.45612224842768, 'Knife': 93.48388671875, 'Lamp': 91.54778327141608, 'Laptop': 98.15394390060241, 'Motorbike': 87.28553921568627, 'Mug': 99.42434210526315, 'Pistol': 95.60324928977273, 'Rocket': 83.49609375, 'Skateboard': 95.24319556451613, 'Table': 95.37745061910378}
    test_macc_per_class = {'Airplane': 89.59398019426834, 'Bag': 85.28062852319545, 'Cap': 84.52241498802603, 'Car': 87.01600490486159, 'Chair': 92.75033588783408, 'Earphone': 72.05975989154702, 'Guitar': 94.67357516198763, 'Knife': 93.45909748455101, 'Lamp': 89.68129612815532, 'Laptop': 98.10720109028294, 'Motorbike': 83.24783022020394, 'Mug': 96.21597022123312, 'Pistol': 87.03760405309134, 'Rocket': 77.57758885878224, 'Skateboard': 83.63407659638695, 'Table': 90.48090857587772}
    test_miou_per_class = {'Airplane': 81.95773880502645, 'Bag': 80.98661296175902, 'Cap': 78.65768057736122, 'Car': 78.74991881268437, 'Chair': 86.16496045844615, 'Earphone': 65.34265666229584, 'Guitar': 90.66638312522616, 'Knife': 87.7516571937474, 'Lamp': 81.59466468187154, 'Laptop': 96.34684942416915, 'Motorbike': 72.36834956573702, 'Mug': 94.38945298589975, 'Pistol': 81.27890890730552, 'Rocket': 64.35961792134917, 'Skateboard': 74.58497650343836, 'Table': 84.50232214455981}
==================================================
EPOCH 54 / 100
100%|█████████████████████████████| 876/876 [15:35<00:00,  1.07s/it, data_loading=0.562, iteration=0.234, train_acc=96.08, train_loss_seg=0.103, train_macc=92.72, train_miou=88.07]
Learning rate = 0.000125
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:28<00:00,  2.03it/s, test_acc=93.30, test_loss_seg=0.114, test_macc=87.34, test_miou=80.93]
==================================================
    test_loss_seg = 0.11489348858594894
    test_acc = 93.3093899450089
    test_macc = 87.34406925148981
    test_miou = 80.93545383221934
    test_acc_per_class = {'Airplane': 91.4794205920088, 'Bag': 96.39020647321429, 'Cap': 91.5926846590909, 'Car': 92.14361649525317, 'Chair': 95.06620927290483, 'Earphone': 90.50641741071429, 'Guitar': 96.3400402908805, 'Knife': 93.3734130859375, 'Lamp': 91.4760776333042, 'Laptop': 98.12217620481928, 'Motorbike': 87.66850490196079, 'Mug': 99.44618626644737, 'Pistol': 95.30584161931817, 'Rocket': 84.06168619791666, 'Skateboard': 94.70136088709677, 'Table': 95.27639712927476}
    test_macc_per_class = {'Airplane': 89.82669080429005, 'Bag': 84.35668954182307, 'Cap': 86.02209523257403, 'Car': 86.74178585859894, 'Chair': 92.65185208016976, 'Earphone': 71.23748562240054, 'Guitar': 94.22580628876601, 'Knife': 93.35868172845029, 'Lamp': 89.35227611881011, 'Laptop': 98.10621274313621, 'Motorbike': 81.73692812822848, 'Mug': 95.84466132022979, 'Pistol': 85.5923616072256, 'Rocket': 73.32538789970499, 'Skateboard': 84.06532864525035, 'Table': 91.060864404179}
    test_miou_per_class = {'Airplane': 82.13824264309179, 'Bag': 81.29212455616572, 'Cap': 79.84852188337643, 'Car': 78.45049164898714, 'Chair': 86.15107328361684, 'Earphone': 63.11930988759696, 'Guitar': 90.38139388764705, 'Knife': 87.56393656723786, 'Lamp': 81.31459288003569, 'Laptop': 96.28743429447728, 'Motorbike': 71.83322968210068, 'Mug': 94.54125219767494, 'Pistol': 80.6675891146952, 'Rocket': 63.58058583389802, 'Skateboard': 73.44038651796396, 'Table': 84.35709643694379}
==================================================
EPOCH 55 / 100
100%|█████████████████████████████| 876/876 [15:36<00:00,  1.07s/it, data_loading=0.570, iteration=0.229, train_acc=95.70, train_loss_seg=0.103, train_macc=92.05, train_miou=87.56]
Learning rate = 0.000125
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:29<00:00,  2.02it/s, test_acc=93.32, test_loss_seg=0.104, test_macc=87.35, test_miou=81.05]
==================================================
    test_loss_seg = 0.10475680977106094
    test_acc = 93.32453934951818
    test_macc = 87.35033862095706
    test_miou = 81.05043780840859
    test_acc_per_class = {'Airplane': 91.53340359237536, 'Bag': 96.40764508928571, 'Cap': 91.2198153409091, 'Car': 92.13650860363924, 'Chair': 95.08888938210227, 'Earphone': 90.52036830357143, 'Guitar': 96.43431849449685, 'Knife': 93.3819580078125, 'Lamp': 91.40317690122379, 'Laptop': 98.10746893825302, 'Motorbike': 87.18022365196079, 'Mug': 99.44618626644737, 'Pistol': 95.63432173295455, 'Rocket': 84.18375651041666, 'Skateboard': 95.12821320564517, 'Table': 95.38637557119694}
    test_macc_per_class = {'Airplane': 89.77961590205196, 'Bag': 85.30591182550296, 'Cap': 84.53747781488967, 'Car': 86.27241604871301, 'Chair': 92.28909665466655, 'Earphone': 71.38333772705559, 'Guitar': 94.46948009089819, 'Knife': 93.36343414243726, 'Lamp': 89.27837054973081, 'Laptop': 98.09589305242056, 'Motorbike': 83.03609529848973, 'Mug': 96.5175210992247, 'Pistol': 87.81514066068334, 'Rocket': 71.52190888961462, 'Skateboard': 83.68220071199907, 'Table': 90.2575174669351}
    test_miou_per_class = {'Airplane': 82.13499975782516, 'Bag': 81.68557620878822, 'Cap': 78.65034055955331, 'Car': 78.36535609640467, 'Chair': 86.21664953217775, 'Earphone': 63.078211923938355, 'Guitar': 90.60501804306492, 'Knife': 87.57663668850978, 'Lamp': 81.27472321161096, 'Laptop': 96.25922876125546, 'Motorbike': 72.3276664009819, 'Mug': 94.61191743573477, 'Pistol': 81.97248585662635, 'Rocket': 63.094875569255535, 'Skateboard': 74.44093609875728, 'Table': 84.51238279005285}
==================================================
EPOCH 56 / 100
100%|█████████████████████████████| 876/876 [15:37<00:00,  1.07s/it, data_loading=0.565, iteration=0.230, train_acc=95.52, train_loss_seg=0.099, train_macc=91.84, train_miou=87.22]
Learning rate = 0.000125
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:28<00:00,  2.03it/s, test_acc=93.47, test_loss_seg=0.170, test_macc=87.91, test_miou=81.38]
==================================================
    test_loss_seg = 0.17049561440944672
    test_acc = 93.47973066652233
    test_macc = 87.91379849874247
    test_miou = 81.38508807159097
    test_acc_per_class = {'Airplane': 91.60084654508798, 'Bag': 96.3623046875, 'Cap': 91.29527698863636, 'Car': 92.21005982990506, 'Chair': 95.03090598366477, 'Earphone': 92.07938058035714, 'Guitar': 96.52767541273585, 'Knife': 93.2769775390625, 'Lamp': 91.42810314685315, 'Laptop': 98.05511106927712, 'Motorbike': 87.25394454656863, 'Mug': 99.42691200657895, 'Pistol': 95.59326171875, 'Rocket': 84.4970703125, 'Skateboard': 95.64169606854838, 'Table': 95.39616422833137}
    test_macc_per_class = {'Airplane': 89.37553842962157, 'Bag': 83.96831273114279, 'Cap': 84.88463843277356, 'Car': 88.07898466328095, 'Chair': 93.07200402125095, 'Earphone': 71.89347256901611, 'Guitar': 94.82483092774173, 'Knife': 93.27874008700499, 'Lamp': 88.31928677013046, 'Laptop': 98.04289951162029, 'Motorbike': 82.01864437252725, 'Mug': 95.99690653389229, 'Pistol': 88.76118787235113, 'Rocket': 75.93427848318211, 'Skateboard': 87.28773530425022, 'Table': 90.88331527009296}
    test_miou_per_class = {'Airplane': 82.09484509600347, 'Bag': 81.05581407858054, 'Cap': 78.91618511981233, 'Car': 78.78685971018203, 'Chair': 85.85568940376274, 'Earphone': 65.25051183581648, 'Guitar': 90.7937725889489, 'Knife': 87.4004659761614, 'Lamp': 80.72760191350166, 'Laptop': 96.15772174475109, 'Motorbike': 71.9473666061127, 'Mug': 94.388149222982, 'Pistol': 82.22486083007755, 'Rocket': 65.20315983137539, 'Skateboard': 76.68323635452961, 'Table': 84.6751688328578}
==================================================
EPOCH 57 / 100
100%|█████████████████████████████| 876/876 [15:31<00:00,  1.06s/it, data_loading=0.560, iteration=0.222, train_acc=95.63, train_loss_seg=0.105, train_macc=91.17, train_miou=86.47]
Learning rate = 0.000125
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.08it/s, test_acc=93.36, test_loss_seg=0.126, test_macc=87.51, test_miou=81.06]
==================================================
    test_loss_seg = 0.12660755217075348
    test_acc = 93.36472935399682
    test_macc = 87.51534495748851
    test_miou = 81.06212253666911
    test_acc_per_class = {'Airplane': 91.52209150476538, 'Bag': 96.240234375, 'Cap': 91.19762073863636, 'Car': 92.12167474287975, 'Chair': 95.11455189098011, 'Earphone': 92.27120535714286, 'Guitar': 96.56759777908806, 'Knife': 93.2623291015625, 'Lamp': 91.50117460664336, 'Laptop': 97.94863045933735, 'Motorbike': 86.93895526960785, 'Mug': 99.42305715460526, 'Pistol': 95.64652876420455, 'Rocket': 84.02506510416666, 'Skateboard': 94.79744203629032, 'Table': 95.25751077903891}
    test_macc_per_class = {'Airplane': 89.27392459766794, 'Bag': 84.21340088208389, 'Cap': 84.95001065851507, 'Car': 87.89771264888911, 'Chair': 92.60179940278309, 'Earphone': 71.76946242962913, 'Guitar': 94.8241922450697, 'Knife': 93.25493381633112, 'Lamp': 89.09306678589238, 'Laptop': 97.96488375869544, 'Motorbike': 83.98278928849717, 'Mug': 96.92295514452833, 'Pistol': 86.28074347810644, 'Rocket': 74.54361759493787, 'Skateboard': 81.9313693098231, 'Table': 90.74065727836643}
    test_miou_per_class = {'Airplane': 81.95501813085801, 'Bag': 80.73302841167748, 'Cap': 78.7909439270815, 'Car': 78.70027624055551, 'Chair': 86.18020943281498, 'Earphone': 65.3925938824057, 'Guitar': 90.8990413393601, 'Knife': 87.37220480932446, 'Lamp': 81.43344238840471, 'Laptop': 95.95386862468517, 'Motorbike': 72.41370856092216, 'Mug': 94.45370755955771, 'Pistol': 81.29231172975273, 'Rocket': 64.00090156235028, 'Skateboard': 73.19549175663592, 'Table': 84.22721223031937}
==================================================
EPOCH 58 / 100
100%|█████████████████████████████| 876/876 [15:24<00:00,  1.05s/it, data_loading=0.566, iteration=0.216, train_acc=95.91, train_loss_seg=0.100, train_macc=92.45, train_miou=87.66]
Learning rate = 0.000063
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.08it/s, test_acc=93.43, test_loss_seg=0.105, test_macc=87.56, test_miou=81.20]
==================================================
    test_loss_seg = 0.10538526624441147
    test_acc = 93.43827427058025
    test_macc = 87.56831853753893
    test_miou = 81.20878540867594
    test_acc_per_class = {'Airplane': 91.69334791972142, 'Bag': 96.38323102678571, 'Cap': 90.98455255681817, 'Car': 92.1915175039557, 'Chair': 95.08882002397017, 'Earphone': 91.48995535714286, 'Guitar': 96.54917207154088, 'Knife': 92.9608154296875, 'Lamp': 91.85560533216784, 'Laptop': 98.00687123493977, 'Motorbike': 87.6043581495098, 'Mug': 99.45389597039474, 'Pistol': 95.60657848011364, 'Rocket': 84.92024739583334, 'Skateboard': 94.85414566532258, 'Table': 95.36927421137972}
    test_macc_per_class = {'Airplane': 89.55651822731025, 'Bag': 85.57563667045993, 'Cap': 84.12881182819399, 'Car': 85.86819907484585, 'Chair': 92.44241383371849, 'Earphone': 71.4821174455458, 'Guitar': 94.7070546413691, 'Knife': 92.96164477485652, 'Lamp': 89.60131033664787, 'Laptop': 98.00713566894194, 'Motorbike': 83.8275390382765, 'Mug': 95.88353209854681, 'Pistol': 86.29666117752336, 'Rocket': 76.47002022714527, 'Skateboard': 83.3856715398786, 'Table': 90.8988300173624}
    test_miou_per_class = {'Airplane': 82.271263416433, 'Bag': 81.6948891694764, 'Cap': 78.10831636397803, 'Car': 78.2944288795648, 'Chair': 86.11169410476566, 'Earphone': 64.21606678153931, 'Guitar': 90.81787302996021, 'Knife': 86.84675608898216, 'Lamp': 81.94355272134692, 'Laptop': 96.06526764943304, 'Motorbike': 73.07641002835375, 'Mug': 94.61356365970353, 'Pistol': 80.92277161823625, 'Rocket': 65.96719402179977, 'Skateboard': 73.79425920133428, 'Table': 84.59625980390769}
==================================================
EPOCH 59 / 100
100%|█████████████████████████████| 876/876 [15:23<00:00,  1.05s/it, data_loading=0.568, iteration=0.224, train_acc=96.02, train_loss_seg=0.102, train_macc=92.71, train_miou=87.71]
Learning rate = 0.000063
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.08it/s, test_acc=93.43, test_loss_seg=0.098, test_macc=87.63, test_miou=81.24]
==================================================
    test_loss_seg = 0.0983835980296135
    test_acc = 93.43495664515599
    test_macc = 87.63755242757166
    test_miou = 81.24369973916482
    test_acc_per_class = {'Airplane': 91.63693067265396, 'Bag': 96.27511160714286, 'Cap': 91.37517755681817, 'Car': 92.21747676028481, 'Chair': 95.05226828835227, 'Earphone': 91.62248883928571, 'Guitar': 96.54026631289308, 'Knife': 93.0682373046875, 'Lamp': 91.52336920891608, 'Laptop': 98.1574736445783, 'Motorbike': 87.35830269607843, 'Mug': 99.44490131578947, 'Pistol': 95.6265536221591, 'Rocket': 84.84293619791666, 'Skateboard': 94.87462197580645, 'Table': 95.34319031913326}
    test_macc_per_class = {'Airplane': 89.63617660549588, 'Bag': 84.51619193726415, 'Cap': 85.09754906489898, 'Car': 86.65447641831281, 'Chair': 92.49907905846729, 'Earphone': 71.52097363529963, 'Guitar': 94.76195615087445, 'Knife': 93.06925947671434, 'Lamp': 89.09453857224375, 'Laptop': 98.1227775283306, 'Motorbike': 84.07336239295688, 'Mug': 95.78597821915149, 'Pistol': 87.33516534742455, 'Rocket': 75.99218854884265, 'Skateboard': 83.41363380569601, 'Table': 90.62753207917285}
    test_miou_per_class = {'Airplane': 82.18959982286673, 'Bag': 80.95936566228063, 'Cap': 79.13081949010068, 'Car': 78.5425568457506, 'Chair': 86.02347102599953, 'Earphone': 64.49414261033142, 'Guitar': 90.86867713511792, 'Knife': 87.03450293725304, 'Lamp': 81.3737004435295, 'Laptop': 96.35458287136656, 'Motorbike': 72.76224989707075, 'Mug': 94.52363310779205, 'Pistol': 81.61833921172335, 'Rocket': 65.72038372706677, 'Skateboard': 73.8004264051804, 'Table': 84.50274463320707}
==================================================
EPOCH 60 / 100
100%|█████████████████████████████| 876/876 [15:22<00:00,  1.05s/it, data_loading=0.573, iteration=0.221, train_acc=95.77, train_loss_seg=0.107, train_macc=92.65, train_miou=87.79]
Learning rate = 0.000063
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.09it/s, test_acc=93.42, test_loss_seg=0.118, test_macc=87.63, test_miou=81.15]
==================================================
    test_loss_seg = 0.11819978058338165
    test_acc = 93.42168794060935
    test_macc = 87.63405398460577
    test_miou = 81.15195236866127
    test_acc_per_class = {'Airplane': 91.57879513379766, 'Bag': 96.25767299107143, 'Cap': 91.0555752840909, 'Car': 92.05028678797468, 'Chair': 95.0742548162287, 'Earphone': 91.59458705357143, 'Guitar': 96.57097582547169, 'Knife': 93.507080078125, 'Lamp': 91.37381173513987, 'Laptop': 98.08805534638554, 'Motorbike': 87.62063419117648, 'Mug': 99.4384765625, 'Pistol': 95.63210227272727, 'Rocket': 84.66796875, 'Skateboard': 94.84154485887096, 'Table': 95.39518536261792}
    test_macc_per_class = {'Airplane': 89.59473284507841, 'Bag': 85.02840931894494, 'Cap': 84.6524619559452, 'Car': 86.74035608615017, 'Chair': 92.93292395591946, 'Earphone': 71.68646076561652, 'Guitar': 94.82546310519032, 'Knife': 93.49887551196242, 'Lamp': 88.39111323216711, 'Laptop': 98.07305439081121, 'Motorbike': 84.06468293363332, 'Mug': 96.37424104485675, 'Pistol': 86.24432824993633, 'Rocket': 75.48679432337335, 'Skateboard': 83.51767276041434, 'Table': 91.03329327369242}
    test_miou_per_class = {'Airplane': 82.10562883315983, 'Bag': 81.08316762437062, 'Cap': 78.44489148041083, 'Car': 78.23538551548546, 'Chair': 86.1153068923095, 'Earphone': 64.5669882527477, 'Guitar': 90.9464955443775, 'Knife': 87.80261222475451, 'Lamp': 80.7147909279039, 'Laptop': 96.22134729261649, 'Motorbike': 73.16709167090076, 'Mug': 94.52961288041513, 'Pistol': 81.02128130293185, 'Rocket': 65.23007005207339, 'Skateboard': 73.55980278913853, 'Table': 84.68676461498454}
==================================================
EPOCH 61 / 100
100%|█████████████████████████████| 876/876 [15:23<00:00,  1.05s/it, data_loading=0.565, iteration=0.224, train_acc=95.42, train_loss_seg=0.100, train_macc=91.10, train_miou=86.47]
Learning rate = 0.000063
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.08it/s, test_acc=93.37, test_loss_seg=0.105, test_macc=87.77, test_miou=81.13]
==================================================
    test_loss_seg = 0.10501675307750702
    test_acc = 93.379627934502
    test_macc = 87.77650411686243
    test_miou = 81.13172840056505
    test_acc_per_class = {'Airplane': 91.64938828812316, 'Bag': 96.09723772321429, 'Cap': 91.23757102272727, 'Car': 92.23478293117088, 'Chair': 95.0897216796875, 'Earphone': 91.59109933035714, 'Guitar': 96.61642590408806, 'Knife': 93.1512451171875, 'Lamp': 91.67702414772727, 'Laptop': 98.11276355421687, 'Motorbike': 87.32479319852942, 'Mug': 99.42819695723685, 'Pistol': 95.74085582386364, 'Rocket': 83.86637369791666, 'Skateboard': 94.87934727822581, 'Table': 95.37722029775944}
    test_macc_per_class = {'Airplane': 89.90631322089222, 'Bag': 85.25210168680036, 'Cap': 84.87166805660145, 'Car': 86.88640112019243, 'Chair': 92.61142608227246, 'Earphone': 71.56238082785235, 'Guitar': 95.10503041715995, 'Knife': 93.14731242326229, 'Lamp': 88.39455485897251, 'Laptop': 98.08339714612109, 'Motorbike': 84.21605117234243, 'Mug': 95.49874016269892, 'Pistol': 86.88915074063681, 'Rocket': 77.45324427013122, 'Skateboard': 83.77749519089002, 'Table': 90.76879849297241}
    test_miou_per_class = {'Airplane': 82.26802163854434, 'Bag': 80.63471047644863, 'Cap': 78.82041342131531, 'Car': 78.79095522099612, 'Chair': 86.10496135591507, 'Earphone': 64.51532737429226, 'Guitar': 91.05907382737084, 'Knife': 87.17858998912662, 'Lamp': 80.78593233475696, 'Laptop': 96.26820442697688, 'Motorbike': 72.74828810566163, 'Mug': 94.34479283175395, 'Pistol': 81.65003387900707, 'Rocket': 64.67883599310835, 'Skateboard': 73.68076256897494, 'Table': 84.57875096479216}
==================================================
EPOCH 62 / 100
100%|█████████████████████████████| 876/876 [15:23<00:00,  1.05s/it, data_loading=0.563, iteration=0.221, train_acc=95.79, train_loss_seg=0.102, train_macc=92.32, train_miou=85.52]
Learning rate = 0.000063
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:25<00:00,  2.10it/s, test_acc=93.44, test_loss_seg=0.134, test_macc=87.54, test_miou=81.18]
==================================================
    test_loss_seg = 0.1340121179819107
    test_acc = 93.44642176096599
    test_macc = 87.54884688774098
    test_miou = 81.18235319786902
    test_acc_per_class = {'Airplane': 91.65712060117302, 'Bag': 96.01004464285714, 'Cap': 91.064453125, 'Car': 92.23447389240506, 'Chair': 95.0563604181463, 'Earphone': 92.35142299107143, 'Guitar': 96.58295253537736, 'Knife': 93.17626953125, 'Lamp': 91.47846782124127, 'Laptop': 97.98451618975903, 'Motorbike': 87.71541819852942, 'Mug': 99.44104646381578, 'Pistol': 95.75750177556817, 'Rocket': 84.22037760416666, 'Skateboard': 95.07465977822581, 'Table': 95.3376626068691}
    test_macc_per_class = {'Airplane': 89.83592890541281, 'Bag': 85.4719026006979, 'Cap': 84.34172246031943, 'Car': 87.47014354776493, 'Chair': 92.43253599477138, 'Earphone': 71.82062133333694, 'Guitar': 94.83235442442357, 'Knife': 93.16476971080615, 'Lamp': 88.7088311753136, 'Laptop': 98.00815369048802, 'Motorbike': 81.53175316053867, 'Mug': 95.73754025331216, 'Pistol': 87.14945819557155, 'Rocket': 75.72129826169916, 'Skateboard': 84.00507552023981, 'Table': 90.54946096915968}
    test_miou_per_class = {'Airplane': 82.2797976077493, 'Bag': 80.43155053618388, 'Cap': 78.32463221531141, 'Car': 78.82432383639781, 'Chair': 86.12801367991301, 'Earphone': 65.65353926214583, 'Guitar': 90.96297608294063, 'Knife': 87.21937987614098, 'Lamp': 80.92503287546958, 'Laptop': 96.02375125236031, 'Motorbike': 72.3919774331324, 'Mug': 94.4843651283495, 'Pistol': 81.72940427727077, 'Rocket': 64.76625940293378, 'Skateboard': 74.31272008931245, 'Table': 84.45992761029245}
==================================================
EPOCH 63 / 100
100%|█████████████████████████████| 876/876 [15:23<00:00,  1.05s/it, data_loading=0.566, iteration=0.222, train_acc=96.00, train_loss_seg=0.101, train_macc=91.77, train_miou=87.23]
Learning rate = 0.000063
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.08it/s, test_acc=93.46, test_loss_seg=0.115, test_macc=87.45, test_miou=81.16]
==================================================
    test_loss_seg = 0.11503757536411285
    test_acc = 93.46861047449224
    test_macc = 87.4567182531403
    test_miou = 81.16060007345021
    test_acc_per_class = {'Airplane': 91.64294469391496, 'Bag': 96.10770089285714, 'Cap': 91.06889204545455, 'Car': 92.28608336629746, 'Chair': 95.08923617276278, 'Earphone': 92.19098772321429, 'Guitar': 96.5685190644654, 'Knife': 93.2232666015625, 'Lamp': 91.7898751638986, 'Laptop': 98.10805722891565, 'Motorbike': 87.70009957107843, 'Mug': 99.44104646381578, 'Pistol': 95.6387606534091, 'Rocket': 84.130859375, 'Skateboard': 95.17389112903226, 'Table': 95.33754744619694}
    test_macc_per_class = {'Airplane': 89.93056771022484, 'Bag': 83.02103945338857, 'Cap': 84.17579283147931, 'Car': 86.5423821894326, 'Chair': 91.87288697562556, 'Earphone': 71.55944921596267, 'Guitar': 94.68287918425263, 'Knife': 93.2165898540307, 'Lamp': 88.67539282700511, 'Laptop': 98.08552544759017, 'Motorbike': 80.46714589083398, 'Mug': 95.99276292810333, 'Pistol': 87.54257458962478, 'Rocket': 77.0500970778345, 'Skateboard': 85.46668752406428, 'Table': 91.02571835079158}
    test_miou_per_class = {'Airplane': 82.30239644006076, 'Bag': 79.84593566994742, 'Cap': 78.25870752227404, 'Car': 78.70669638182628, 'Chair': 86.17536351338178, 'Earphone': 65.23158186672248, 'Guitar': 90.86994412461384, 'Knife': 87.30392062616828, 'Lamp': 81.1574788148222, 'Laptop': 96.25957671803755, 'Motorbike': 72.02327660468656, 'Mug': 94.51171667079427, 'Pistol': 81.62493900467874, 'Rocket': 65.06270544205783, 'Skateboard': 74.70554415984387, 'Table': 84.52981761528724}
==================================================
EPOCH 64 / 100
100%|█████████████████████████████| 876/876 [15:27<00:00,  1.06s/it, data_loading=0.569, iteration=0.223, train_acc=95.58, train_loss_seg=0.101, train_macc=92.61, train_miou=87.54]
Learning rate = 0.000063
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.07it/s, test_acc=93.45, test_loss_seg=0.073, test_macc=87.57, test_miou=81.21]
==================================================
    test_loss_seg = 0.07326789945363998
    test_acc = 93.45250785917825
    test_macc = 87.57269098467837
    test_miou = 81.21925946021004
    test_acc_per_class = {'Airplane': 91.66800311583577, 'Bag': 96.19838169642857, 'Cap': 91.50390625, 'Car': 92.32069570806962, 'Chair': 95.12433138760653, 'Earphone': 91.748046875, 'Guitar': 96.58848024764151, 'Knife': 93.292236328125, 'Lamp': 91.7724609375, 'Laptop': 98.16570971385542, 'Motorbike': 87.75658700980392, 'Mug': 99.43462171052632, 'Pistol': 95.6454190340909, 'Rocket': 83.447265625, 'Skateboard': 95.21011844758065, 'Table': 95.36386165978774}
    test_macc_per_class = {'Airplane': 89.87698995993259, 'Bag': 83.93654498381748, 'Cap': 85.17481474413826, 'Car': 86.8505136207002, 'Chair': 92.26953481305203, 'Earphone': 71.67375964133635, 'Guitar': 94.75114461474044, 'Knife': 93.28537656867101, 'Lamp': 89.00828414573472, 'Laptop': 98.12521495671923, 'Motorbike': 82.32554663467019, 'Mug': 95.93136803615835, 'Pistol': 87.01087193602683, 'Rocket': 75.39902521885699, 'Skateboard': 85.25649254797997, 'Table': 90.28757333231917}
    test_miou_per_class = {'Airplane': 82.31581679114603, 'Bag': 80.49192918514092, 'Cap': 79.3666237256631, 'Car': 78.82308193673435, 'Chair': 86.22554871698003, 'Earphone': 64.77435543813083, 'Guitar': 90.9274618219039, 'Knife': 87.42493695917985, 'Lamp': 81.40377315461458, 'Laptop': 96.37015618887501, 'Motorbike': 72.50465616228122, 'Mug': 94.4486530016361, 'Pistol': 81.50886178109134, 'Rocket': 63.52135318959626, 'Skateboard': 74.9771702779134, 'Table': 84.42377303247346}
==================================================
EPOCH 65 / 100
100%|█████████████████████████████| 876/876 [15:25<00:00,  1.06s/it, data_loading=0.56 , iteration=0.226, train_acc=96.10, train_loss_seg=0.103, train_macc=92.50, train_miou=88.14]
Learning rate = 0.000063
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.09it/s, test_acc=93.46, test_loss_seg=0.13 , test_macc=87.39, test_miou=81.12]
==================================================
    test_loss_seg = 0.13004061579704285
    test_acc = 93.46167601825279
    test_macc = 87.39695717691238
    test_miou = 81.12711334153677
    test_acc_per_class = {'Airplane': 91.6592684659091, 'Bag': 96.21233258928571, 'Cap': 91.18874289772727, 'Car': 92.19893443433544, 'Chair': 95.11455189098011, 'Earphone': 91.13071986607143, 'Guitar': 96.59431505503144, 'Knife': 93.23974609375, 'Lamp': 91.57424606643356, 'Laptop': 98.15453219126506, 'Motorbike': 87.70392922794117, 'Mug': 99.43462171052632, 'Pistol': 95.6997958096591, 'Rocket': 85.14404296875, 'Skateboard': 94.96597782258065, 'Table': 95.37105920179835}
    test_macc_per_class = {'Airplane': 89.69137316129412, 'Bag': 85.22681313817813, 'Cap': 84.65883693558935, 'Car': 85.35549894251314, 'Chair': 92.53698970095435, 'Earphone': 71.31130448482583, 'Guitar': 94.98669904466138, 'Knife': 93.23304214850634, 'Lamp': 88.68892601601794, 'Laptop': 98.11390513393057, 'Motorbike': 80.6047442161693, 'Mug': 96.81304636725511, 'Pistol': 86.03019181296467, 'Rocket': 75.51149130062494, 'Skateboard': 84.92542146065645, 'Table': 90.66303096645659}
    test_miou_per_class = {'Airplane': 82.3314875311629, 'Bag': 81.00233065016891, 'Cap': 78.6540266786758, 'Car': 78.20877007676744, 'Chair': 86.15886784644107, 'Earphone': 63.76158796287801, 'Guitar': 91.00607168906512, 'Knife': 87.33282800196783, 'Lamp': 80.90612324513266, 'Laptop': 96.34844038867622, 'Motorbike': 72.08805164903085, 'Mug': 94.54202175653883, 'Pistol': 81.07979694494247, 'Rocket': 65.81585279076316, 'Skateboard': 74.25858677222018, 'Table': 84.53896948015692}
==================================================
EPOCH 66 / 100
100%|█████████████████████████████| 876/876 [15:22<00:00,  1.05s/it, data_loading=0.557, iteration=0.225, train_acc=95.92, train_loss_seg=0.100, train_macc=92.59, train_miou=87.85]
Learning rate = 0.000063
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.09it/s, test_acc=93.36, test_loss_seg=0.089, test_macc=87.84, test_miou=81.22]
==================================================
    test_loss_seg = 0.08990087360143661
    test_acc = 93.36608057427193
    test_macc = 87.84893736443917
    test_miou = 81.2241158416873
    test_acc_per_class = {'Airplane': 91.59139594024927, 'Bag': 96.25418526785714, 'Cap': 91.47727272727273, 'Car': 92.28392009493672, 'Chair': 95.08209228515625, 'Earphone': 91.23186383928571, 'Guitar': 96.64037932389937, 'Knife': 93.3868408203125, 'Lamp': 91.7270473666958, 'Laptop': 98.1410015060241, 'Motorbike': 86.63641237745098, 'Mug': 99.45646587171053, 'Pistol': 95.60990767045455, 'Rocket': 84.09423828125, 'Skateboard': 94.88092237903226, 'Table': 95.36334343676297}
    test_macc_per_class = {'Airplane': 89.41406376259545, 'Bag': 84.16153056825615, 'Cap': 85.13552417651144, 'Car': 87.06684347215288, 'Chair': 92.44519541872683, 'Earphone': 71.50410371898724, 'Guitar': 95.19641519686185, 'Knife': 93.37920438791602, 'Lamp': 88.90131978650832, 'Laptop': 98.10992243350745, 'Motorbike': 85.15075310848589, 'Mug': 96.16331273011596, 'Pistol': 87.17054101571192, 'Rocket': 76.41633702812098, 'Skateboard': 84.22621752797274, 'Table': 91.14171349859545}
    test_miou_per_class = {'Airplane': 82.14917947423476, 'Bag': 80.76074724643227, 'Cap': 79.30805672228604, 'Car': 78.80704867826026, 'Chair': 85.94287982103704, 'Earphone': 64.04965295934463, 'Guitar': 91.1355749232694, 'Knife': 87.59098500732043, 'Lamp': 81.33836479144337, 'Laptop': 96.32286858443517, 'Motorbike': 72.96321043751118, 'Mug': 94.66547533366662, 'Pistol': 81.4176828822334, 'Rocket': 64.67825859884832, 'Skateboard': 73.86970734489768, 'Table': 84.58616066177619}
==================================================
EPOCH 67 / 100
100%|█████████████████████████████| 876/876 [15:23<00:00,  1.05s/it, data_loading=0.566, iteration=0.219, train_acc=95.99, train_loss_seg=0.095, train_macc=92.53, train_miou=87.98]
Learning rate = 0.000063
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:25<00:00,  2.10it/s, test_acc=93.40, test_loss_seg=0.108, test_macc=87.67, test_miou=81.18]
==================================================
    test_loss_seg = 0.10797581821680069
    test_acc = 93.40168527274128
    test_macc = 87.67911187203845
    test_miou = 81.18718255895207
    test_acc_per_class = {'Airplane': 91.61101310483872, 'Bag': 96.06236049107143, 'Cap': 91.09108664772727, 'Car': 92.19862539556962, 'Chair': 95.12863159179688, 'Earphone': 91.2353515625, 'Guitar': 96.65450569968553, 'Knife': 93.226318359375, 'Lamp': 91.5474418159965, 'Laptop': 98.1086455195783, 'Motorbike': 87.69626991421569, 'Mug': 99.45646587171053, 'Pistol': 95.67205255681817, 'Rocket': 84.59065755208334, 'Skateboard': 94.84154485887096, 'Table': 95.30599342202241}
    test_macc_per_class = {'Airplane': 89.75803578772621, 'Bag': 85.38177747136967, 'Cap': 84.53413130284092, 'Car': 86.22877260085569, 'Chair': 92.33332708605289, 'Earphone': 71.644386630911, 'Guitar': 95.41092834401549, 'Knife': 93.224472271971, 'Lamp': 89.47561930215026, 'Laptop': 98.09808840018292, 'Motorbike': 82.34235695291763, 'Mug': 96.12850963809898, 'Pistol': 87.46487531226911, 'Rocket': 77.0068125155525, 'Skateboard': 83.38061232852316, 'Table': 90.45308400717792}
    test_miou_per_class = {'Airplane': 82.23201368102967, 'Bag': 80.56760755760139, 'Cap': 78.44908469451164, 'Car': 78.3922529668571, 'Chair': 86.21465365218826, 'Earphone': 64.21578167058742, 'Guitar': 91.20576953938074, 'Knife': 87.3107977363651, 'Lamp': 81.50183371738599, 'Laptop': 96.26158306539156, 'Motorbike': 72.6608015695417, 'Mug': 94.66186845642852, 'Pistol': 81.68086593568525, 'Rocket': 65.67932300592668, 'Skateboard': 73.65383650145536, 'Table': 84.30684719289697}
==================================================
EPOCH 68 / 100
100%|█████████████████████████████| 876/876 [15:23<00:00,  1.05s/it, data_loading=0.566, iteration=0.220, train_acc=96.05, train_loss_seg=0.098, train_macc=92.86, train_miou=88.27]
Learning rate = 0.000063
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.09it/s, test_acc=93.56, test_loss_seg=0.085, test_macc=87.90, test_miou=81.51]
==================================================
    test_loss_seg = 0.08513511717319489
    test_acc = 93.56662997123917
    test_macc = 87.90273451874144
    test_miou = 81.51345742793414
    test_acc_per_class = {'Airplane': 91.60299440982405, 'Bag': 96.24372209821429, 'Cap': 91.52610085227273, 'Car': 92.27681220332279, 'Chair': 95.10664506392045, 'Earphone': 92.1875, 'Guitar': 96.54395145440252, 'Knife': 93.7420654296875, 'Lamp': 91.76341236888112, 'Laptop': 98.09158509036145, 'Motorbike': 87.51531862745098, 'Mug': 99.45004111842105, 'Pistol': 95.66206498579545, 'Rocket': 85.05045572916666, 'Skateboard': 94.96597782258065, 'Table': 95.33743228552476}
    test_macc_per_class = {'Airplane': 89.93850013396317, 'Bag': 83.85744236175658, 'Cap': 85.30611577341467, 'Car': 87.47046990427334, 'Chair': 92.69227366082968, 'Earphone': 72.0316925312194, 'Guitar': 94.77202546915441, 'Knife': 93.73731384301954, 'Lamp': 89.54683729455955, 'Laptop': 98.07529676022516, 'Motorbike': 83.00195824830615, 'Mug': 96.32233742094517, 'Pistol': 86.85356899741016, 'Rocket': 79.01648425042903, 'Skateboard': 83.24907096075871, 'Table': 90.57236468959863}
    test_miou_per_class = {'Airplane': 82.25793243982771, 'Bag': 80.61384443634891, 'Cap': 79.45599201291714, 'Car': 78.90469104909741, 'Chair': 86.26449611818889, 'Earphone': 65.60151266760845, 'Guitar': 90.85574554918155, 'Knife': 88.21923339323565, 'Lamp': 81.71222740639254, 'Laptop': 96.22809520070837, 'Motorbike': 72.68653508163193, 'Mug': 94.62543132816106, 'Pistol': 81.42037180380478, 'Rocket': 66.79346816449673, 'Skateboard': 74.15257780997926, 'Table': 84.423164385366}
==================================================
EPOCH 69 / 100
100%|█████████████████████████████| 876/876 [15:25<00:00,  1.06s/it, data_loading=0.560, iteration=0.224, train_acc=95.69, train_loss_seg=0.100, train_macc=91.57, train_miou=86.37]
Learning rate = 0.000063
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.08it/s, test_acc=93.50, test_loss_seg=0.127, test_macc=87.63, test_miou=81.29]
==================================================
    test_loss_seg = 0.1275930106639862
    test_acc = 93.50893581885047
    test_macc = 87.63670192793421
    test_miou = 81.29894702329905
    test_acc_per_class = {'Airplane': 91.62948474156892, 'Bag': 96.36579241071429, 'Cap': 91.52166193181817, 'Car': 92.08273585838607, 'Chair': 95.08757157759233, 'Earphone': 91.83175223214286, 'Guitar': 96.61274076257862, 'Knife': 92.918701171875, 'Lamp': 91.75829053758741, 'Laptop': 98.10040945030121, 'Motorbike': 87.59765625, 'Mug': 99.4371916118421, 'Pistol': 95.6787109375, 'Rocket': 85.14404296875, 'Skateboard': 94.99117943548387, 'Table': 95.38505122346697}
    test_macc_per_class = {'Airplane': 89.68385089048596, 'Bag': 84.62641438678004, 'Cap': 85.27668760325122, 'Car': 85.84379030252008, 'Chair': 92.39032975774467, 'Earphone': 71.69912494756846, 'Guitar': 95.04967980037313, 'Knife': 92.92456347685787, 'Lamp': 89.36019310724981, 'Laptop': 98.0878727650913, 'Motorbike': 83.65010866444376, 'Mug': 96.57078061856961, 'Pistol': 86.6593532166407, 'Rocket': 74.78295833028416, 'Skateboard': 84.31133910848789, 'Table': 91.2701838705988}
    test_miou_per_class = {'Airplane': 82.15655909181248, 'Bag': 81.30515974771943, 'Cap': 79.43685181743221, 'Car': 78.15886266497239, 'Chair': 86.17623197729307, 'Earphone': 64.8516156603958, 'Guitar': 91.0550437162396, 'Knife': 86.77390575496854, 'Lamp': 81.66707899171553, 'Laptop': 96.24547320174226, 'Motorbike': 72.8509041344011, 'Mug': 94.53904767221168, 'Pistol': 81.40886046947475, 'Rocket': 65.40323131462131, 'Skateboard': 74.03350262749974, 'Table': 84.72082353028495}
==================================================
EPOCH 70 / 100
100%|█████████████████████████████| 876/876 [15:26<00:00,  1.06s/it, data_loading=0.565, iteration=0.223, train_acc=96.01, train_loss_seg=0.107, train_macc=92.64, train_miou=87.71]
Learning rate = 0.000063
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.07it/s, test_acc=93.48, test_loss_seg=0.231, test_macc=87.44, test_miou=81.20]
==================================================
    test_loss_seg = 0.23108461499214172
    test_acc = 93.48800320656267
    test_macc = 87.44024134858017
    test_miou = 81.20890234549348
    test_acc_per_class = {'Airplane': 91.54629078079178, 'Bag': 96.35184151785714, 'Cap': 91.13991477272727, 'Car': 92.20357001582279, 'Chair': 95.09069269353692, 'Earphone': 92.13169642857143, 'Guitar': 96.635158706761, 'Knife': 92.8643798828125, 'Lamp': 91.65431736232517, 'Laptop': 98.046875, 'Motorbike': 87.94519761029412, 'Mug': 99.41920230263158, 'Pistol': 95.50115411931817, 'Rocket': 84.7900390625, 'Skateboard': 95.10773689516128, 'Table': 95.37998415389151}
    test_macc_per_class = {'Airplane': 89.77256713628127, 'Bag': 84.43968230626318, 'Cap': 84.70472289974413, 'Car': 87.23514283145123, 'Chair': 92.65202755138048, 'Earphone': 71.93400435627639, 'Guitar': 95.0937981433646, 'Knife': 92.86698848980373, 'Lamp': 89.04151246609308, 'Laptop': 98.0235924660966, 'Motorbike': 81.8982911415504, 'Mug': 96.30606667574504, 'Pistol': 84.96638614075627, 'Rocket': 75.47565927955416, 'Skateboard': 83.99745790118969, 'Table': 90.63596179173281}
    test_miou_per_class = {'Airplane': 82.1306673731024, 'Bag': 81.19099470292659, 'Cap': 78.59770001645029, 'Car': 78.67778890695249, 'Chair': 86.17073848560521, 'Earphone': 65.36253252863762, 'Guitar': 91.09171243977528, 'Knife': 86.67886670886601, 'Lamp': 81.65562593957958, 'Laptop': 96.14094561805537, 'Motorbike': 72.40039556658084, 'Mug': 94.35451142880456, 'Pistol': 80.4714599123745, 'Rocket': 65.3418710786756, 'Skateboard': 74.46162963449514, 'Table': 84.61499718701407}
==================================================
EPOCH 71 / 100
100%|█████████████████████████████| 876/876 [15:23<00:00,  1.05s/it, data_loading=0.559, iteration=0.226, train_acc=95.74, train_loss_seg=0.100, train_macc=92.20, train_miou=87.63]
Learning rate = 0.000063
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:25<00:00,  2.11it/s, test_acc=93.48, test_loss_seg=0.071, test_macc=87.69, test_miou=81.21]
==================================================
    test_loss_seg = 0.07190120220184326
    test_acc = 93.48237598277818
    test_macc = 87.6918111286846
    test_miou = 81.21532771180559
    test_acc_per_class = {'Airplane': 91.63363728005865, 'Bag': 96.04840959821429, 'Cap': 91.05113636363636, 'Car': 92.2471444818038, 'Chair': 95.0838262384588, 'Earphone': 92.05496651785714, 'Guitar': 96.59216538915094, 'Knife': 93.16162109375, 'Lamp': 91.68402398382867, 'Laptop': 98.03569747740963, 'Motorbike': 87.85520067401961, 'Mug': 99.44361636513158, 'Pistol': 95.80300071022727, 'Rocket': 84.70052083333334, 'Skateboard': 95.068359375, 'Table': 95.25468934257076}
    test_macc_per_class = {'Airplane': 89.67634073952418, 'Bag': 84.88187974827552, 'Cap': 84.22175830674743, 'Car': 87.06928536322087, 'Chair': 92.2945482054622, 'Earphone': 71.5448023156399, 'Guitar': 94.78020324380351, 'Knife': 93.1603766848917, 'Lamp': 89.10156687564454, 'Laptop': 98.0348091380452, 'Motorbike': 83.21208492191495, 'Mug': 96.31894768236181, 'Pistol': 86.48766606751934, 'Rocket': 77.15477829477994, 'Skateboard': 84.09509752043357, 'Table': 91.03483295068912}
    test_miou_per_class = {'Airplane': 82.18265940855744, 'Bag': 80.34563160866607, 'Cap': 78.25159442378111, 'Car': 78.74173084348439, 'Chair': 86.07467122111294, 'Earphone': 65.005545913611, 'Guitar': 90.93335315259147, 'Knife': 87.19750965410552, 'Lamp': 81.51404797695959, 'Laptop': 96.12096410250035, 'Motorbike': 72.61797289952825, 'Mug': 94.56875036600725, 'Pistol': 81.62716235283041, 'Rocket': 65.6512408422959, 'Skateboard': 74.29182239134575, 'Table': 84.32058623151185}
==================================================
EPOCH 72 / 100
100%|█████████████████████████████| 876/876 [15:22<00:00,  1.05s/it, data_loading=0.562, iteration=0.225, train_acc=95.70, train_loss_seg=0.102, train_macc=92.30, train_miou=87.38]
Learning rate = 0.000031
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.09it/s, test_acc=93.59, test_loss_seg=0.102, test_macc=87.98, test_miou=81.53]
==================================================
    test_loss_seg = 0.10293351113796234
    test_acc = 93.59233664837875
    test_macc = 87.98132432753593
    test_miou = 81.53718226991556
    test_acc_per_class = {'Airplane': 91.57836556085044, 'Bag': 96.35532924107143, 'Cap': 91.57936789772727, 'Car': 92.28762856012658, 'Chair': 95.05774758078836, 'Earphone': 92.10728236607143, 'Guitar': 96.56667649371069, 'Knife': 93.3184814453125, 'Lamp': 91.74685178103147, 'Laptop': 98.02393166415662, 'Motorbike': 87.5765931372549, 'Mug': 99.46032072368422, 'Pistol': 95.68314985795455, 'Rocket': 85.75846354166666, 'Skateboard': 94.99905493951613, 'Table': 95.3781415831368}
    test_macc_per_class = {'Airplane': 89.875259748206, 'Bag': 84.57584122427161, 'Cap': 85.3371774440458, 'Car': 88.06281192040906, 'Chair': 92.45733545255715, 'Earphone': 71.75915788528961, 'Guitar': 94.73888643707213, 'Knife': 93.30594556738087, 'Lamp': 89.30292391816069, 'Laptop': 98.02315925891138, 'Motorbike': 83.88799534935335, 'Mug': 96.82660532158856, 'Pistol': 87.02679619850207, 'Rocket': 77.67303803931503, 'Skateboard': 83.96245745060168, 'Table': 90.88579802490968}
    test_miou_per_class = {'Airplane': 82.19749144563943, 'Bag': 81.25155732579334, 'Cap': 79.55367469260983, 'Car': 79.0107798851113, 'Chair': 86.04527283138529, 'Earphone': 65.1839834306363, 'Guitar': 90.95099447483396, 'Knife': 87.46851260197161, 'Lamp': 81.81247264590584, 'Laptop': 96.0982005962207, 'Motorbike': 72.59836985063926, 'Mug': 94.76689174563613, 'Pistol': 81.54721306864268, 'Rocket': 67.2705601109945, 'Skateboard': 74.21759820327767, 'Table': 84.62134340935098}
==================================================
EPOCH 73 / 100
100%|█████████████████████████████| 876/876 [15:22<00:00,  1.05s/it, data_loading=0.562, iteration=0.222, train_acc=95.50, train_loss_seg=0.099, train_macc=91.42, train_miou=86.61]
Learning rate = 0.000031
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.09it/s, test_acc=93.54, test_loss_seg=0.08 , test_macc=87.52, test_miou=81.32]
==================================================
    test_loss_seg = 0.07999621331691742
    test_acc = 93.54411465726321
    test_macc = 87.52986380652371
    test_miou = 81.3225755283742
    test_acc_per_class = {'Airplane': 91.528678289956, 'Bag': 96.337890625, 'Cap': 91.2686434659091, 'Car': 92.27990259098101, 'Chair': 95.11711814186789, 'Earphone': 92.47698102678571, 'Guitar': 96.60567757468553, 'Knife': 92.7459716796875, 'Lamp': 91.63365930944056, 'Laptop': 98.08570218373494, 'Motorbike': 87.91647518382352, 'Mug': 99.45518092105263, 'Pistol': 95.71311257102273, 'Rocket': 85.24169921875, 'Skateboard': 94.95652721774194, 'Table': 95.34261451577241}
    test_macc_per_class = {'Airplane': 89.7768741285928, 'Bag': 84.6108538172632, 'Cap': 84.78198857898343, 'Car': 87.16242997885414, 'Chair': 92.3492421247602, 'Earphone': 71.6255809636629, 'Guitar': 94.93073938993003, 'Knife': 92.75169843243609, 'Lamp': 88.8439285506709, 'Laptop': 98.0565418147104, 'Motorbike': 81.8583992418874, 'Mug': 96.04662447567604, 'Pistol': 87.48495520503225, 'Rocket': 75.89956535395524, 'Skateboard': 83.51214802823294, 'Table': 90.7862508197313}
    test_miou_per_class = {'Airplane': 82.11867552347215, 'Bag': 81.20485842494159, 'Cap': 78.83110648487038, 'Car': 78.85881524791891, 'Chair': 86.16333334691964, 'Earphone': 65.55267466584405, 'Guitar': 91.05546605779992, 'Knife': 86.47310265894019, 'Lamp': 81.36981349775714, 'Laptop': 96.21574359322422, 'Motorbike': 72.31085346420436, 'Mug': 94.64200619474721, 'Pistol': 81.95717450443463, 'Rocket': 66.03231804795678, 'Skateboard': 73.92733869438212, 'Table': 84.44792804657384}
==================================================
EPOCH 74 / 100
100%|█████████████████████████████| 876/876 [15:25<00:00,  1.06s/it, data_loading=0.562, iteration=0.225, train_acc=96.13, train_loss_seg=0.098, train_macc=92.05, train_miou=87.33]
Learning rate = 0.000031
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.08it/s, test_acc=93.49, test_loss_seg=0.124, test_macc=87.71, test_miou=81.33]
==================================================
    test_loss_seg = 0.12418320029973984
    test_acc = 93.4998996617015
    test_macc = 87.7148267652354
    test_miou = 81.3372423493814
    test_acc_per_class = {'Airplane': 91.56705347324046, 'Bag': 96.27162388392857, 'Cap': 91.02894176136364, 'Car': 92.3203866693038, 'Chair': 95.07044011896308, 'Earphone': 92.42815290178571, 'Guitar': 96.6041420990566, 'Knife': 93.05419921875, 'Lamp': 91.54112489073427, 'Laptop': 98.08629047439759, 'Motorbike': 87.44638480392157, 'Mug': 99.46032072368422, 'Pistol': 95.6509676846591, 'Rocket': 84.912109375, 'Skateboard': 95.17546622983872, 'Table': 95.38079027859669}
    test_macc_per_class = {'Airplane': 89.75145115421893, 'Bag': 84.11160532561803, 'Cap': 84.54453216164146, 'Car': 87.00434275879907, 'Chair': 92.29609178780215, 'Earphone': 71.79806779703726, 'Guitar': 94.83530761968946, 'Knife': 93.05162744275573, 'Lamp': 88.9726917494252, 'Laptop': 98.06324585835803, 'Motorbike': 84.41587819612592, 'Mug': 96.02613420519808, 'Pistol': 87.00988411200618, 'Rocket': 75.93601625258825, 'Skateboard': 84.65739583797203, 'Table': 90.96295598453061}
    test_miou_per_class = {'Airplane': 82.1721292184926, 'Bag': 80.80099276212502, 'Cap': 78.35763834828032, 'Car': 78.8367927742536, 'Chair': 86.09932708370195, 'Earphone': 65.63367197895444, 'Guitar': 90.95155157984553, 'Knife': 87.00911972481795, 'Lamp': 81.29717402060642, 'Laptop': 96.2173369093627, 'Motorbike': 72.55807069123605, 'Mug': 94.68533922216704, 'Pistol': 81.50071221691437, 'Rocket': 65.63953233905943, 'Skateboard': 74.9758442789066, 'Table': 84.66064444137818}
==================================================
EPOCH 75 / 100
100%|█████████████████████████████| 876/876 [15:25<00:00,  1.06s/it, data_loading=0.564, iteration=0.223, train_acc=95.65, train_loss_seg=0.102, train_macc=92.02, train_miou=87.07]
Learning rate = 0.000031
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.08it/s, test_acc=93.51, test_loss_seg=0.113, test_macc=88.01, test_miou=81.46]
==================================================
    test_loss_seg = 0.11391552537679672
    test_acc = 93.51108240489727
    test_macc = 88.01383357611358
    test_miou = 81.4699478560723
    test_acc_per_class = {'Airplane': 91.63134622434018, 'Bag': 96.44252232142857, 'Cap': 90.9579190340909, 'Car': 92.28577432753164, 'Chair': 95.07994218306108, 'Earphone': 91.92592075892857, 'Guitar': 96.6554269850629, 'Knife': 93.0035400390625, 'Lamp': 91.7548759833916, 'Laptop': 98.0456984186747, 'Motorbike': 87.46266084558823, 'Mug': 99.4397615131579, 'Pistol': 95.6210049715909, 'Rocket': 85.48583984375, 'Skateboard': 95.04788306451613, 'Table': 95.33720196418042}
    test_macc_per_class = {'Airplane': 89.76143468197888, 'Bag': 85.9218685232605, 'Cap': 84.42215751292458, 'Car': 87.4655036845764, 'Chair': 92.3401508768114, 'Earphone': 71.68673533728412, 'Guitar': 95.17558157473684, 'Knife': 93.00122164866855, 'Lamp': 89.23103687224965, 'Laptop': 98.03695353174024, 'Motorbike': 84.91906072206343, 'Mug': 95.7368623055955, 'Pistol': 86.76908815963937, 'Rocket': 77.9483867139955, 'Skateboard': 84.74630397676934, 'Table': 91.05899109552324}
    test_miou_per_class = {'Airplane': 82.23068902511528, 'Bag': 82.0137133836888, 'Cap': 78.195741529906, 'Car': 78.91120099187206, 'Chair': 86.14485610612323, 'Earphone': 64.91989876947795, 'Guitar': 91.15720869814177, 'Knife': 86.92063895804321, 'Lamp': 81.6230276194154, 'Laptop': 96.13974891657742, 'Motorbike': 73.0651544741091, 'Mug': 94.47294189144925, 'Pistol': 81.40401179170024, 'Rocket': 67.10583725648095, 'Skateboard': 74.67741701336537, 'Table': 84.53707927169037}
==================================================
EPOCH 76 / 100
100%|█████████████████████████████| 876/876 [15:26<00:00,  1.06s/it, data_loading=0.563, iteration=0.222, train_acc=95.67, train_loss_seg=0.099, train_macc=92.00, train_miou=87.34]
Learning rate = 0.000031
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:27<00:00,  2.07it/s, test_acc=93.53, test_loss_seg=0.151, test_macc=88.03, test_miou=81.47]
==================================================
    test_loss_seg = 0.15097211301326752
    test_acc = 93.5336258539574
    test_macc = 88.03795073463186
    test_miou = 81.47365537629832
    test_acc_per_class = {'Airplane': 91.64681085043989, 'Bag': 96.337890625, 'Cap': 91.09996448863636, 'Car': 92.26970431170885, 'Chair': 95.00253850763495, 'Earphone': 92.578125, 'Guitar': 96.61028400157232, 'Knife': 93.253173828125, 'Lamp': 91.79260680725524, 'Laptop': 97.9851044804217, 'Motorbike': 87.27500765931373, 'Mug': 99.44618626644737, 'Pistol': 95.68093039772727, 'Rocket': 85.33935546875, 'Skateboard': 94.90454889112904, 'Table': 95.31578207915685}
    test_macc_per_class = {'Airplane': 89.84819589962248, 'Bag': 84.67050441584935, 'Cap': 84.75666579908972, 'Car': 87.19042692732482, 'Chair': 92.58161493844226, 'Earphone': 71.8986889122919, 'Guitar': 94.79896852686961, 'Knife': 93.24535944596042, 'Lamp': 89.47633101064687, 'Laptop': 97.9782900610644, 'Motorbike': 85.34309118158019, 'Mug': 96.58712728325864, 'Pistol': 87.19717674421496, 'Rocket': 78.43467603869419, 'Skateboard': 83.67621036390473, 'Table': 90.92388420529511}
    test_miou_per_class = {'Airplane': 82.26044568555221, 'Bag': 81.22609316840246, 'Cap': 78.55780746196695, 'Car': 78.81272376047963, 'Chair': 85.97079538281722, 'Earphone': 65.90521597269495, 'Guitar': 90.97994989769165, 'Knife': 87.35596426007697, 'Lamp': 81.77103239497399, 'Laptop': 96.02263663166926, 'Motorbike': 72.90563694680506, 'Mug': 94.6191167570525, 'Pistol': 81.57198274136852, 'Rocket': 67.0883255626187, 'Skateboard': 74.06155416980982, 'Table': 84.46920522679369}
==================================================
EPOCH 77 / 100
100%|█████████████████████████████| 876/876 [15:26<00:00,  1.06s/it, data_loading=0.563, iteration=0.218, train_acc=95.94, train_loss_seg=0.101, train_macc=92.54, train_miou=87.90]
Learning rate = 0.000031
100%|██████████████████████████████████████████████████████████████████████| 180/180 [01:26<00:00,  2.07it/s, test_acc=93.56, test_loss_seg=0.110, test_macc=87.67, test_miou=81.41]
==================================================
    test_loss_seg = 0.11060494184494019
    test_acc = 93.56243476524268
    test_macc = 87.67020941029652
    test_miou = 81.41458651996867
    test_acc_per_class = {'Airplane': 91.60814928519062, 'Bag': 96.35532924107143, 'Cap': 91.1709872159091, 'Car': 92.2703223892405, 'Chair': 95.0990156693892, 'Earphone': 92.45256696428571, 'Guitar': 96.60444919418238, 'Knife': 93.1280517578125, 'Lamp': 91.5336128715035, 'Laptop': 98.10335090361446, 'Motorbike': 87.80732996323529, 'Mug': 99.45775082236842, 'Pistol': 95.70201526988636, 'Rocket': 85.49397786458334, 'Skateboard': 94.86517137096774, 'Table': 95.34687546064269}
    test_macc_per_class = {'Airplane': 89.65373251980641, 'Bag': 85.30656105695195, 'Cap': 84.73120211342552, 'Car': 87.02253734268159, 'Chair': 92.40939540447329, 'Earphone': 71.58845636525862, 'Guitar': 94.74508580554156, 'Knife': 93.12303977965223, 'Lamp': 89.20919583795185, 'Laptop': 98.07209733565327, 'Motorbike': 82.99097965568535, 'Mug': 96.18719273917729, 'Pistol': 86.58798129826496, 'Rocket': 76.76136352411852, 'Skateboard': 83.39737662659411, 'Table': 90.93715315950762}
    test_miou_per_class = {'Airplane': 82.20207196539907, 'Bag': 81.50803393860748, 'Cap': 78.65722834122613, 'Car': 78.74743266112239, 'Chair': 86.10100041348142, 'Earphone': 65.5017545887069, 'Guitar': 90.97426555868329, 'Knife': 87.13761380322912, 'Lamp': 81.62452711878434, 'Laptop': 96.24980920736026, 'Motorbike': 72.75294981776521, 'Mug': 94.679280956964, 'Pistol': 81.43662990679782, 'Rocket': 66.76754352678827, 'Skateboard': 73.76460084657239, 'Table': 84.52864166801048}
==================================================
```
BEST
* best_loss_seg: 0.06631797552108765
* test_Cmiou = 97.9368103974044
* test_Imiou = 96.3840916196669
* miou_per_class 
```json
{
'Airplane':0.9688190282061018,
'Cap':0.9915819532701674,
'Car':0.9696876538536436,
'Chair':0.9739581531329329,
'Earphone':0.9742590536947074,
'Guitar':0.984593913468968,
'Knife':0.9901378835358035,
'Lamp':0.9745299819648745,
'Laptop':0.9977452919677099,
'Motorbike':0.979398697349813,
'Mug':0.9969803181260177,
'Pistol':0.985717691380712,
'Rocket':0.9706233311081442,
'Skateboard':0.9947034192761516,
'Table':0.9377851892749126
}
```