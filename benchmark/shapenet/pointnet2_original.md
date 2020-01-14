```
SegmentationModel(
(model): UnetSkipConnectionBlock(
(down): PointNetMSGDown(
    (mlps): ModuleList(
    (0): SharedMLP(
        (layer0): Conv2d(
        (conv): Conv2d(6, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (normlayer): BatchNorm2d(
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): ReLU(inplace=True)
        )
        (layer1): Conv2d(
        (conv): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (normlayer): BatchNorm2d(
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): ReLU(inplace=True)
        )
        (layer2): Conv2d(
        (conv): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (normlayer): BatchNorm2d(
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): ReLU(inplace=True)
        )
    )
    (1): SharedMLP(
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
    )
)
(submodule): UnetSkipConnectionBlock(
    (down): PointNetMSGDown(
    (mlps): ModuleList(
        (0): SharedMLP(
        (layer0): Conv2d(
            (conv): Conv2d(99, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
        (1): SharedMLP(
        (layer0): Conv2d(
            (conv): Conv2d(99, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (conv): Conv2d(259, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
        (1): SharedMLP(
            (layer0): Conv2d(
            (conv): Conv2d(259, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
        (down): PointNetMSGDown(
        (mlps): ModuleList(
            (0): SharedMLP(
            (layer0): Conv2d(
                (conv): Conv2d(515, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
            (layer2): Conv2d(
                (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (activation): ReLU(inplace=True)
            )
            )
            (1): SharedMLP(
            (layer0): Conv2d(
                (conv): Conv2d(515, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (activation): ReLU(inplace=True)
            )
            (layer1): Conv2d(
                (conv): Conv2d(256, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (activation): ReLU(inplace=True)
            )
            (layer2): Conv2d(
                (conv): Conv2d(384, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (activation): ReLU(inplace=True)
            )
            )
        )
        )
        (submodule): Identity()
        (up): DenseFPModule(
        (nn): SharedMLP(
            (layer0): Conv2d(
            (conv): Conv2d(1536, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
            )
            (layer1): Conv2d(
            (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
            )
        )
        )
    )
    (up): DenseFPModule(
        (nn): SharedMLP(
        (layer0): Conv2d(
            (conv): Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
        )
        (layer1): Conv2d(
            (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
        )
        )
    )
    )
    (up): DenseFPModule(
    (nn): SharedMLP(
        (layer0): Conv2d(
        (conv): Conv2d(608, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
        (conv): Conv2d(259, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
Model size = 3031074
EPOCH 48 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.385, iteration=0.233, train_acc=94.05, train_loss_seg=0.159, train_macc=87.97, train_miou=82.02]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.54it/s, test_acc=92.23, test_loss_seg=0.136, test_macc=84.79, test_miou=77.96]
==================================================
test_loss_seg = 0.13694451749324799
test_acc = 92.23585026085786
test_macc = 84.79950890358096
test_miou = 77.96565079826186
test_acc_per_class = {'Airplane': 89.70411727910388, 'Bag': 95.31766000943544, 'Cap': 93.51807688638748, 'Car': 89.47064569179926, 'Chair': 94.66742430346181, 'Earphone': 93.37222202340479, 'Guitar': 95.99519287210445, 'Knife': 90.12021225762645, 'Lamp': 91.30890184954701, 'Laptop': 98.0789277736411, 'Motorbike': 85.25061036909378, 'Mug': 98.85631357511849, 'Pistol': 95.22651620670932, 'Rocket': 75.61040575335169, 'Skateboard': 95.86223310213519, 'Table': 93.41414422080591}
test_macc_per_class = {'Airplane': 89.05569336868405, 'Bag': 78.4997784515664, 'Cap': 86.95107682938534, 'Car': 79.3342427650161, 'Chair': 91.9488475222346, 'Earphone': 67.18101166217906, 'Guitar': 93.46312576570946, 'Knife': 90.09280312919688, 'Lamp': 90.29490647511884, 'Laptop': 98.03925508170154, 'Motorbike': 70.7361264981618, 'Mug': 92.27836321402745, 'Pistol': 86.75048262632475, 'Rocket': 77.03721618374136, 'Skateboard': 82.33039226358963, 'Table': 82.79882062065803}
test_miou_per_class = {'Airplane': 79.14738339090582, 'Bag': 75.34212145313919, 'Cap': 82.67170700364103, 'Car': 71.42114879447522, 'Chair': 84.46583816508183, 'Earphone': 63.69526992385298, 'Guitar': 89.2557350657739, 'Knife': 82.0021289905475, 'Lamp': 80.62561192804749, 'Laptop': 96.20737916023164, 'Motorbike': 63.43150865406491, 'Mug': 89.65174193463838, 'Pistol': 80.42020491343428, 'Rocket': 58.429299294218154, 'Skateboard': 74.54147233332658, 'Table': 76.14186176681062}
==================================================

EPOCH 49 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.380, iteration=0.237, train_acc=93.17, train_loss_seg=0.162, train_macc=85.53, train_miou=79.58]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.53it/s, test_acc=92.79, test_loss_seg=0.207, test_macc=85.96, test_miou=78.94]
==================================================
test_loss_seg = 0.20756636559963226
test_acc = 92.79110422714442
test_macc = 85.96491974995148
test_miou = 78.9492204679207
test_acc_per_class = {'Airplane': 90.339866031032, 'Bag': 95.32267838206332, 'Cap': 92.99537430378551, 'Car': 90.4205426176132, 'Chair': 94.61407877027197, 'Earphone': 91.17564039235057, 'Guitar': 95.763391516255, 'Knife': 92.74354936858153, 'Lamp': 91.02847059537213, 'Laptop': 98.03259319373915, 'Motorbike': 85.50666360294117, 'Mug': 98.9142147835932, 'Pistol': 95.27379512869236, 'Rocket': 81.92855803487717, 'Skateboard': 95.86466809421842, 'Table': 94.73358281892406}
test_macc_per_class = {'Airplane': 87.41651159831837, 'Bag': 79.32357869470358, 'Cap': 85.51607037140221, 'Car': 84.44068402219483, 'Chair': 91.87457355974078, 'Earphone': 69.91041744737808, 'Guitar': 95.13969854805319, 'Knife': 92.74158653238744, 'Lamp': 87.62451926898093, 'Laptop': 98.03761328530142, 'Motorbike': 76.97296361279032, 'Mug': 96.42437150586476, 'Pistol': 82.0142810201944, 'Rocket': 77.27171693972656, 'Skateboard': 82.31274976706993, 'Table': 88.41737982511701}
test_miou_per_class = {'Airplane': 79.82197648503191, 'Bag': 75.69880304902726, 'Cap': 81.09046356655823, 'Car': 75.12879583698628, 'Chair': 84.56523190373643, 'Earphone': 63.27691441973508, 'Guitar': 89.48263627408683, 'Knife': 86.46807415252464, 'Lamp': 78.97610329788016, 'Laptop': 96.12246533721157, 'Motorbike': 64.69246020653127, 'Mug': 90.97638201960982, 'Pistol': 78.15753232454442, 'Rocket': 62.40414517397442, 'Skateboard': 74.0043363465546, 'Table': 82.32120709273829}
==================================================
acc: 92.70168674820557 -> 92.79110422714442, macc: 85.24097836905904 -> 85.96491974995148, miou: 78.74573405858823 -> 78.9492204679207

EPOCH 50 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.384, iteration=0.234, train_acc=94.32, train_loss_seg=0.176, train_macc=88.99, train_miou=83.35]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.56it/s, test_acc=92.29, test_loss_seg=0.165, test_macc=85.37, test_miou=78.09]
==================================================
test_loss_seg = 0.1652200073003769
test_acc = 92.29238665560658
test_macc = 85.3718340854832
test_miou = 78.0937910858786
test_acc_per_class = {'Airplane': 90.51737599111682, 'Bag': 95.24891308605224, 'Cap': 89.32960381511373, 'Car': 90.07707641611876, 'Chair': 94.68981585806063, 'Earphone': 92.71252229254817, 'Guitar': 95.98799572332237, 'Knife': 90.75219248931865, 'Lamp': 90.51337752748881, 'Laptop': 97.994654008894, 'Motorbike': 85.22805606617648, 'Mug': 98.55107723082746, 'Pistol': 95.40589626143672, 'Rocket': 79.17491597219025, 'Skateboard': 95.83175835881929, 'Table': 94.66295539222081}
test_macc_per_class = {'Airplane': 88.47989527704037, 'Bag': 82.5191785209833, 'Cap': 78.8683051950302, 'Car': 84.53980670906698, 'Chair': 91.34572229074648, 'Earphone': 70.70143401175267, 'Guitar': 94.14286430593563, 'Knife': 90.77226281876966, 'Lamp': 86.04294412402595, 'Laptop': 97.88591107710607, 'Motorbike': 69.61391377780947, 'Mug': 95.23815313010682, 'Pistol': 86.31449071821832, 'Rocket': 75.89949610823254, 'Skateboard': 85.08841651654639, 'Table': 88.4965507863601}
test_miou_per_class = {'Airplane': 80.52114438488962, 'Bag': 76.82976451426644, 'Cap': 72.57016767622437, 'Car': 74.94365778078654, 'Chair': 84.84112938807358, 'Earphone': 65.66462942091108, 'Guitar': 89.60528053715778, 'Knife': 82.99331545729551, 'Lamp': 77.42326536441087, 'Laptop': 96.03054086257659, 'Motorbike': 60.94387700493219, 'Mug': 88.2638090762702, 'Pistol': 80.60826860134073, 'Rocket': 60.28761040524289, 'Skateboard': 75.8650629102917, 'Table': 82.1091339893873}
==================================================

EPOCH 51 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.388, iteration=0.235, train_acc=94.10, train_loss_seg=0.152, train_macc=87.76, train_miou=82.41]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.53it/s, test_acc=92.30, test_loss_seg=0.131, test_macc=85.57, test_miou=77.71]
==================================================
test_loss_seg = 0.13169628381729126
test_acc = 92.30253826154946
test_macc = 85.57594685748823
test_miou = 77.71107208526453
test_acc_per_class = {'Airplane': 90.34803106134949, 'Bag': 95.6486243775002, 'Cap': 90.68905563689604, 'Car': 90.5320358982051, 'Chair': 94.67876175416285, 'Earphone': 91.15521556738464, 'Guitar': 96.04791417692671, 'Knife': 92.985041445676, 'Lamp': 90.9541032858372, 'Laptop': 98.03972926476453, 'Motorbike': 83.8311887254902, 'Mug': 98.40284417630662, 'Pistol': 95.19187385031873, 'Rocket': 79.40609876427196, 'Skateboard': 93.98382536655927, 'Table': 94.94626883314193}
test_macc_per_class = {'Airplane': 86.52735465073414, 'Bag': 80.6574531627642, 'Cap': 82.07353082843679, 'Car': 84.68146490443571, 'Chair': 92.48674835550932, 'Earphone': 66.76643058996504, 'Guitar': 93.93821357385926, 'Knife': 92.96787931912553, 'Lamp': 90.39581309806343, 'Laptop': 97.9785764151673, 'Motorbike': 74.33940760220618, 'Mug': 96.58597601748644, 'Pistol': 82.07282293319432, 'Rocket': 74.6669920586862, 'Skateboard': 83.89049062906301, 'Table': 89.18599558111477}
test_miou_per_class = {'Airplane': 79.25700993583148, 'Bag': 77.17118920698954, 'Cap': 76.32024759483734, 'Car': 75.48295601943431, 'Chair': 84.6007691294542, 'Earphone': 62.65546257671383, 'Guitar': 89.67407505427568, 'Knife': 86.87005473668994, 'Lamp': 78.6563227564693, 'Laptop': 96.13045941388532, 'Motorbike': 61.629574617972814, 'Mug': 87.60389400723294, 'Pistol': 78.28179712458501, 'Rocket': 56.91936457112187, 'Skateboard': 68.92455272885152, 'Table': 83.19942388988751}
==================================================

EPOCH 52 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.381, iteration=0.239, train_acc=93.44, train_loss_seg=0.156, train_macc=85.99, train_miou=80.00]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.53it/s, test_acc=92.52, test_loss_seg=0.244, test_macc=85.42, test_miou=78.32]
==================================================
test_loss_seg = 0.24458159506320953
test_acc = 92.52308038323106
test_macc = 85.42024985954077
test_miou = 78.32203287681014
test_acc_per_class = {'Airplane': 90.06504528174663, 'Bag': 95.07698304424089, 'Cap': 93.92058420812415, 'Car': 90.58027338933475, 'Chair': 94.55317232083789, 'Earphone': 89.2060786820054, 'Guitar': 96.03199724649127, 'Knife': 92.26445241563408, 'Lamp': 90.9055980191433, 'Laptop': 98.05112373278344, 'Motorbike': 85.45496323529412, 'Mug': 98.76652341837567, 'Pistol': 95.05207440180175, 'Rocket': 80.72174307762143, 'Skateboard': 95.39312657166806, 'Table': 94.32554708659413}
test_macc_per_class = {'Airplane': 89.75794360716219, 'Bag': 77.24952635881918, 'Cap': 89.07808913201016, 'Car': 85.05985074521887, 'Chair': 92.113677876953, 'Earphone': 66.24017238185394, 'Guitar': 94.41852587976362, 'Knife': 92.28560147688822, 'Lamp': 87.83610488714987, 'Laptop': 98.05410805504697, 'Motorbike': 76.6928832610052, 'Mug': 91.84197272749425, 'Pistol': 85.04768114235043, 'Rocket': 64.84899085769237, 'Skateboard': 88.82023000550019, 'Table': 87.37863935774385}
test_miou_per_class = {'Airplane': 80.29926075338587, 'Bag': 74.48914866565175, 'Cap': 84.60454310617868, 'Car': 75.56709662864164, 'Chair': 84.24394487393741, 'Earphone': 57.918504190153186, 'Guitar': 89.69597108964273, 'Knife': 85.63974729158106, 'Lamp': 79.14087036222391, 'Laptop': 96.1577958772321, 'Motorbike': 65.61807209040809, 'Mug': 89.03868098582677, 'Pistol': 79.38524752311166, 'Rocket': 55.20726428048827, 'Skateboard': 75.65513021792866, 'Table': 80.49124809257036}
==================================================

EPOCH 53 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.384, iteration=0.233, train_acc=93.94, train_loss_seg=0.160, train_macc=89.32, train_miou=82.76]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.53it/s, test_acc=92.28, test_loss_seg=0.218, test_macc=83.29, test_miou=76.89]
==================================================
test_loss_seg = 0.21879932284355164
test_acc = 92.28455012835606
test_macc = 83.29175643670159
test_miou = 76.89567246730873
test_acc_per_class = {'Airplane': 89.097157274923, 'Bag': 94.27200365616788, 'Cap': 92.29469260984744, 'Car': 90.52477588476391, 'Chair': 94.60622499662877, 'Earphone': 93.69324806557609, 'Guitar': 95.7923732888292, 'Knife': 91.6550508232928, 'Lamp': 91.06529463747593, 'Laptop': 98.01892601160907, 'Motorbike': 83.61482928323855, 'Mug': 99.07308518914071, 'Pistol': 94.94905434575938, 'Rocket': 77.89213483146067, 'Skateboard': 95.6040416369986, 'Table': 94.39990951798505}
test_macc_per_class = {'Airplane': 87.83500261446726, 'Bag': 70.65529787837572, 'Cap': 85.27638738389125, 'Car': 86.0815578785282, 'Chair': 90.50005506298409, 'Earphone': 70.49780635777353, 'Guitar': 92.66983457873842, 'Knife': 91.55995591867094, 'Lamp': 86.73271608105291, 'Laptop': 98.05226658266253, 'Motorbike': 58.42967084546967, 'Mug': 94.14202190521141, 'Pistol': 85.42626532944831, 'Rocket': 59.559729286908016, 'Skateboard': 89.18749236145884, 'Table': 86.06204292158411}
test_miou_per_class = {'Airplane': 78.03203788877818, 'Bag': 67.62963073915711, 'Cap': 80.36665279836225, 'Car': 75.74939564226304, 'Chair': 84.44164114196052, 'Earphone': 66.7864610318397, 'Guitar': 88.4622434312297, 'Knife': 84.52636186942215, 'Lamp': 79.06409749471304, 'Laptop': 96.09673924131256, 'Motorbike': 52.876247067644385, 'Mug': 91.54220442012154, 'Pistol': 79.24297871444762, 'Rocket': 48.77056847265169, 'Skateboard': 76.50636910702525, 'Table': 80.23713041601096}
==================================================

EPOCH 54 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.381, iteration=0.235, train_acc=94.35, train_loss_seg=0.158, train_macc=87.46, train_miou=81.17]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.56it/s, test_acc=92.70, test_loss_seg=0.147, test_macc=85.39, test_miou=78.86]
==================================================
test_loss_seg = 0.14782103896141052
test_acc = 92.70841655397545
test_macc = 85.39474220295739
test_miou = 78.86685983775284
test_acc_per_class = {'Airplane': 90.20578445451397, 'Bag': 95.30989543870282, 'Cap': 93.80359612724757, 'Car': 90.80999549052233, 'Chair': 94.65351987613766, 'Earphone': 89.15820853147845, 'Guitar': 95.63629052214799, 'Knife': 92.92119100214678, 'Lamp': 90.92568859866029, 'Laptop': 97.950462548493, 'Motorbike': 84.92647058823529, 'Mug': 98.8716236231923, 'Pistol': 95.64125601861451, 'Rocket': 81.9056192551881, 'Skateboard': 95.85904381557197, 'Table': 94.75601897275425}
test_macc_per_class = {'Airplane': 89.33722816702412, 'Bag': 77.67234930266444, 'Cap': 89.57584707715708, 'Car': 83.42935619892506, 'Chair': 89.82941460895276, 'Earphone': 70.21549345371993, 'Guitar': 94.74457482800234, 'Knife': 92.89936770436802, 'Lamp': 86.92909254626537, 'Laptop': 97.90382633628482, 'Motorbike': 72.51099350170979, 'Mug': 94.68265745840587, 'Pistol': 84.32975084783494, 'Rocket': 70.07828003288536, 'Skateboard': 83.09161317367736, 'Table': 89.0860300094408}
test_miou_per_class = {'Airplane': 80.20511447231856, 'Bag': 74.75810021769003, 'Cap': 84.35950434935143, 'Car': 75.51920843972802, 'Chair': 84.37251016899407, 'Earphone': 60.90501094912998, 'Guitar': 89.35835899654832, 'Knife': 86.76471529962721, 'Lamp': 79.07201411558295, 'Laptop': 95.95943060756204, 'Motorbike': 63.48353303903719, 'Mug': 90.3009535478671, 'Pistol': 80.0781863629592, 'Rocket': 59.86021754011143, 'Skateboard': 74.42925145858761, 'Table': 82.44364783895008}
==================================================

EPOCH 55 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.384, iteration=0.232, train_acc=94.21, train_loss_seg=0.148, train_macc=87.64, train_miou=81.65]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.55it/s, test_acc=92.76, test_loss_seg=0.114, test_macc=84.99, test_miou=78.48]
==================================================
test_loss_seg = 0.11473225802183151
test_acc = 92.76388727152874
test_macc = 84.99094419022491
test_miou = 78.4887924664926
test_acc_per_class = {'Airplane': 90.51290505787631, 'Bag': 95.53972727965262, 'Cap': 91.51053339309728, 'Car': 90.73405872096818, 'Chair': 94.7046705711132, 'Earphone': 92.77453580901856, 'Guitar': 95.89038546825911, 'Knife': 92.7643329727554, 'Lamp': 91.43131621316302, 'Laptop': 97.99136676864497, 'Motorbike': 85.00689338235294, 'Mug': 98.76915683238106, 'Pistol': 95.29491014545704, 'Rocket': 80.29891908722921, 'Skateboard': 96.00307712890495, 'Table': 94.9954075135861}
test_macc_per_class = {'Airplane': 89.64379762769057, 'Bag': 77.79714897785786, 'Cap': 84.53038691626008, 'Car': 83.11800883287496, 'Chair': 91.30950829168111, 'Earphone': 68.852458121408, 'Guitar': 93.63406099530364, 'Knife': 92.72243206337903, 'Lamp': 87.50706863178952, 'Laptop': 97.98005178063072, 'Motorbike': 82.00853087584275, 'Mug': 92.51050207682174, 'Pistol': 82.72196433429157, 'Rocket': 60.76742367130712, 'Skateboard': 85.39692688458388, 'Table': 89.35483696187596}
test_miou_per_class = {'Airplane': 80.76288573962451, 'Bag': 75.31753431516151, 'Cap': 79.09371386396185, 'Car': 75.4025418767961, 'Chair': 84.86696999862325, 'Earphone': 63.61448896480982, 'Guitar': 89.46014559710316, 'Knife': 86.48419877115467, 'Lamp': 79.6225557688306, 'Laptop': 96.03908977717349, 'Motorbike': 65.1393804100078, 'Mug': 89.10245666763059, 'Pistol': 78.3826165594264, 'Rocket': 52.61719490356889, 'Skateboard': 76.66048120381247, 'Table': 83.25442504619667}
==================================================

EPOCH 56 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.383, iteration=0.230, train_acc=94.15, train_loss_seg=0.146, train_macc=88.01, train_miou=82.42]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.54it/s, test_acc=93.13, test_loss_seg=0.170, test_macc=85.85, test_miou=79.73]
==================================================
test_loss_seg = 0.1704130470752716
test_acc = 93.131845465478
test_macc = 85.85748081514947
test_miou = 79.73526909626351
test_acc_per_class = {'Airplane': 90.6807388016883, 'Bag': 95.68952351863165, 'Cap': 94.00926013417745, 'Car': 90.79766035368047, 'Chair': 94.7150087348264, 'Earphone': 93.3110129648098, 'Guitar': 96.10476933658629, 'Knife': 92.38905175899717, 'Lamp': 91.1809138888344, 'Laptop': 98.04983229516053, 'Motorbike': 85.7421875, 'Mug': 99.17272837690118, 'Pistol': 96.05001287784776, 'Rocket': 81.92915482954545, 'Skateboard': 95.64049272570588, 'Table': 94.6471793502551}
test_macc_per_class = {'Airplane': 89.03526298975807, 'Bag': 79.83077057743174, 'Cap': 87.89109201654651, 'Car': 84.52361390796307, 'Chair': 91.71165053666688, 'Earphone': 70.55116065984252, 'Guitar': 94.36227811947853, 'Knife': 92.3933661503129, 'Lamp': 89.87764402131914, 'Laptop': 98.05139066739888, 'Motorbike': 72.38545392779888, 'Mug': 93.78183875722128, 'Pistol': 87.11018354241989, 'Rocket': 73.95618377849392, 'Skateboard': 80.65354807599972, 'Table': 87.60425531373998}
test_miou_per_class = {'Airplane': 80.82132461618077, 'Bag': 77.0492067114566, 'Cap': 83.87532565294975, 'Car': 75.77676197184725, 'Chair': 84.64668413485094, 'Earphone': 66.10543307928207, 'Guitar': 89.8792547312869, 'Knife': 85.85142350123478, 'Lamp': 80.97976999177779, 'Laptop': 96.1533486114766, 'Motorbike': 62.802923152925615, 'Mug': 92.39192532531273, 'Pistol': 82.4934666466875, 'Rocket': 61.442860126816136, 'Skateboard': 73.54590713398014, 'Table': 81.9486901521505}
==================================================
acc: 92.79110422714442 -> 93.131845465478, miou: 78.9492204679207 -> 79.73526909626351

EPOCH 57 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.382, iteration=0.235, train_acc=94.63, train_loss_seg=0.154, train_macc=89.64, train_miou=83.89]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.53it/s, test_acc=91.76, test_loss_seg=0.205, test_macc=81.62, test_miou=75.66]
==================================================
test_loss_seg = 0.20528805255889893
test_acc = 91.7685235274328
test_macc = 81.62028500187428
test_miou = 75.66130203912586
test_acc_per_class = {'Airplane': 90.0271159216932, 'Bag': 95.2878473837824, 'Cap': 94.53357671246036, 'Car': 89.27724086375054, 'Chair': 94.47114824950738, 'Earphone': 93.43918726911055, 'Guitar': 95.5221244969593, 'Knife': 93.12268937000732, 'Lamp': 89.05066470335426, 'Laptop': 98.0543671585829, 'Motorbike': 73.32565615853366, 'Mug': 98.9611030399386, 'Pistol': 95.02194545522752, 'Rocket': 78.1401126935534, 'Skateboard': 95.50878001415334, 'Table': 94.55281694831034}
test_macc_per_class = {'Airplane': 88.01839109939762, 'Bag': 77.34741242280516, 'Cap': 88.68421414094203, 'Car': 79.0118369298465, 'Chair': 90.20547234025382, 'Earphone': 65.95617037556207, 'Guitar': 92.91539215020812, 'Knife': 93.06447801068707, 'Lamp': 84.61425059644807, 'Laptop': 97.9849314653962, 'Motorbike': 44.52983530843859, 'Mug': 92.33679669694442, 'Pistol': 85.83820861108329, 'Rocket': 57.5824106030692, 'Skateboard': 78.9636076979208, 'Table': 88.87115158098553}
test_miou_per_class = {'Airplane': 79.63633038198137, 'Bag': 74.62045075702594, 'Cap': 84.98380516810285, 'Car': 71.65525420063248, 'Chair': 84.25380630052418, 'Earphone': 61.903814239180576, 'Guitar': 87.97374608648227, 'Knife': 87.1048540677627, 'Lamp': 75.07946411416098, 'Laptop': 96.15521001363123, 'Motorbike': 35.4036030678853, 'Mug': 90.46514407536317, 'Pistol': 80.03612081415449, 'Rocket': 48.787119370921936, 'Skateboard': 70.63609186444943, 'Table': 81.88601810375475}
==================================================

EPOCH 58 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.378, iteration=0.234, train_acc=93.89, train_loss_seg=0.163, train_macc=88.02, train_miou=82.11]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.54it/s, test_acc=92.32, test_loss_seg=0.136, test_macc=84.48, test_miou=77.88]
==================================================
test_loss_seg = 0.1367768943309784
test_acc = 92.3241045142741
test_macc = 84.48316738820772
test_miou = 77.88193934999433
test_acc_per_class = {'Airplane': 90.38977242696912, 'Bag': 95.42406461937057, 'Cap': 90.35528316667435, 'Car': 90.32707463273995, 'Chair': 94.8348626103164, 'Earphone': 88.57102888717912, 'Guitar': 96.06606244453309, 'Knife': 91.01418747289512, 'Lamp': 90.49472577247066, 'Laptop': 98.09924326122338, 'Motorbike': 84.62739015118873, 'Mug': 99.0565786672344, 'Pistol': 95.09044828869048, 'Rocket': 82.1688565340909, 'Skateboard': 96.01962829346692, 'Table': 94.64646499934206}
test_macc_per_class = {'Airplane': 88.3673876273744, 'Bag': 79.02780689061272, 'Cap': 80.73647287891738, 'Car': 85.01993625877111, 'Chair': 91.78737551004626, 'Earphone': 72.08383583992742, 'Guitar': 93.8093332882249, 'Knife': 90.95062284583116, 'Lamp': 80.46136257068255, 'Laptop': 98.04320364553764, 'Motorbike': 70.66370850047313, 'Mug': 92.69680402100815, 'Pistol': 82.50992625861811, 'Rocket': 69.89601073744339, 'Skateboard': 86.85293297614055, 'Table': 88.82395836171432}
test_miou_per_class = {'Airplane': 80.21990834751858, 'Bag': 75.05239143206568, 'Cap': 74.92773494895337, 'Car': 74.81764463036242, 'Chair': 85.08846766867705, 'Earphone': 61.253054361909165, 'Guitar': 89.62853133441186, 'Knife': 83.43690668700854, 'Lamp': 73.84157124537538, 'Laptop': 96.24501395825168, 'Motorbike': 62.50030773459066, 'Mug': 91.23744403089343, 'Pistol': 78.79193002937771, 'Rocket': 60.117247829692154, 'Skateboard': 76.82235856838162, 'Table': 82.13051679244002}
==================================================

EPOCH 59 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.381, iteration=0.233, train_acc=94.04, train_loss_seg=0.159, train_macc=83.97, train_miou=77.88]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.56it/s, test_acc=92.41, test_loss_seg=0.163, test_macc=84.75, test_miou=78.12]
==================================================
test_loss_seg = 0.1634737253189087
test_acc = 92.41088380735549
test_macc = 84.75909672876756
test_miou = 78.12219814319732
test_acc_per_class = {'Airplane': 90.70844109289237, 'Bag': 94.67969598262758, 'Cap': 82.51949117767748, 'Car': 90.47268576703456, 'Chair': 94.71678452430471, 'Earphone': 92.88040784535863, 'Guitar': 96.01214418912879, 'Knife': 93.08735525473911, 'Lamp': 91.28805495416032, 'Laptop': 98.02742993440667, 'Motorbike': 85.0174249387255, 'Mug': 98.96698100534644, 'Pistol': 95.73407523329276, 'Rocket': 83.77759457726481, 'Skateboard': 96.0455009782364, 'Table': 94.64007346249166}
test_macc_per_class = {'Airplane': 89.69587262776557, 'Bag': 76.13990929238798, 'Cap': 65.85835789891506, 'Car': 82.77772893285514, 'Chair': 91.32075425290816, 'Earphone': 71.40027876077686, 'Guitar': 94.21111214595132, 'Knife': 93.07473970527742, 'Lamp': 88.7836331022031, 'Laptop': 98.0394441686467, 'Motorbike': 67.69972636262328, 'Mug': 91.60944830405879, 'Pistol': 87.33743454144974, 'Rocket': 80.70352048331458, 'Skateboard': 90.14179705565778, 'Table': 87.35179002548924}
test_miou_per_class = {'Airplane': 81.0527011723585, 'Bag': 72.76698387523129, 'Cap': 56.34660197812741, 'Car': 74.87404303072496, 'Chair': 84.9078577141909, 'Earphone': 66.20190100308459, 'Guitar': 89.83244657947755, 'Knife': 87.06001101032281, 'Lamp': 81.24693337126368, 'Laptop': 96.11288612982787, 'Motorbike': 61.183761459501675, 'Mug': 90.41321482948372, 'Pistol': 82.1058138124949, 'Rocket': 66.32405764003096, 'Skateboard': 78.09624415826283, 'Table': 81.4297125267735}
==================================================

EPOCH 60 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.383, iteration=0.235, train_acc=95.25, train_loss_seg=0.142, train_macc=89.69, train_miou=84.49]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.55it/s, test_acc=93.00, test_loss_seg=0.119, test_macc=86.31, test_miou=79.71]
==================================================
test_loss_seg = 0.1197643056511879
test_acc = 93.00543872671068
test_macc = 86.31261257112098
test_miou = 79.7159002106574
test_acc_per_class = {'Airplane': 90.49692737793758, 'Bag': 95.67549064354176, 'Cap': 93.18878261677035, 'Car': 90.92409807366447, 'Chair': 94.74400178755373, 'Earphone': 92.97708311331714, 'Guitar': 96.16711738241561, 'Knife': 90.88628121967413, 'Lamp': 90.97871437663531, 'Laptop': 98.12346503254727, 'Motorbike': 86.19216809201392, 'Mug': 99.08096397863207, 'Pistol': 95.4342269581354, 'Rocket': 82.71791352093342, 'Skateboard': 95.64771025127489, 'Table': 94.85207520232336}
test_macc_per_class = {'Airplane': 88.07244940073059, 'Bag': 79.08539971093087, 'Cap': 86.5726527113506, 'Car': 84.15077764917022, 'Chair': 90.10236786494403, 'Earphone': 70.58711172585525, 'Guitar': 94.65857290787115, 'Knife': 90.77420892412817, 'Lamp': 89.7306756323798, 'Laptop': 98.0672191907832, 'Motorbike': 76.32556072472309, 'Mug': 93.73608203147734, 'Pistol': 83.89307187104068, 'Rocket': 78.28079305482115, 'Skateboard': 88.05767263482815, 'Table': 88.90718510290144}
test_miou_per_class = {'Airplane': 80.07059897309891, 'Bag': 76.2288523510962, 'Cap': 82.1583135258815, 'Car': 76.02430410269227, 'Chair': 84.77664137228423, 'Earphone': 66.3719750625665, 'Guitar': 90.14210215498714, 'Knife': 83.2156064370032, 'Lamp': 80.11172683274064, 'Laptop': 96.29100046370984, 'Motorbike': 66.23227577482412, 'Mug': 91.64622945990473, 'Pistol': 79.59677033352216, 'Rocket': 63.939738496531284, 'Skateboard': 75.95270912800522, 'Table': 82.69555890167055}
==================================================
macc: 85.96491974995148 -> 86.31261257112098

EPOCH 61 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.385, iteration=0.234, train_acc=94.25, train_loss_seg=0.141, train_macc=89.02, train_miou=83.23]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.53it/s, test_acc=92.87, test_loss_seg=0.117, test_macc=86.37, test_miou=79.36]
==================================================
test_loss_seg = 0.117872454226017
test_acc = 92.87962886820966
test_macc = 86.37245826583312
test_miou = 79.3639047491631
test_acc_per_class = {'Airplane': 90.64753352221007, 'Bag': 96.1657292244316, 'Cap': 93.02503710086648, 'Car': 90.28284415511533, 'Chair': 94.74920815877259, 'Earphone': 93.10124106539597, 'Guitar': 95.9963258298477, 'Knife': 93.01948153671665, 'Lamp': 89.29875517167608, 'Laptop': 97.97121906288503, 'Motorbike': 84.95410377539655, 'Mug': 99.14241677754212, 'Pistol': 95.78153867207992, 'Rocket': 81.78671282119558, 'Skateboard': 95.37026041224871, 'Table': 94.78165460497449}
test_macc_per_class = {'Airplane': 90.02966722009702, 'Bag': 82.22388519649502, 'Cap': 85.1270992562751, 'Car': 86.40296727092496, 'Chair': 91.43559465167858, 'Earphone': 71.48308094620037, 'Guitar': 94.35807513105803, 'Knife': 92.94825526669895, 'Lamp': 87.12594658501436, 'Laptop': 98.0140601260765, 'Motorbike': 68.08880732189975, 'Mug': 94.43724353875132, 'Pistol': 89.74532884940677, 'Rocket': 76.45385008466378, 'Skateboard': 85.12947215438737, 'Table': 88.95599865370176}
test_miou_per_class = {'Airplane': 80.42430945412644, 'Bag': 79.27822306261896, 'Cap': 80.84125127827, 'Car': 75.30213894422036, 'Chair': 84.83927250382426, 'Earphone': 66.87621469276654, 'Guitar': 89.5756431750184, 'Knife': 86.91965548764463, 'Lamp': 76.90391243140708, 'Laptop': 96.00449779472928, 'Motorbike': 59.143976596709535, 'Mug': 92.1192397585591, 'Pistol': 82.83754438299337, 'Rocket': 62.96092420073474, 'Skateboard': 73.19709973248315, 'Table': 82.59857249050353}
==================================================
macc: 86.31261257112098 -> 86.37245826583312

EPOCH 62 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.383, iteration=0.238, train_acc=95.16, train_loss_seg=0.152, train_macc=89.55, train_miou=84.30]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.54it/s, test_acc=92.30, test_loss_seg=0.162, test_macc=85.64, test_miou=77.99]
==================================================
test_loss_seg = 0.16277746856212616
test_acc = 92.30130056492402
test_macc = 85.64476131997272
test_miou = 77.9993940335307
test_acc_per_class = {'Airplane': 90.61531986409399, 'Bag': 95.40193309733105, 'Cap': 93.74284624189241, 'Car': 90.36542469833535, 'Chair': 94.65472609460114, 'Earphone': 88.70618628683144, 'Guitar': 95.81750234542774, 'Knife': 91.44847180154697, 'Lamp': 89.90037250144341, 'Laptop': 98.04484272827074, 'Motorbike': 84.14617800245098, 'Mug': 98.43220620474324, 'Pistol': 95.85943947830332, 'Rocket': 78.87972289321362, 'Skateboard': 96.17055567234252, 'Table': 94.63508112795643}
test_macc_per_class = {'Airplane': 90.10221569098812, 'Bag': 80.41686738696579, 'Cap': 87.1395359564963, 'Car': 85.47932000102773, 'Chair': 91.9122744272451, 'Earphone': 66.06827143742566, 'Guitar': 94.66175962105675, 'Knife': 91.48323691517601, 'Lamp': 88.98651574631357, 'Laptop': 98.07414305256074, 'Motorbike': 65.39644212855687, 'Mug': 91.78077141439076, 'Pistol': 86.96048937205255, 'Rocket': 74.94616351925288, 'Skateboard': 87.272595348101, 'Table': 89.63557910195377}
test_miou_per_class = {'Airplane': 80.65748607313186, 'Bag': 76.91903392736643, 'Cap': 83.09091178967446, 'Car': 75.13521540140496, 'Chair': 84.53914600641119, 'Earphone': 56.77214033236412, 'Guitar': 88.75785828033962, 'Knife': 84.23641400053741, 'Lamp': 78.56052658545153, 'Laptop': 96.14523333364883, 'Motorbike': 57.30329853913167, 'Mug': 86.64303974718379, 'Pistol': 82.36803775136416, 'Rocket': 57.426860633070994, 'Skateboard': 76.98171077373424, 'Table': 82.45339136167587}
==================================================

EPOCH 63 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.378, iteration=0.236, train_acc=94.25, train_loss_seg=0.144, train_macc=89.69, train_miou=82.14]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.56it/s, test_acc=92.51, test_loss_seg=0.173, test_macc=85.99, test_miou=78.63]
==================================================
test_loss_seg = 0.17372559010982513
test_acc = 92.51545705874182
test_macc = 85.99037318017011
test_miou = 78.63202433378295
test_acc_per_class = {'Airplane': 90.29297084186058, 'Bag': 95.50779734657459, 'Cap': 91.1312592047128, 'Car': 90.83917284432306, 'Chair': 94.39041396113605, 'Earphone': 92.10470161462409, 'Guitar': 95.68993878347922, 'Knife': 92.681007178719, 'Lamp': 89.96344401326516, 'Laptop': 98.10643727461152, 'Motorbike': 84.4027650122549, 'Mug': 98.9568522185162, 'Pistol': 94.59634963705095, 'Rocket': 80.71506472568637, 'Skateboard': 96.1320085166785, 'Table': 94.73712976637634}
test_macc_per_class = {'Airplane': 88.80203654948882, 'Bag': 83.22971032949896, 'Cap': 82.76164980011175, 'Car': 84.95118439833603, 'Chair': 92.93238182097652, 'Earphone': 71.49732417728913, 'Guitar': 93.22771690188901, 'Knife': 92.64098407567747, 'Lamp': 86.72124446138774, 'Laptop': 98.0807606656613, 'Motorbike': 77.53940542763083, 'Mug': 95.47560989404302, 'Pistol': 80.90224248656938, 'Rocket': 69.46681761036344, 'Skateboard': 88.51471021695416, 'Table': 89.10219206684422}
test_miou_per_class = {'Airplane': 80.00748325059912, 'Bag': 78.30002696922848, 'Cap': 77.26384824189304, 'Car': 76.07853352480315, 'Chair': 83.20806173314702, 'Earphone': 64.32530644496957, 'Guitar': 88.85517410999725, 'Knife': 86.32510533325356, 'Lamp': 76.65088271840746, 'Laptop': 96.26256565374307, 'Motorbike': 63.504370455340286, 'Mug': 90.94374614747775, 'Pistol': 77.19127360849816, 'Rocket': 58.97784226592343, 'Skateboard': 77.6376337796293, 'Table': 82.58053510361633}
==================================================

EPOCH 64 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.384, iteration=0.236, train_acc=95.44, train_loss_seg=0.139, train_macc=91.27, train_miou=85.95]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.54it/s, test_acc=92.62, test_loss_seg=0.128, test_macc=87.04, test_miou=79.39]
==================================================
test_loss_seg = 0.1284656822681427
test_acc = 92.62450610265954
test_macc = 87.04467619836664
test_miou = 79.39879005088274
test_acc_per_class = {'Airplane': 90.57067610791766, 'Bag': 95.20990106573971, 'Cap': 94.34173669467786, 'Car': 90.61873263824009, 'Chair': 94.65992209175238, 'Earphone': 89.12639079007405, 'Guitar': 96.12340210256157, 'Knife': 92.09059364391386, 'Lamp': 91.25982490893986, 'Laptop': 98.08305178156374, 'Motorbike': 86.24549911897648, 'Mug': 99.17527981577679, 'Pistol': 95.34309198374584, 'Rocket': 78.4885530117804, 'Skateboard': 95.93776863822544, 'Table': 94.71767324866696}
test_macc_per_class = {'Airplane': 90.2659436666069, 'Bag': 82.48853680386745, 'Cap': 90.03288776004379, 'Car': 80.86426540176421, 'Chair': 91.63363530660936, 'Earphone': 69.1942459917266, 'Guitar': 94.0334339165522, 'Knife': 92.03739758258918, 'Lamp': 87.26279680352299, 'Laptop': 98.09902714673188, 'Motorbike': 82.829667201505, 'Mug': 93.92011413266259, 'Pistol': 85.76961353695981, 'Rocket': 80.69457562803427, 'Skateboard': 85.19425272857796, 'Table': 88.3944255661122}
test_miou_per_class = {'Airplane': 80.97045202417297, 'Bag': 76.71064185778131, 'Cap': 85.33355572017486, 'Car': 74.15851731187432, 'Chair': 84.84536686396491, 'Earphone': 60.72189937356125, 'Guitar': 89.8727528321944, 'Knife': 85.30696305946599, 'Lamp': 79.0063541128735, 'Laptop': 96.21733553987501, 'Motorbike': 67.86411386720678, 'Mug': 92.40013449498099, 'Pistol': 80.25102804815326, 'Rocket': 59.1144340941628, 'Skateboard': 75.49703665183377, 'Table': 82.11005496184787}
==================================================
macc: 86.37245826583312 -> 87.04467619836664

EPOCH 65 / 100
100%|█████████████████████████████| 438/438 [05:16<00:00,  1.38it/s, data_loading=0.381, iteration=0.236, train_acc=94.70, train_loss_seg=0.140, train_macc=90.18, train_miou=84.57]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.52it/s, test_acc=93.02, test_loss_seg=0.199, test_macc=87.32, test_miou=80.11]
==================================================
test_loss_seg = 0.19943374395370483
test_acc = 93.0218022902602
test_macc = 87.32110078339136
test_miou = 80.11268496674514
test_acc_per_class = {'Airplane': 89.94987228050576, 'Bag': 94.98063133602571, 'Cap': 97.32160905599353, 'Car': 90.6327983234483, 'Chair': 94.4580972681486, 'Earphone': 89.94761084350058, 'Guitar': 96.18334511313236, 'Knife': 93.44226402347098, 'Lamp': 89.18570660697587, 'Laptop': 97.98406530961972, 'Motorbike': 85.43390012254902, 'Mug': 99.11247721315183, 'Pistol': 95.96283316425344, 'Rocket': 83.18651685393257, 'Skateboard': 96.16550177798798, 'Table': 94.40160735146681}
test_macc_per_class = {'Airplane': 88.66018079119769, 'Bag': 85.84981328439896, 'Cap': 94.16714413037943, 'Car': 86.65523427756689, 'Chair': 91.16160806137394, 'Earphone': 69.87003987576449, 'Guitar': 94.80076702417223, 'Knife': 93.44108776099223, 'Lamp': 89.28577501783025, 'Laptop': 97.89655253153242, 'Motorbike': 77.060544152636, 'Mug': 93.80158296382861, 'Pistol': 86.49047941914517, 'Rocket': 71.76967382981984, 'Skateboard': 89.13432069470001, 'Table': 87.09280871892369}
test_miou_per_class = {'Airplane': 79.7431071038529, 'Bag': 77.57803842979482, 'Cap': 91.97972628911354, 'Car': 75.98652591806712, 'Chair': 84.22421840542545, 'Earphone': 62.01503750088827, 'Guitar': 90.1036343759137, 'Knife': 87.69090232844347, 'Lamp': 74.92740297147697, 'Laptop': 96.02037039498533, 'Motorbike': 66.49139559417361, 'Mug': 91.79938856044028, 'Pistol': 82.37966310362701, 'Rocket': 61.81239908487637, 'Skateboard': 78.18297916674304, 'Table': 80.86817024010018}
==================================================
macc: 87.04467619836664 -> 87.32110078339136, miou: 79.73526909626351 -> 80.11268496674514

EPOCH 66 / 100
100%|█████████████████████████████| 438/438 [05:16<00:00,  1.38it/s, data_loading=0.381, iteration=0.234, train_acc=94.10, train_loss_seg=0.150, train_macc=88.17, train_miou=82.38]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.54it/s, test_acc=92.87, test_loss_seg=0.145, test_macc=85.55, test_miou=79.13]
==================================================
test_loss_seg = 0.1450541466474533
test_acc = 92.8764114927877
test_macc = 85.55259893955437
test_miou = 79.13305419290589
test_acc_per_class = {'Airplane': 90.90814223474473, 'Bag': 95.9128777972877, 'Cap': 93.42738668318955, 'Car': 90.60044089623993, 'Chair': 94.94618492247004, 'Earphone': 93.58509176735829, 'Guitar': 95.04879063959673, 'Knife': 93.06660213834957, 'Lamp': 90.22191450204508, 'Laptop': 98.07786323257616, 'Motorbike': 83.38280068169198, 'Mug': 99.16382454800437, 'Pistol': 95.22044190727452, 'Rocket': 81.50421743205249, 'Skateboard': 96.23164932111104, 'Table': 94.72435518061087}
test_macc_per_class = {'Airplane': 88.55651313705624, 'Bag': 80.02302079139136, 'Cap': 87.02999105686911, 'Car': 87.85473805574723, 'Chair': 91.77183190506445, 'Earphone': 65.18755400597982, 'Guitar': 95.00329897565335, 'Knife': 93.06530673234785, 'Lamp': 89.00983370746202, 'Laptop': 98.01812595305381, 'Motorbike': 74.00351068512336, 'Mug': 94.70482072648332, 'Pistol': 83.68167306153657, 'Rocket': 65.06797902249839, 'Skateboard': 88.21709935710692, 'Table': 87.64628585949646}
test_miou_per_class = {'Airplane': 81.02623499173906, 'Bag': 77.02572768300885, 'Cap': 82.65239000228418, 'Car': 76.24831260908518, 'Chair': 85.4522957797802, 'Earphone': 61.02423002300308, 'Guitar': 88.5556413950363, 'Knife': 87.03094072544178, 'Lamp': 77.54182119976561, 'Laptop': 96.20270754164927, 'Motorbike': 64.4743728981254, 'Mug': 92.32124630328072, 'Pistol': 79.7037892707439, 'Rocket': 56.48617002995645, 'Skateboard': 78.20127323640938, 'Table': 82.18171339718458}
==================================================

EPOCH 67 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.385, iteration=0.232, train_acc=94.36, train_loss_seg=0.140, train_macc=88.31, train_miou=82.59]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.56it/s, test_acc=93.08, test_loss_seg=0.139, test_macc=86.09, test_miou=79.82]
==================================================
test_loss_seg = 0.1394781619310379
test_acc = 93.08790359364168
test_macc = 86.09888817537468
test_miou = 79.8199854602785
test_acc_per_class = {'Airplane': 91.10324069487584, 'Bag': 95.81599123767799, 'Cap': 92.46316758747697, 'Car': 90.96866654853281, 'Chair': 94.60656842241058, 'Earphone': 92.82069206571128, 'Guitar': 95.78976851267625, 'Knife': 92.65335235378032, 'Lamp': 91.49570860372619, 'Laptop': 97.85616397545135, 'Motorbike': 85.61590038314176, 'Mug': 98.9683602867047, 'Pistol': 95.8761585757548, 'Rocket': 82.16122718375894, 'Skateboard': 96.2480047046963, 'Table': 94.96348636189124}
test_macc_per_class = {'Airplane': 90.33491102586643, 'Bag': 80.20238874456393, 'Cap': 85.52694462651087, 'Car': 84.35882340881656, 'Chair': 90.95275363299878, 'Earphone': 70.4312984400606, 'Guitar': 93.20539559228312, 'Knife': 92.63139203369501, 'Lamp': 88.41433739424735, 'Laptop': 97.92882619906167, 'Motorbike': 71.85421832473551, 'Mug': 92.99652508610802, 'Pistol': 85.73697756154357, 'Rocket': 75.28802942625286, 'Skateboard': 87.82200671333598, 'Table': 89.89738259591476}
test_miou_per_class = {'Airplane': 81.59300643333793, 'Bag': 77.08283428866093, 'Cap': 80.65094504155952, 'Car': 76.14054063849063, 'Chair': 84.80466342363664, 'Earphone': 65.3733535695619, 'Guitar': 89.13832691607482, 'Knife': 86.28708159087711, 'Lamp': 80.50104717521688, 'Laptop': 95.78625107860461, 'Motorbike': 63.92270735293452, 'Mug': 90.82947642960428, 'Pistol': 81.31891271800392, 'Rocket': 62.564237833885684, 'Skateboard': 77.83987820864185, 'Table': 83.28650466536493}
==================================================

EPOCH 68 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.383, iteration=0.237, train_acc=94.92, train_loss_seg=0.135, train_macc=90.94, train_miou=85.38]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.54it/s, test_acc=93.00, test_loss_seg=0.165, test_macc=86.12, test_miou=79.63]
==================================================
test_loss_seg = 0.16555707156658173
test_acc = 93.00289570083956
test_macc = 86.12143956905781
test_miou = 79.63028800482006
test_acc_per_class = {'Airplane': 90.70879402926538, 'Bag': 94.65287710419187, 'Cap': 92.16085047376936, 'Car': 91.08599891042104, 'Chair': 94.46678234759457, 'Earphone': 92.57530759439966, 'Guitar': 95.79738491629575, 'Knife': 92.4491917003719, 'Lamp': 90.84666435805806, 'Laptop': 98.04927273161528, 'Motorbike': 86.69296918391971, 'Mug': 99.20550633847311, 'Pistol': 95.18822724161534, 'Rocket': 83.69749482011677, 'Skateboard': 96.2294220665499, 'Table': 94.23958739677529}
test_macc_per_class = {'Airplane': 88.21255882761105, 'Bag': 73.77942596486614, 'Cap': 84.53652568157298, 'Car': 85.46858701440281, 'Chair': 92.58217051740725, 'Earphone': 70.92895008621974, 'Guitar': 94.95486737638504, 'Knife': 92.40960441250904, 'Lamp': 88.4689579523705, 'Laptop': 98.07043290122597, 'Motorbike': 80.84401164530716, 'Mug': 95.04207954814717, 'Pistol': 82.52683578703292, 'Rocket': 77.27254867458258, 'Skateboard': 85.80130341896508, 'Table': 87.04417329631947}
test_miou_per_class = {'Airplane': 80.61633541490748, 'Bag': 70.82655330080776, 'Cap': 79.57762497359036, 'Car': 76.54832529152057, 'Chair': 84.31856004030502, 'Earphone': 65.77282672239716, 'Guitar': 89.5995331723568, 'Knife': 85.9318316947101, 'Lamp': 80.0387147051126, 'Laptop': 96.15397973675348, 'Motorbike': 70.09020110247714, 'Mug': 92.81176236592437, 'Pistol': 78.80925633529633, 'Rocket': 65.38456373839868, 'Skateboard': 77.33782287609327, 'Table': 80.26671660647001}
==================================================

EPOCH 69 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.384, iteration=0.236, train_acc=94.72, train_loss_seg=0.131, train_macc=88.11, train_miou=83.17]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.52it/s, test_acc=93.01, test_loss_seg=0.148, test_macc=87.42, test_miou=80.04]
==================================================
test_loss_seg = 0.14869752526283264
test_acc = 93.01873481038872
test_macc = 87.42139484363707
test_miou = 80.04624852304367
test_acc_per_class = {'Airplane': 90.1900835312162, 'Bag': 96.21922626025791, 'Cap': 92.81334200501719, 'Car': 90.93631208325543, 'Chair': 94.69096325159178, 'Earphone': 91.62638612453796, 'Guitar': 96.09173076982178, 'Knife': 92.45909782372482, 'Lamp': 90.981179801522, 'Laptop': 98.13508304383001, 'Motorbike': 85.95325411492095, 'Mug': 98.97474868677173, 'Pistol': 95.59640140933399, 'Rocket': 82.6232732799857, 'Skateboard': 96.1202732249699, 'Table': 94.8884015554622}
test_macc_per_class = {'Airplane': 90.83674764647832, 'Bag': 84.76861829998052, 'Cap': 86.40107254862207, 'Car': 86.87621315070272, 'Chair': 91.65141217818615, 'Earphone': 70.26747936265207, 'Guitar': 94.97347438656408, 'Knife': 92.41281056773153, 'Lamp': 91.90532305886005, 'Laptop': 98.13799488515141, 'Motorbike': 81.9995112986718, 'Mug': 93.02990977157184, 'Pistol': 85.27430159356729, 'Rocket': 73.09392442376023, 'Skateboard': 87.06022899099212, 'Table': 90.05329533470108}
test_miou_per_class = {'Airplane': 80.28633116462544, 'Bag': 80.38111187722394, 'Cap': 81.44403559391658, 'Car': 76.43710434040584, 'Chair': 84.80925598049178, 'Earphone': 64.11674946872597, 'Guitar': 89.8664018848273, 'Knife': 85.9465129244788, 'Lamp': 79.23185155183884, 'Laptop': 96.31829848559404, 'Motorbike': 67.80844480851087, 'Mug': 90.66690520677582, 'Pistol': 81.2352609494221, 'Rocket': 62.126446132252724, 'Skateboard': 76.8294321520969, 'Table': 83.23583384751176}
==================================================
macc: 87.32110078339136 -> 87.42139484363707

EPOCH 70 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.383, iteration=0.231, train_acc=94.81, train_loss_seg=0.141, train_macc=88.74, train_miou=83.45]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.52it/s, test_acc=92.52, test_loss_seg=0.161, test_macc=84.55, test_miou=78.35]
==================================================
test_loss_seg = 0.16122622787952423
test_acc = 92.52321900383492
test_macc = 84.55202965157446
test_miou = 78.35170710116245
test_acc_per_class = {'Airplane': 90.15798428729805, 'Bag': 95.51282051282051, 'Cap': 91.04367816091954, 'Car': 90.73683751036722, 'Chair': 94.84879365235541, 'Earphone': 92.74920234213386, 'Guitar': 96.10011208887843, 'Knife': 92.36280169715518, 'Lamp': 91.25637051695259, 'Laptop': 98.15791206094809, 'Motorbike': 85.60786679305623, 'Mug': 99.0007863553784, 'Pistol': 95.3419925226423, 'Rocket': 76.86069389126158, 'Skateboard': 95.83534524384355, 'Table': 94.79830642534779}
test_macc_per_class = {'Airplane': 89.95389671100173, 'Bag': 79.37441081429431, 'Cap': 82.2705333747462, 'Car': 84.60097483596407, 'Chair': 91.8157133530355, 'Earphone': 71.37648063603204, 'Guitar': 94.50577977396364, 'Knife': 92.30106682339465, 'Lamp': 89.02091132284359, 'Laptop': 98.10126990687311, 'Motorbike': 76.47057179079594, 'Mug': 95.41146895888275, 'Pistol': 82.95481299200097, 'Rocket': 55.22699242492218, 'Skateboard': 81.31001952170699, 'Table': 88.1375711847333}
test_miou_per_class = {'Airplane': 80.48542024725084, 'Bag': 76.38420060143493, 'Cap': 76.82364804725732, 'Car': 76.0124908897895, 'Chair': 85.45744853300663, 'Earphone': 65.95317352705197, 'Guitar': 89.87649480845803, 'Knife': 85.7649308511309, 'Lamp': 81.0604828929326, 'Laptop': 96.3591564289571, 'Motorbike': 66.39900782173002, 'Mug': 91.34936520390262, 'Pistol': 78.58903722341694, 'Rocket': 46.400852692186476, 'Skateboard': 74.39974866516309, 'Table': 82.31185518493011}
==================================================

EPOCH 71 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.384, iteration=0.236, train_acc=94.78, train_loss_seg=0.143, train_macc=89.55, train_miou=84.51]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.53it/s, test_acc=93.09, test_loss_seg=0.275, test_macc=86.75, test_miou=79.83]
==================================================
test_loss_seg = 0.27547940611839294
test_acc = 93.09112976492203
test_macc = 86.74998640091617
test_miou = 79.83659342215158
test_acc_per_class = {'Airplane': 90.77664556815283, 'Bag': 95.85308513508305, 'Cap': 91.14201395099266, 'Car': 91.02741450646738, 'Chair': 94.81404607834469, 'Earphone': 92.90582735610035, 'Guitar': 96.17139873255408, 'Knife': 92.80158580328488, 'Lamp': 90.9910280828882, 'Laptop': 98.11782836458339, 'Motorbike': 86.18711512493176, 'Mug': 99.22146626126556, 'Pistol': 95.9232751216719, 'Rocket': 82.40106714095154, 'Skateboard': 96.23742284603027, 'Table': 94.88685616545006}
test_macc_per_class = {'Airplane': 89.01113044478457, 'Bag': 81.64907438282177, 'Cap': 83.81392720107041, 'Car': 83.6180431548558, 'Chair': 91.78718358323717, 'Earphone': 70.23042300442324, 'Guitar': 94.94428517142268, 'Knife': 92.7527007512216, 'Lamp': 88.80122961163555, 'Laptop': 98.06846010869073, 'Motorbike': 79.95837217976587, 'Mug': 95.08901811279495, 'Pistol': 87.08327298459447, 'Rocket': 73.47890342475843, 'Skateboard': 87.12884521056958, 'Table': 90.58491308801186}
test_miou_per_class = {'Airplane': 80.96475711501161, 'Bag': 78.37316537370144, 'Cap': 78.16566227013988, 'Car': 75.93324504235689, 'Chair': 85.2040603740793, 'Earphone': 65.09130775413225, 'Guitar': 90.0297622531476, 'Knife': 86.54493693059754, 'Lamp': 77.75197010116783, 'Laptop': 96.28230103337532, 'Motorbike': 65.94673399200502, 'Mug': 92.99768755039608, 'Pistol': 82.20875178298037, 'Rocket': 60.85897081696415, 'Skateboard': 77.76398209326736, 'Table': 83.26820027110278}
==================================================

EPOCH 72 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.383, iteration=0.231, train_acc=94.74, train_loss_seg=0.142, train_macc=89.81, train_miou=84.35]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.56it/s, test_acc=92.91, test_loss_seg=0.187, test_macc=85.75, test_miou=79.00]
==================================================
test_loss_seg = 0.18705867230892181
test_acc = 92.91235892237097
test_macc = 85.755530923388
test_miou = 79.0008046563338
test_acc_per_class = {'Airplane': 90.5787676094079, 'Bag': 94.94454573078092, 'Cap': 93.79870281368112, 'Car': 90.95865035556218, 'Chair': 94.73850902389003, 'Earphone': 92.90252757994693, 'Guitar': 96.13411851170201, 'Knife': 93.11408789885611, 'Lamp': 90.76937723068743, 'Laptop': 97.88146940330039, 'Motorbike': 85.53634344362744, 'Mug': 99.07727954204717, 'Pistol': 94.87671451902102, 'Rocket': 80.54128440366972, 'Skateboard': 96.16310588076838, 'Table': 94.58225881098646}
test_macc_per_class = {'Airplane': 89.64650326386983, 'Bag': 79.51567836346705, 'Cap': 88.08938514958396, 'Car': 87.3025488646948, 'Chair': 90.24402116273859, 'Earphone': 72.95808499589594, 'Guitar': 94.52677468581861, 'Knife': 93.06161337843379, 'Lamp': 89.59720641901431, 'Laptop': 97.9516054766005, 'Motorbike': 65.73101418154765, 'Mug': 95.26656239145484, 'Pistol': 83.8868545147554, 'Rocket': 70.84071080841694, 'Skateboard': 84.40211743121799, 'Table': 89.06781368669758}
test_miou_per_class = {'Airplane': 80.61875884904286, 'Bag': 73.66348409007878, 'Cap': 83.80453673690332, 'Car': 76.59883111153198, 'Chair': 84.73436063880622, 'Earphone': 66.94196781175475, 'Guitar': 89.95142418115798, 'Knife': 87.08978144677937, 'Lamp': 80.09384634470702, 'Laptop': 95.8346906190208, 'Motorbike': 59.278922539107405, 'Mug': 91.93064529907993, 'Pistol': 78.23410903329714, 'Rocket': 56.579481777808795, 'Skateboard': 76.39095066663869, 'Table': 82.26708335562591}
==================================================

EPOCH 73 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.386, iteration=0.236, train_acc=94.87, train_loss_seg=0.128, train_macc=89.59, train_miou=84.35]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.55it/s, test_acc=92.57, test_loss_seg=0.141, test_macc=86.73, test_miou=78.93]
==================================================
test_loss_seg = 0.14109750092029572
test_acc = 92.57595104328712
test_macc = 86.73925479993594
test_miou = 78.93218531266963
test_acc_per_class = {'Airplane': 90.42576637657793, 'Bag': 95.43941538814707, 'Cap': 91.87063341507427, 'Car': 90.86268441015113, 'Chair': 94.8529236163694, 'Earphone': 90.31594624304103, 'Guitar': 96.18535341215413, 'Knife': 92.66473478644323, 'Lamp': 90.35584974077469, 'Laptop': 98.12130906991568, 'Motorbike': 86.47031224566007, 'Mug': 98.9951452360934, 'Pistol': 96.0906038332391, 'Rocket': 80.34540934114722, 'Skateboard': 95.87378640776699, 'Table': 92.3453431700387}
test_macc_per_class = {'Airplane': 90.0703525945487, 'Bag': 80.28730628677579, 'Cap': 84.1824902409883, 'Car': 84.35086386103214, 'Chair': 92.39515804223245, 'Earphone': 70.09270100469594, 'Guitar': 94.40004849333154, 'Knife': 92.59468259403359, 'Lamp': 90.42657748424396, 'Laptop': 98.11180948450556, 'Motorbike': 83.85761757842538, 'Mug': 93.82629290513384, 'Pistol': 87.26142219512049, 'Rocket': 79.4450656578635, 'Skateboard': 87.56142854848896, 'Table': 78.9642598275551}
test_miou_per_class = {'Airplane': 80.63665326814457, 'Bag': 76.86782804113778, 'Cap': 78.87830979441918, 'Car': 75.94108903917278, 'Chair': 85.3225461312451, 'Earphone': 62.105565326954206, 'Guitar': 90.06247438674282, 'Knife': 86.2831032887118, 'Lamp': 76.88493851393126, 'Laptop': 96.28976810743399, 'Motorbike': 69.64646254592805, 'Mug': 91.09072575657954, 'Pistol': 82.59784744076187, 'Rocket': 62.477762374073684, 'Skateboard': 76.52500251461188, 'Table': 71.30488847286557}
==================================================

EPOCH 74 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.385, iteration=0.237, train_acc=94.96, train_loss_seg=0.143, train_macc=90.04, train_miou=84.55]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.53it/s, test_acc=92.99, test_loss_seg=0.139, test_macc=85.44, test_miou=78.89]
==================================================
test_loss_seg = 0.1390828639268875
test_acc = 92.99941952844539
test_macc = 85.44686557740827
test_miou = 78.89476783604931
test_acc_per_class = {'Airplane': 90.51344043677626, 'Bag': 94.74537298805082, 'Cap': 91.96063452118815, 'Car': 90.37527854893125, 'Chair': 94.78816127824592, 'Earphone': 93.14967278868482, 'Guitar': 95.63022781837242, 'Knife': 92.96464646464646, 'Lamp': 90.59664876874652, 'Laptop': 98.10982010783987, 'Motorbike': 86.20273068957583, 'Mug': 98.51944154138864, 'Pistol': 95.73136612271466, 'Rocket': 83.79870852816744, 'Skateboard': 96.12895410811933, 'Table': 94.77560774367761}
test_macc_per_class = {'Airplane': 89.73269248697777, 'Bag': 75.84920255300942, 'Cap': 85.25556849162999, 'Car': 88.38695164263844, 'Chair': 89.8521037304255, 'Earphone': 68.45065408965107, 'Guitar': 92.19981691599303, 'Knife': 92.97091083755035, 'Lamp': 91.14124166788599, 'Laptop': 98.14091133281964, 'Motorbike': 73.4649609334596, 'Mug': 87.53790470443032, 'Pistol': 85.54920826531443, 'Rocket': 70.72882423094123, 'Skateboard': 89.87342341990286, 'Table': 88.01547393590278}
test_miou_per_class = {'Airplane': 80.23687210810182, 'Bag': 72.87358261931756, 'Cap': 80.0849636962337, 'Car': 75.65536930253901, 'Chair': 84.63056851731703, 'Earphone': 63.35416168336112, 'Guitar': 88.32705476172514, 'Knife': 86.84337485701583, 'Lamp': 79.05707145159344, 'Laptop': 96.27311698091073, 'Motorbike': 66.40445785667772, 'Mug': 86.20272356640095, 'Pistol': 80.98481261370823, 'Rocket': 61.7254883512663, 'Skateboard': 78.08764154035322, 'Table': 81.57502547026736}
==================================================

EPOCH 75 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.385, iteration=0.238, train_acc=93.89, train_loss_seg=0.165, train_macc=86.26, train_miou=81.12]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.53it/s, test_acc=93.06, test_loss_seg=0.132, test_macc=85.52, test_miou=79.26]
==================================================
test_loss_seg = 0.13204242289066315
test_acc = 93.06435266075621
test_macc = 85.52724144953164
test_miou = 79.26780384054021
test_acc_per_class = {'Airplane': 90.70025849538858, 'Bag': 95.43528362203554, 'Cap': 96.07306828172317, 'Car': 90.4052391273118, 'Chair': 94.72313978603633, 'Earphone': 93.66968277998802, 'Guitar': 95.86069700919914, 'Knife': 92.81649526628729, 'Lamp': 91.06468898593265, 'Laptop': 98.11738318649002, 'Motorbike': 85.46549479166666, 'Mug': 99.04304282687039, 'Pistol': 95.43078257693361, 'Rocket': 79.07236783254316, 'Skateboard': 96.35996992275618, 'Table': 94.79204808093718}
test_macc_per_class = {'Airplane': 89.5921580228554, 'Bag': 77.5955139777634, 'Cap': 92.18597332283245, 'Car': 83.71889770938348, 'Chair': 92.12176142403304, 'Earphone': 69.58736678885846, 'Guitar': 92.75322019597321, 'Knife': 92.7706037215169, 'Lamp': 89.56057623478361, 'Laptop': 98.1372004860254, 'Motorbike': 71.78719699197083, 'Mug': 92.14330549505573, 'Pistol': 81.95298368644673, 'Rocket': 66.72077568826978, 'Skateboard': 88.75451806164244, 'Table': 89.05381138509505}
test_miou_per_class = {'Airplane': 80.8514573536228, 'Bag': 74.91948398130158, 'Cap': 89.12903814724753, 'Car': 75.1602345150327, 'Chair': 85.05450246016748, 'Earphone': 65.47044528257814, 'Guitar': 88.9079324920521, 'Knife': 86.56393217694105, 'Lamp': 78.23253080389213, 'Laptop': 96.28750161240139, 'Motorbike': 63.4280796709342, 'Mug': 91.13279598845988, 'Pistol': 78.22153024356201, 'Rocket': 53.98345471321618, 'Skateboard': 78.15212012572212, 'Table': 82.78982188151245}
==================================================

EPOCH 76 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.383, iteration=0.233, train_acc=94.62, train_loss_seg=0.137, train_macc=90.46, train_miou=84.97]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.55it/s, test_acc=92.97, test_loss_seg=0.124, test_macc=86.38, test_miou=79.73]
==================================================
test_loss_seg = 0.12437088787555695
test_acc = 92.9709503513897
test_macc = 86.38500057599977
test_miou = 79.73104777069315
test_acc_per_class = {'Airplane': 90.88245871778706, 'Bag': 95.94835262689226, 'Cap': 92.23014395349894, 'Car': 91.16284139391486, 'Chair': 94.80883098000525, 'Earphone': 94.09891148367225, 'Guitar': 96.11418433599862, 'Knife': 91.93012173077652, 'Lamp': 90.53041127596545, 'Laptop': 98.25336637506345, 'Motorbike': 86.13759957107843, 'Mug': 99.10303159351628, 'Pistol': 95.86555044142673, 'Rocket': 80.25185778489742, 'Skateboard': 95.54068207108726, 'Table': 94.67686128665459}
test_macc_per_class = {'Airplane': 90.15289968359052, 'Bag': 82.72470215855692, 'Cap': 85.6722193219344, 'Car': 85.40172823653836, 'Chair': 92.62107793092545, 'Earphone': 72.52307246386326, 'Guitar': 94.24242509590633, 'Knife': 91.93325770327202, 'Lamp': 89.56787410095205, 'Laptop': 98.21968726018319, 'Motorbike': 79.84971616518789, 'Mug': 95.78027150900728, 'Pistol': 85.6498952491527, 'Rocket': 66.99756854584264, 'Skateboard': 82.19937062841453, 'Table': 88.62424316266906}
test_miou_per_class = {'Airplane': 81.09886677609654, 'Bag': 79.01372444350685, 'Cap': 80.6075321980393, 'Car': 76.7027238573233, 'Chair': 85.20312738040452, 'Earphone': 68.54324133758146, 'Guitar': 90.02137662703458, 'Knife': 85.06245176599082, 'Lamp': 79.63502002305044, 'Laptop': 96.54706763547057, 'Motorbike': 68.08617767464388, 'Mug': 92.07977574753727, 'Pistol': 81.46919617625858, 'Rocket': 55.587431921493234, 'Skateboard': 73.83195972813837, 'Table': 82.20709103852087}
==================================================

EPOCH 77 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.384, iteration=0.234, train_acc=94.66, train_loss_seg=0.136, train_macc=88.74, train_miou=83.37]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.55it/s, test_acc=93.30, test_loss_seg=0.152, test_macc=87.29, test_miou=80.39]
==================================================
test_loss_seg = 0.15225756168365479
test_acc = 93.3047479983827
test_macc = 87.29179021076165
test_miou = 80.39827298478326
test_acc_per_class = {'Airplane': 90.6390837708498, 'Bag': 95.61865088065483, 'Cap': 95.16880372664842, 'Car': 90.82472753482708, 'Chair': 94.95613235463233, 'Earphone': 93.13870151770658, 'Guitar': 96.01143815017502, 'Knife': 92.90609857248876, 'Lamp': 90.7975394794288, 'Laptop': 98.13628352307913, 'Motorbike': 86.7502298146162, 'Mug': 99.14073635801307, 'Pistol': 95.55847869602516, 'Rocket': 82.27133265151852, 'Skateboard': 96.05310962116458, 'Table': 94.90462132229491}
test_macc_per_class = {'Airplane': 89.47592812367571, 'Bag': 79.83823175427489, 'Cap': 90.23032792994272, 'Car': 88.42178268108194, 'Chair': 91.96255656816076, 'Earphone': 70.33495755442455, 'Guitar': 94.4824987729434, 'Knife': 92.90513932741608, 'Lamp': 90.29565117349262, 'Laptop': 98.09372839931699, 'Motorbike': 78.93974529306882, 'Mug': 96.20373008862167, 'Pistol': 85.46719045427923, 'Rocket': 68.95905668611947, 'Skateboard': 90.71515804153131, 'Table': 90.34296052383624}
test_miou_per_class = {'Airplane': 80.79171003472835, 'Bag': 76.72792281930516, 'Cap': 86.96941452133488, 'Car': 76.74198776844663, 'Chair': 85.44553516009394, 'Earphone': 65.5916855362211, 'Guitar': 89.80096345963466, 'Knife': 86.75130212360452, 'Lamp': 78.76009088812455, 'Laptop': 96.317319214838, 'Motorbike': 68.91771908854004, 'Mug': 92.4812414420112, 'Pistol': 80.49494041060453, 'Rocket': 59.23216227392081, 'Skateboard': 78.32041317915431, 'Table': 83.02795983596974}
==================================================
acc: 93.131845465478 -> 93.3047479983827, miou: 80.11268496674514 -> 80.39827298478326

EPOCH 78 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.381, iteration=0.233, train_acc=94.55, train_loss_seg=0.141, train_macc=89.88, train_miou=84.36]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:35<00:00,  2.54it/s, test_acc=93.38, test_loss_seg=0.118, test_macc=87.47, test_miou=80.75]
==================================================
test_loss_seg = 0.11885284632444382
test_acc = 93.38952466037554
test_macc = 87.47376673758077
test_miou = 80.75561479818244
test_acc_per_class = {'Airplane': 90.74407529988059, 'Bag': 95.76852791878171, 'Cap': 98.10781306856579, 'Car': 91.06668695671077, 'Chair': 94.85324224424664, 'Earphone': 91.5950197995077, 'Guitar': 96.12735334200774, 'Knife': 92.65084068923248, 'Lamp': 91.342512133846, 'Laptop': 98.10579866143105, 'Motorbike': 86.48309669018737, 'Mug': 99.21712604270712, 'Pistol': 95.80559797624183, 'Rocket': 81.76984770658736, 'Skateboard': 95.87945879458795, 'Table': 94.71539724148636}
test_macc_per_class = {'Airplane': 88.94277529979978, 'Bag': 84.6732831757643, 'Cap': 97.67430021819195, 'Car': 86.46799281003538, 'Chair': 91.10199909616783, 'Earphone': 70.10878542771323, 'Guitar': 93.9804109547127, 'Knife': 92.64233160859212, 'Lamp': 90.75097034427111, 'Laptop': 98.01977839818736, 'Motorbike': 78.99167033940043, 'Mug': 94.52938361179018, 'Pistol': 87.6252334349872, 'Rocket': 65.82067695062698, 'Skateboard': 89.86066675932824, 'Table': 88.39000937172365}
test_miou_per_class = {'Airplane': 80.742959212211, 'Bag': 78.88241320080229, 'Cap': 94.69538045513967, 'Car': 76.76547982160095, 'Chair': 85.24347058326546, 'Earphone': 63.40326926758867, 'Guitar': 89.8158695982111, 'Knife': 86.30268360989554, 'Lamp': 79.90065147995985, 'Laptop': 96.25021225410497, 'Motorbike': 68.08453671214623, 'Mug': 92.86548763755454, 'Pistol': 82.40543742059234, 'Rocket': 57.05849225562227, 'Skateboard': 77.48755019673716, 'Table': 82.18594306548685}
==================================================
acc: 93.3047479983827 -> 93.38952466037554, macc: 87.42139484363707 -> 87.47376673758077, miou: 80.39827298478326 -> 80.75561479818244

EPOCH 79 / 100
100%|█████████████████████████████| 438/438 [05:17<00:00,  1.38it/s, data_loading=0.378, iteration=0.234, train_acc=95.05, train_loss_seg=0.131, train_macc=87.81, train_miou=83.1 ]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.63it/s, test_acc=92.46, test_loss_seg=0.118, test_macc=84.93, test_miou=78.20]
==================================================
test_loss_seg = 0.11817304790019989
test_acc = 92.4612757975796
test_macc = 84.93344257771454
test_miou = 78.20497023863263
test_acc_per_class = {'Airplane': 90.67131559738375, 'Bag': 95.73568958893584, 'Cap': 92.12902625209458, 'Car': 91.15255501678479, 'Chair': 94.80507613214138, 'Earphone': 93.39426232400578, 'Guitar': 95.73619991244891, 'Knife': 86.5669700910273, 'Lamp': 90.98953359190753, 'Laptop': 98.00219394994456, 'Motorbike': 86.26671900581157, 'Mug': 99.2172131147541, 'Pistol': 95.48045697731467, 'Rocket': 78.91993116712219, 'Skateboard': 95.68496161780122, 'Table': 94.62830842179545}
test_macc_per_class = {'Airplane': 89.10105203953998, 'Bag': 81.6996529211836, 'Cap': 84.38287897406848, 'Car': 84.02906195470187, 'Chair': 91.53007269202818, 'Earphone': 70.13268877416508, 'Guitar': 92.00745573771512, 'Knife': 86.48101658163864, 'Lamp': 90.77949431140088, 'Laptop': 97.94297717470342, 'Motorbike': 77.49474467216973, 'Mug': 94.99849767316118, 'Pistol': 83.05336898843208, 'Rocket': 57.945445872613334, 'Skateboard': 88.4450346864966, 'Table': 88.9116381894143}
test_miou_per_class = {'Airplane': 80.84040451114713, 'Bag': 78.10568147296479, 'Cap': 79.36632797312096, 'Car': 76.19409582185848, 'Chair': 84.99735344696862, 'Earphone': 65.61006931994163, 'Guitar': 88.17254064586827, 'Knife': 75.99852146634825, 'Lamp': 79.15136117179595, 'Laptop': 96.05859621179471, 'Motorbike': 68.00849341360873, 'Mug': 92.97840517108638, 'Pistol': 78.70294364905867, 'Rocket': 48.61748066115888, 'Skateboard': 76.28391475171506, 'Table': 82.1933341296855}
==================================================

EPOCH 80 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.380, iteration=0.236, train_acc=95.15, train_loss_seg=0.136, train_macc=89.58, train_miou=84.36]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.64it/s, test_acc=92.78, test_loss_seg=0.120, test_macc=86.20, test_miou=79.26]
==================================================
test_loss_seg = 0.12038599699735641
test_acc = 92.78651144586685
test_macc = 86.2046934668664
test_miou = 79.26165595431695
test_acc_per_class = {'Airplane': 91.17509594338034, 'Bag': 95.31951438497872, 'Cap': 93.56176650798888, 'Car': 91.03134400473823, 'Chair': 94.87209680494342, 'Earphone': 92.29801069817852, 'Guitar': 96.26243365380765, 'Knife': 89.26108003712157, 'Lamp': 91.2138218566443, 'Laptop': 98.10222370522202, 'Motorbike': 86.1739813112745, 'Mug': 99.2248062015504, 'Pistol': 95.43770423896534, 'Rocket': 80.46420763756453, 'Skateboard': 95.34494866735778, 'Table': 94.84114748015323}
test_macc_per_class = {'Airplane': 89.97417146410025, 'Bag': 79.9076225018702, 'Cap': 87.0548956273095, 'Car': 85.25914612702117, 'Chair': 91.13731301432779, 'Earphone': 71.04223639697476, 'Guitar': 94.78757857015728, 'Knife': 89.22859384496431, 'Lamp': 90.33195793827113, 'Laptop': 98.0436369985853, 'Motorbike': 78.98634339118004, 'Mug': 95.55364436747504, 'Pistol': 84.24965312654513, 'Rocket': 67.54200547834822, 'Skateboard': 87.43227360281138, 'Table': 88.74402301992104}
test_miou_per_class = {'Airplane': 81.66815266239877, 'Bag': 75.49556674886895, 'Cap': 82.80588604918972, 'Car': 76.25200676439664, 'Chair': 85.092525475393, 'Earphone': 65.15219874951073, 'Guitar': 90.41831290168865, 'Knife': 80.46761174951301, 'Lamp': 79.61999800079043, 'Laptop': 96.25105257273677, 'Motorbike': 68.49934621131153, 'Mug': 93.0212123732129, 'Pistol': 79.50043513283894, 'Rocket': 56.93180660743762, 'Skateboard': 74.39952046605902, 'Table': 82.61086280372457}
==================================================

EPOCH 81 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.381, iteration=0.234, train_acc=94.96, train_loss_seg=0.137, train_macc=90.45, train_miou=85.33]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.64it/s, test_acc=93.23, test_loss_seg=0.093, test_macc=86.51, test_miou=80.08]
==================================================
test_loss_seg = 0.09392178803682327
test_acc = 93.2302702808586
test_macc = 86.5188119900699
test_miou = 80.08840357158344
test_acc_per_class = {'Airplane': 90.99603270710077, 'Bag': 94.91620740074232, 'Cap': 94.76846424384526, 'Car': 90.94047622790772, 'Chair': 94.7530939801501, 'Earphone': 93.15484501535883, 'Guitar': 96.19828446777832, 'Knife': 92.04170243159761, 'Lamp': 90.25468207184989, 'Laptop': 97.94804497599094, 'Motorbike': 86.15674785539215, 'Mug': 99.16990226268577, 'Pistol': 95.8496947136664, 'Rocket': 83.64029837001566, 'Skateboard': 95.77778511087645, 'Table': 95.11806265877962}
test_macc_per_class = {'Airplane': 89.65528501725069, 'Bag': 75.95910427799485, 'Cap': 89.03979621621048, 'Car': 83.46170302854294, 'Chair': 91.81877429387895, 'Earphone': 74.49881297522597, 'Guitar': 95.04013657385264, 'Knife': 92.04292989093942, 'Lamp': 88.75682476778296, 'Laptop': 98.00515647964443, 'Motorbike': 75.32036297825448, 'Mug': 94.24326789323328, 'Pistol': 89.86212953382858, 'Rocket': 76.2752407127929, 'Skateboard': 80.71066749783786, 'Table': 89.61079970384787}
test_miou_per_class = {'Airplane': 80.88795026114784, 'Bag': 73.19191379029716, 'Cap': 85.5353030829303, 'Car': 75.82666911823102, 'Chair': 84.87322747927176, 'Earphone': 69.45270872127128, 'Guitar': 90.28606934190564, 'Knife': 85.25242273179249, 'Lamp': 76.36575542848266, 'Laptop': 95.96005716314376, 'Motorbike': 66.74958665722338, 'Mug': 92.40555080227712, 'Pistol': 82.77433271109726, 'Rocket': 64.84264325198376, 'Skateboard': 73.47122930923452, 'Table': 83.53903729504503}
==================================================
loss_seg: 0.10609838366508484 -> 0.09392178803682327

EPOCH 82 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.382, iteration=0.232, train_acc=95.20, train_loss_seg=0.128, train_macc=89.20, train_miou=83.99]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.63it/s, test_acc=92.85, test_loss_seg=0.149, test_macc=86.74, test_miou=79.54]
==================================================
test_loss_seg = 0.14912843704223633
test_acc = 92.85367904557583
test_macc = 86.74303179288145
test_miou = 79.54684933932674
test_acc_per_class = {'Airplane': 91.03220562836533, 'Bag': 95.22657635865183, 'Cap': 94.92790673893036, 'Car': 90.94464295666928, 'Chair': 94.63829637945551, 'Earphone': 93.13718589427747, 'Guitar': 96.20427305244088, 'Knife': 92.9368520263902, 'Lamp': 89.97513079327179, 'Laptop': 98.20399985689242, 'Motorbike': 85.74221759901371, 'Mug': 99.01125729011258, 'Pistol': 95.41382154805005, 'Rocket': 78.31019754404699, 'Skateboard': 96.21944286526436, 'Table': 93.73485819738072}
test_macc_per_class = {'Airplane': 88.01915928710109, 'Bag': 76.68098685249281, 'Cap': 89.92271622421636, 'Car': 85.92866385695895, 'Chair': 91.45173599095634, 'Earphone': 71.43822690759343, 'Guitar': 94.46186188013188, 'Knife': 92.90867824236616, 'Lamp': 91.18106176435131, 'Laptop': 98.19440336425664, 'Motorbike': 83.60971793387981, 'Mug': 95.16086512796423, 'Pistol': 83.89476550906923, 'Rocket': 68.01821281431867, 'Skateboard': 86.57890457979495, 'Table': 90.43854835065139}
test_miou_per_class = {'Airplane': 80.90386640559728, 'Bag': 74.13184645622901, 'Cap': 86.41665410319112, 'Car': 76.43615038240965, 'Chair': 84.79600685754114, 'Earphone': 67.1997963312802, 'Guitar': 90.24324230413062, 'Knife': 86.78349430302731, 'Lamp': 78.05820262082611, 'Laptop': 96.45024815397771, 'Motorbike': 69.19914299877775, 'Mug': 91.42708952485397, 'Pistol': 80.24717252047267, 'Rocket': 53.01248594386389, 'Skateboard': 77.68817049025442, 'Table': 79.75602003279515}
==================================================

EPOCH 83 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.377, iteration=0.234, train_acc=95.55, train_loss_seg=0.127, train_macc=90.93, train_miou=86.11]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.64it/s, test_acc=93.05, test_loss_seg=0.123, test_macc=86.65, test_miou=79.98]
==================================================
test_loss_seg = 0.12374790012836456
test_acc = 93.05331358686607
test_macc = 86.65310503834752
test_miou = 79.98882076384447
test_acc_per_class = {'Airplane': 90.32542017746519, 'Bag': 96.15253272623791, 'Cap': 92.65814487053426, 'Car': 91.11328915025358, 'Chair': 94.89553311582611, 'Earphone': 91.13500176242509, 'Guitar': 96.14740994886274, 'Knife': 93.01321357630192, 'Lamp': 90.71550393600104, 'Laptop': 98.08040069479145, 'Motorbike': 86.93899437917133, 'Mug': 99.20376747288046, 'Pistol': 95.65112345977289, 'Rocket': 82.79872444194335, 'Skateboard': 95.57606301314999, 'Table': 94.44789466423994}
test_macc_per_class = {'Airplane': 88.99095584796648, 'Bag': 82.20366729840869, 'Cap': 85.03816469745557, 'Car': 85.30548822311842, 'Chair': 91.01766578839442, 'Earphone': 72.69943398498793, 'Guitar': 94.73364107466665, 'Knife': 93.00934762291898, 'Lamp': 90.46960159541626, 'Laptop': 97.99937359180113, 'Motorbike': 78.24939304743891, 'Mug': 94.45057076804322, 'Pistol': 86.46757024199727, 'Rocket': 72.96027336546122, 'Skateboard': 87.28942153642035, 'Table': 85.56511192906467}
test_miou_per_class = {'Airplane': 80.07256724818052, 'Bag': 79.26270820297259, 'Cap': 80.48402595177618, 'Car': 76.43603036184643, 'Chair': 85.05919042565574, 'Earphone': 65.10377938481008, 'Guitar': 89.91906102934179, 'Knife': 86.9361357062451, 'Lamp': 78.9493889241436, 'Laptop': 96.20710540065383, 'Motorbike': 69.16428168441483, 'Mug': 92.77835565032666, 'Pistol': 81.34384554606643, 'Rocket': 62.22911214941872, 'Skateboard': 75.3363359436623, 'Table': 80.5392086119967}
==================================================

EPOCH 84 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.379, iteration=0.230, train_acc=95.23, train_loss_seg=0.124, train_macc=90.72, train_miou=85.26]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.64it/s, test_acc=93.18, test_loss_seg=0.118, test_macc=87.23, test_miou=80.36]
==================================================
test_loss_seg = 0.11829501390457153
test_acc = 93.18433750848855
test_macc = 87.235969583182
test_miou = 80.36118939234017
test_acc_per_class = {'Airplane': 90.94243305368033, 'Bag': 95.25529593257328, 'Cap': 93.35950278050376, 'Car': 91.11793013363285, 'Chair': 94.87956643492174, 'Earphone': 92.20425413013866, 'Guitar': 96.18212942050012, 'Knife': 92.15254813487034, 'Lamp': 90.09596649172961, 'Laptop': 98.10243161457772, 'Motorbike': 86.77929739876643, 'Mug': 99.03467905428084, 'Pistol': 95.49619719600476, 'Rocket': 84.23439445136049, 'Skateboard': 96.22729631551634, 'Table': 94.88547759275974}
test_macc_per_class = {'Airplane': 89.4191221977074, 'Bag': 80.79324698698814, 'Cap': 86.7337434045937, 'Car': 85.25519565657478, 'Chair': 90.53315453119866, 'Earphone': 72.91881730100708, 'Guitar': 93.85177421908341, 'Knife': 92.11117605203911, 'Lamp': 89.71125701022865, 'Laptop': 98.13128301918978, 'Motorbike': 79.76891829944078, 'Mug': 92.7652491519541, 'Pistol': 84.30340925605346, 'Rocket': 79.40371464627201, 'Skateboard': 90.83493098952337, 'Table': 89.24052060905728}
test_miou_per_class = {'Airplane': 81.32181479191446, 'Bag': 76.63636104672923, 'Cap': 82.46260631949124, 'Car': 76.53690882515936, 'Chair': 85.12279870256432, 'Earphone': 66.14504559398746, 'Guitar': 89.80160072593077, 'Knife': 85.40756376154748, 'Lamp': 77.7880727602585, 'Laptop': 96.2592035185952, 'Motorbike': 69.35262085322151, 'Mug': 91.12862090462717, 'Pistol': 80.31680308450917, 'Rocket': 65.78136296654814, 'Skateboard': 78.87764008214103, 'Table': 82.84000634021746}
==================================================

EPOCH 85 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.377, iteration=0.230, train_acc=94.63, train_loss_seg=0.140, train_macc=90.44, train_miou=84.81]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.63it/s, test_acc=92.58, test_loss_seg=0.115, test_macc=86.70, test_miou=79.50]
==================================================
test_loss_seg = 0.1150493398308754
test_acc = 92.58341196384566
test_macc = 86.70767506819344
test_miou = 79.5087678667701
test_acc_per_class = {'Airplane': 91.12133774441686, 'Bag': 95.60684470970423, 'Cap': 95.13658841038935, 'Car': 91.21919444142873, 'Chair': 94.81828645102937, 'Earphone': 82.8890527868507, 'Guitar': 96.21389623544843, 'Knife': 91.83546586090698, 'Lamp': 89.67490756826585, 'Laptop': 98.15656082631982, 'Motorbike': 86.33925125434196, 'Mug': 99.2040932347925, 'Pistol': 95.75008329790778, 'Rocket': 83.51760579015142, 'Skateboard': 95.36183245902707, 'Table': 94.48959035054936}
test_macc_per_class = {'Airplane': 89.31776832759589, 'Bag': 85.01417097660435, 'Cap': 89.9265479957261, 'Car': 84.96326929497681, 'Chair': 90.09889541061102, 'Earphone': 68.71844442325111, 'Guitar': 94.7784965450369, 'Knife': 91.83517779955821, 'Lamp': 88.82348640432524, 'Laptop': 98.15861007606645, 'Motorbike': 75.03223067832802, 'Mug': 94.81612834964312, 'Pistol': 85.65934856068053, 'Rocket': 72.38651117207687, 'Skateboard': 90.45425612220762, 'Table': 87.3394589544071}
test_miou_per_class = {'Airplane': 81.4456440188381, 'Bag': 79.39806288126653, 'Cap': 86.52215206000265, 'Car': 76.72053866278033, 'Chair': 84.91424363748587, 'Earphone': 54.38697354329024, 'Guitar': 90.33304259304596, 'Knife': 84.90150392271529, 'Lamp': 76.5917658454053, 'Laptop': 96.35989230851025, 'Motorbike': 66.98234439133799, 'Mug': 92.79409886274688, 'Pistol': 81.13729786452099, 'Rocket': 62.617425446049616, 'Skateboard': 76.18723575173445, 'Table': 80.84806407859105}
==================================================

EPOCH 86 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.378, iteration=0.239, train_acc=95.14, train_loss_seg=0.129, train_macc=90.11, train_miou=85.20]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:33<00:00,  2.65it/s, test_acc=93.00, test_loss_seg=0.111, test_macc=87.59, test_miou=80.18]
==================================================
test_loss_seg = 0.11165614426136017
test_acc = 93.00872606869858
test_macc = 87.59876800041448
test_miou = 80.1856342105089
test_acc_per_class = {'Airplane': 91.17744672500173, 'Bag': 96.07695776341501, 'Cap': 92.74903367019047, 'Car': 91.28363237410295, 'Chair': 94.80433094043097, 'Earphone': 90.4657184407477, 'Guitar': 95.82164362675655, 'Knife': 90.84700725757887, 'Lamp': 91.14650205466958, 'Laptop': 98.07448045558903, 'Motorbike': 86.17628592127505, 'Mug': 99.04791655348508, 'Pistol': 95.7701849345963, 'Rocket': 83.98329928044772, 'Skateboard': 95.8340197015122, 'Table': 94.88115739937808}
test_macc_per_class = {'Airplane': 89.7049342962199, 'Bag': 84.56883950499812, 'Cap': 85.65614317507017, 'Car': 86.05523206449341, 'Chair': 91.73124137045936, 'Earphone': 70.77288691157034, 'Guitar': 94.33666152764, 'Knife': 90.80823413640755, 'Lamp': 91.03189522719165, 'Laptop': 98.01066573910083, 'Motorbike': 81.78258931873013, 'Mug': 95.55748213499507, 'Pistol': 85.7799461324334, 'Rocket': 80.13453664141218, 'Skateboard': 86.74462785691604, 'Table': 88.90437196899353}
test_miou_per_class = {'Airplane': 81.60617968305299, 'Bag': 80.36780568788954, 'Cap': 80.98821086695254, 'Car': 77.11060561671765, 'Chair': 85.10936560988782, 'Earphone': 62.777024382504635, 'Guitar': 88.61576759992292, 'Knife': 83.15197760933935, 'Lamp': 79.24029121546793, 'Laptop': 96.19778215630494, 'Motorbike': 70.06915211788667, 'Mug': 91.70399616342795, 'Pistol': 81.29015983785308, 'Rocket': 66.15467611592058, 'Skateboard': 75.80995479369254, 'Table': 82.77719791132128}
==================================================
macc: 87.47376673758077 -> 87.59876800041448

EPOCH 87 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.379, iteration=0.236, train_acc=94.83, train_loss_seg=0.127, train_macc=89.01, train_miou=83.24]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:33<00:00,  2.65it/s, test_acc=93.16, test_loss_seg=2.787, test_macc=86.67, test_miou=80.09]
==================================================
test_loss_seg = 2.7872393131256104
test_acc = 93.1652741091659
test_macc = 86.6723169550004
test_miou = 80.09818244041884
test_acc_per_class = {'Airplane': 91.34345503683512, 'Bag': 96.01676357320848, 'Cap': 93.04629542680064, 'Car': 91.17030978147535, 'Chair': 94.78147202592343, 'Earphone': 93.70619418650158, 'Guitar': 96.2923548573709, 'Knife': 92.24896802108718, 'Lamp': 90.93381831654773, 'Laptop': 97.73743500349235, 'Motorbike': 86.57207129261117, 'Mug': 99.17181722674744, 'Pistol': 95.38110390267487, 'Rocket': 81.2573284026337, 'Skateboard': 96.38962211725479, 'Table': 94.59537657549014}
test_macc_per_class = {'Airplane': 89.6307285609302, 'Bag': 83.01295409263223, 'Cap': 86.2362102407157, 'Car': 85.76755556747358, 'Chair': 91.20202811736759, 'Earphone': 71.41222126632614, 'Guitar': 94.56623210465308, 'Knife': 92.20485538596823, 'Lamp': 89.70407337645747, 'Laptop': 97.83457507853257, 'Motorbike': 83.71648363570304, 'Mug': 95.93090301390477, 'Pistol': 82.04277333671773, 'Rocket': 67.48395845328513, 'Skateboard': 88.16568674777191, 'Table': 87.84583230156694}
test_miou_per_class = {'Airplane': 81.82459138815776, 'Bag': 79.12114648751862, 'Cap': 81.61921924572528, 'Car': 76.8944499069875, 'Chair': 84.6842322182774, 'Earphone': 67.06466840771067, 'Guitar': 90.39938043831134, 'Knife': 85.57335617714106, 'Lamp': 80.54172277510384, 'Laptop': 95.5604850234959, 'Motorbike': 71.31894549919066, 'Mug': 92.76407172402487, 'Pistol': 78.48940300380417, 'Rocket': 56.19662544509821, 'Skateboard': 78.22493914085896, 'Table': 81.29368216529528}
==================================================

EPOCH 88 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.382, iteration=0.235, train_acc=94.82, train_loss_seg=0.132, train_macc=89.29, train_miou=84.13]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:33<00:00,  2.65it/s, test_acc=92.97, test_loss_seg=0.089, test_macc=87.10, test_miou=80.07]
==================================================
test_loss_seg = 0.08968085050582886
test_acc = 92.97788522309538
test_macc = 87.1099039389467
test_miou = 80.07005470681347
test_acc_per_class = {'Airplane': 90.93112745133509, 'Bag': 95.34629587760946, 'Cap': 92.67034657497058, 'Car': 91.22209151059076, 'Chair': 94.91245603515084, 'Earphone': 93.11621284926144, 'Guitar': 96.01013260332344, 'Knife': 92.05964411928824, 'Lamp': 89.67969102097028, 'Laptop': 98.10513047666046, 'Motorbike': 86.46257265702056, 'Mug': 99.05075424053729, 'Pistol': 96.02157556601504, 'Rocket': 80.94258629940977, 'Skateboard': 96.34697957979613, 'Table': 94.76856670758632}
test_macc_per_class = {'Airplane': 88.71213304781693, 'Bag': 78.98195728266914, 'Cap': 87.65504089737706, 'Car': 84.65568313726922, 'Chair': 92.40941076112681, 'Earphone': 74.53870410734899, 'Guitar': 93.88071608963243, 'Knife': 92.05471251993607, 'Lamp': 88.22858341970961, 'Laptop': 98.02070287495827, 'Motorbike': 74.4068323739551, 'Mug': 93.42229120923156, 'Pistol': 88.0090587201941, 'Rocket': 83.94957569527686, 'Skateboard': 88.18774320048661, 'Table': 86.6453176861583}
test_miou_per_class = {'Airplane': 80.81503776585586, 'Bag': 75.93791242692629, 'Cap': 81.86526179502349, 'Car': 76.43333120116607, 'Chair': 85.16314090595601, 'Earphone': 68.46597604833391, 'Guitar': 89.65877577832782, 'Knife': 85.28495059405257, 'Lamp': 75.24510089850685, 'Laptop': 96.2542908261269, 'Motorbike': 67.04448376277865, 'Mug': 91.2711846137616, 'Pistol': 83.54048800349281, 'Rocket': 64.37640198914215, 'Skateboard': 78.36782116867002, 'Table': 81.39671753089445}
==================================================
loss_seg: 0.09392178803682327 -> 0.08968085050582886

EPOCH 89 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.383, iteration=0.238, train_acc=95.15, train_loss_seg=0.125, train_macc=90.75, train_miou=85.24]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:33<00:00,  2.65it/s, test_acc=92.60, test_loss_seg=0.114, test_macc=85.70, test_miou=79.19]
==================================================
test_loss_seg = 0.11469707638025284
test_acc = 92.60705652755442
test_macc = 85.70805008690316
test_miou = 79.19103749140974
test_acc_per_class = {'Airplane': 91.2799272565902, 'Bag': 96.06877745796092, 'Cap': 94.31395622499412, 'Car': 91.07563375505899, 'Chair': 94.92916306390416, 'Earphone': 88.60696340116175, 'Guitar': 96.1182366955942, 'Knife': 91.50463581094088, 'Lamp': 90.71447275370116, 'Laptop': 98.16451067604991, 'Motorbike': 86.6794496908106, 'Mug': 99.08128539337933, 'Pistol': 95.95210881890299, 'Rocket': 76.43491444661036, 'Skateboard': 95.93342344836572, 'Table': 94.85544554684576}
test_macc_per_class = {'Airplane': 89.57905844157743, 'Bag': 81.18271489787719, 'Cap': 89.23221787844449, 'Car': 84.62044056703934, 'Chair': 90.6264637873171, 'Earphone': 71.96969667925966, 'Guitar': 94.563986511869, 'Knife': 91.48175578253108, 'Lamp': 85.77278381796222, 'Laptop': 98.11770667976761, 'Motorbike': 72.1426036167948, 'Mug': 92.30443903626295, 'Pistol': 85.81080689769918, 'Rocket': 66.7549099143807, 'Skateboard': 88.15123216720994, 'Table': 89.01798471445787}
test_miou_per_class = {'Airplane': 81.60660418087954, 'Bag': 78.68716197473354, 'Cap': 85.02970411570178, 'Car': 76.06149250311822, 'Chair': 85.29067425977284, 'Earphone': 61.26220976929737, 'Guitar': 90.00133220880743, 'Knife': 84.30503972839553, 'Lamp': 77.59128816681124, 'Laptop': 96.37218307989922, 'Motorbike': 65.49960012039918, 'Mug': 91.43100755937431, 'Pistol': 81.56295082568434, 'Rocket': 52.83565088993818, 'Skateboard': 76.7970208659727, 'Table': 82.72267961377041}
==================================================

EPOCH 90 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.380, iteration=0.231, train_acc=94.85, train_loss_seg=0.119, train_macc=90.47, train_miou=84.63]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.64it/s, test_acc=93.21, test_loss_seg=0.106, test_macc=87.90, test_miou=80.55]
==================================================
test_loss_seg = 0.10670359432697296
test_acc = 93.21122525991969
test_macc = 87.90512718023592
test_miou = 80.55905524886462
test_acc_per_class = {'Airplane': 90.92713079772676, 'Bag': 96.16311469237141, 'Cap': 94.55587392550143, 'Car': 91.22016648089443, 'Chair': 94.68442027995975, 'Earphone': 93.0275687435358, 'Guitar': 96.31574071214601, 'Knife': 91.68879299087794, 'Lamp': 89.7405844522981, 'Laptop': 98.03113062725359, 'Motorbike': 87.30181525735294, 'Mug': 98.82296484890495, 'Pistol': 95.91094645123667, 'Rocket': 81.46236755571795, 'Skateboard': 96.42255134604369, 'Table': 95.10443499689349}
test_macc_per_class = {'Airplane': 89.55106641281898, 'Bag': 83.14725241079867, 'Cap': 89.47782267241138, 'Car': 87.70710296658754, 'Chair': 92.04627547826291, 'Earphone': 72.79498270645858, 'Guitar': 94.68263606763216, 'Knife': 91.66042584377135, 'Lamp': 89.33359237406437, 'Laptop': 97.9271682147707, 'Motorbike': 77.68227956271053, 'Mug': 91.42691485653043, 'Pistol': 89.32185538527267, 'Rocket': 80.32158629143574, 'Skateboard': 89.26007707285962, 'Table': 90.14099656738918}
test_miou_per_class = {'Airplane': 81.04211113700924, 'Bag': 79.70865061489636, 'Cap': 85.55018230256628, 'Car': 76.92854983853802, 'Chair': 84.75873456379614, 'Earphone': 67.6884651576405, 'Guitar': 90.52575695909142, 'Knife': 84.62786664736923, 'Lamp': 75.83369635761974, 'Laptop': 96.10978314491341, 'Motorbike': 69.47637019172448, 'Mug': 89.25673319759049, 'Pistol': 83.14836535201376, 'Rocket': 61.701946480635684, 'Skateboard': 78.95875093544012, 'Table': 83.62892110098925}
==================================================
macc: 87.59876800041448 -> 87.90512718023592

EPOCH 91 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.380, iteration=0.233, train_acc=94.82, train_loss_seg=0.127, train_macc=90.38, train_miou=85.39]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.64it/s, test_acc=93.08, test_loss_seg=0.143, test_macc=87.01, test_miou=80.04]
==================================================
test_loss_seg = 0.14328664541244507
test_acc = 93.08578496784574
test_macc = 87.01605375246184
test_miou = 80.04563215539382
test_acc_per_class = {'Airplane': 91.29300373107286, 'Bag': 95.48421095301795, 'Cap': 92.57903031426717, 'Car': 91.090709894073, 'Chair': 94.71026406855506, 'Earphone': 91.7924961985926, 'Guitar': 96.2468803696974, 'Knife': 92.68442775802481, 'Lamp': 90.0179843202573, 'Laptop': 97.87081725061596, 'Motorbike': 86.92842371323529, 'Mug': 99.01707610562039, 'Pistol': 95.97379242408623, 'Rocket': 83.96138419923514, 'Skateboard': 95.06798763256758, 'Table': 94.65407055261306}
test_macc_per_class = {'Airplane': 89.42093906618132, 'Bag': 80.48119048490318, 'Cap': 85.18386630706814, 'Car': 84.75430559403168, 'Chair': 90.37242933239979, 'Earphone': 69.47929630524298, 'Guitar': 94.54941387240972, 'Knife': 92.68696055112625, 'Lamp': 87.32075723014081, 'Laptop': 97.75568372546326, 'Motorbike': 84.16283264329289, 'Mug': 94.92940540055757, 'Pistol': 87.26501197008702, 'Rocket': 76.78195137835611, 'Skateboard': 89.08550935995429, 'Table': 88.02730681817467}
test_miou_per_class = {'Airplane': 81.74461642006929, 'Bag': 77.39952076908598, 'Cap': 80.55158705597569, 'Car': 75.95256497249883, 'Chair': 84.69022901271785, 'Earphone': 63.21350905842452, 'Guitar': 90.22122610057664, 'Knife': 86.36399427688588, 'Lamp': 78.36262659445775, 'Laptop': 95.79997277231566, 'Motorbike': 70.86012264776967, 'Mug': 91.41505393798681, 'Pistol': 82.50159081587206, 'Rocket': 65.05584827506722, 'Skateboard': 74.60708576820782, 'Table': 81.99056600838945}
==================================================

EPOCH 92 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.378, iteration=0.235, train_acc=94.87, train_loss_seg=0.127, train_macc=89.36, train_miou=83.36]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.64it/s, test_acc=93.27, test_loss_seg=0.099, test_macc=87.42, test_miou=80.53]
==================================================
test_loss_seg = 0.09936504065990448
test_acc = 93.27866306128325
test_macc = 87.42907139566856
test_miou = 80.53359235730746
test_acc_per_class = {'Airplane': 90.71603030460878, 'Bag': 95.5511486941555, 'Cap': 93.50286499029218, 'Car': 91.1799079945294, 'Chair': 94.91112648912083, 'Earphone': 92.91794585542841, 'Guitar': 96.14696416747509, 'Knife': 92.33005450786291, 'Lamp': 89.87701125034104, 'Laptop': 98.14570169827901, 'Motorbike': 86.72587590995848, 'Mug': 99.28965746591287, 'Pistol': 95.85655911270847, 'Rocket': 84.06296114117069, 'Skateboard': 96.1730421934001, 'Table': 95.07175720528801}
test_macc_per_class = {'Airplane': 90.54355775237532, 'Bag': 81.89054624532137, 'Cap': 87.12241597302746, 'Car': 85.08915114952632, 'Chair': 91.00489966976512, 'Earphone': 71.77936097721012, 'Guitar': 94.23820157897276, 'Knife': 92.32529287511251, 'Lamp': 88.39247729842309, 'Laptop': 98.11082816757803, 'Motorbike': 77.98041589904238, 'Mug': 95.26045878274016, 'Pistol': 85.3292717963067, 'Rocket': 78.61164455961568, 'Skateboard': 91.42550524952355, 'Table': 89.76111435615616}
test_miou_per_class = {'Airplane': 80.76822305319175, 'Bag': 77.9969258613679, 'Cap': 82.92681717584024, 'Car': 76.44574144451579, 'Chair': 85.19192384247685, 'Earphone': 66.81005493062231, 'Guitar': 90.15921201513262, 'Knife': 85.73623795144478, 'Lamp': 75.8240841464974, 'Laptop': 96.33878763378267, 'Motorbike': 68.3160187472466, 'Mug': 93.45965864744312, 'Pistol': 81.24648323641985, 'Rocket': 64.8429431012464, 'Skateboard': 78.981751845811, 'Table': 83.49261408388004}
==================================================

EPOCH 93 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.382, iteration=0.237, train_acc=94.46, train_loss_seg=0.129, train_macc=89.36, train_miou=83.89]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.63it/s, test_acc=92.81, test_loss_seg=0.146, test_macc=86.83, test_miou=79.70]
==================================================
test_loss_seg = 0.14603674411773682
test_acc = 92.81652878626542
test_macc = 86.83488444415204
test_miou = 79.70096955284632
test_acc_per_class = {'Airplane': 91.12508298625588, 'Bag': 95.83570156492743, 'Cap': 92.70348295772025, 'Car': 90.89882726598823, 'Chair': 94.95276428107951, 'Earphone': 87.84518680697789, 'Guitar': 96.25773933605053, 'Knife': 92.0160498888185, 'Lamp': 90.33859932205087, 'Laptop': 98.19201772251402, 'Motorbike': 86.5661339647784, 'Mug': 99.26117969453139, 'Pistol': 95.61619636465065, 'Rocket': 83.73073436083409, 'Skateboard': 95.19960936842283, 'Table': 94.52515469464622}
test_macc_per_class = {'Airplane': 90.35065402452231, 'Bag': 80.36752482717304, 'Cap': 85.39443953237055, 'Car': 82.4741693549508, 'Chair': 91.40772403814672, 'Earphone': 72.1652401683305, 'Guitar': 94.80406371023818, 'Knife': 91.97297448244399, 'Lamp': 87.16948595497323, 'Laptop': 98.1417406079363, 'Motorbike': 76.6361310158059, 'Mug': 95.20541582824119, 'Pistol': 84.56035437820468, 'Rocket': 83.62345807036942, 'Skateboard': 88.59923584200233, 'Table': 86.4855392707233}
test_miou_per_class = {'Airplane': 81.85993609946343, 'Bag': 77.59404926505628, 'Cap': 80.82144361782748, 'Car': 75.35211670643011, 'Chair': 85.56766275577736, 'Earphone': 60.73439617652405, 'Guitar': 90.32061437809688, 'Knife': 85.18183242362879, 'Lamp': 78.24044431100249, 'Laptop': 96.42451851011934, 'Motorbike': 68.0012685243909, 'Mug': 93.25393919549525, 'Pistol': 80.0079606420931, 'Rocket': 66.42584044498216, 'Skateboard': 74.60135707907443, 'Table': 80.82813271557887}
==================================================

EPOCH 94 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.383, iteration=0.230, train_acc=94.39, train_loss_seg=0.133, train_macc=89.48, train_miou=83.36]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.63it/s, test_acc=92.85, test_loss_seg=0.115, test_macc=86.97, test_miou=79.65]
==================================================
test_loss_seg = 0.11558126658201218
test_acc = 92.85348434111364
test_macc = 86.9709719790031
test_miou = 79.6544670434743
test_acc_per_class = {'Airplane': 91.08334098517874, 'Bag': 94.8794724075197, 'Cap': 90.49543617456065, 'Car': 91.11493726077963, 'Chair': 94.92446225852473, 'Earphone': 91.76175023552811, 'Guitar': 96.11971920158375, 'Knife': 91.48449982041342, 'Lamp': 89.06578530225237, 'Laptop': 98.12184221332207, 'Motorbike': 86.78540291443517, 'Mug': 99.31101835500615, 'Pistol': 95.61626718164263, 'Rocket': 84.14955946151437, 'Skateboard': 95.7025654382081, 'Table': 95.03969024734872}
test_macc_per_class = {'Airplane': 90.07764283424264, 'Bag': 85.81495778137818, 'Cap': 82.15098933644285, 'Car': 84.84020716214731, 'Chair': 90.61421967916878, 'Earphone': 73.1585086403548, 'Guitar': 94.28790648188645, 'Knife': 91.41851198465312, 'Lamp': 89.42258438367298, 'Laptop': 98.07212769633402, 'Motorbike': 77.71033042060644, 'Mug': 96.16295689637829, 'Pistol': 84.92512844734563, 'Rocket': 74.991049288561, 'Skateboard': 88.27556500838621, 'Table': 89.61286562249107}
test_miou_per_class = {'Airplane': 81.55551547472136, 'Bag': 77.18890252996175, 'Cap': 76.38674629096212, 'Car': 76.24955944952066, 'Chair': 84.97961156978468, 'Earphone': 66.18228483806475, 'Guitar': 89.93809791957892, 'Knife': 84.25015070573426, 'Lamp': 74.85312108619114, 'Laptop': 96.28992663577411, 'Motorbike': 68.73858862750009, 'Mug': 93.75772987411179, 'Pistol': 80.37308707117795, 'Rocket': 64.25819767931713, 'Skateboard': 75.95197482857127, 'Table': 83.51797811461671}
==================================================

EPOCH 95 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.381, iteration=0.233, train_acc=95.58, train_loss_seg=0.130, train_macc=90.19, train_miou=85.16]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.64it/s, test_acc=93.10, test_loss_seg=0.136, test_macc=86.78, test_miou=80.00]
==================================================
test_loss_seg = 0.13685207068920135
test_acc = 93.10178777385265
test_macc = 86.78739465025231
test_miou = 80.00258100180159
test_acc_per_class = {'Airplane': 91.34933720719086, 'Bag': 95.2391713747646, 'Cap': 93.4859886431836, 'Car': 91.37187122698755, 'Chair': 94.88479746764524, 'Earphone': 93.41892367217727, 'Guitar': 96.07985749419066, 'Knife': 91.55502199889851, 'Lamp': 89.30681842570269, 'Laptop': 98.17062266782705, 'Motorbike': 87.17818606566009, 'Mug': 99.01802311484684, 'Pistol': 95.47437484812367, 'Rocket': 81.8440853034148, 'Skateboard': 96.3421579315619, 'Table': 94.90936693946716}
test_macc_per_class = {'Airplane': 88.8548628020447, 'Bag': 78.39795363861309, 'Cap': 87.60275331607691, 'Car': 87.30896186182679, 'Chair': 91.60552124255385, 'Earphone': 73.10368376105573, 'Guitar': 93.80575418266034, 'Knife': 91.49712193197594, 'Lamp': 89.20608296841256, 'Laptop': 98.19315502289795, 'Motorbike': 82.17271052539813, 'Mug': 95.11769453058172, 'Pistol': 83.55762169846788, 'Rocket': 72.1653614085911, 'Skateboard': 87.5540655366732, 'Table': 88.45500997620702}
test_miou_per_class = {'Airplane': 81.8484199065403, 'Bag': 75.36943780752388, 'Cap': 83.22261563890746, 'Car': 77.56658066608361, 'Chair': 85.34762291827893, 'Earphone': 68.82317828357358, 'Guitar': 89.81155674320021, 'Knife': 84.37460624780645, 'Lamp': 76.13985279035123, 'Laptop': 96.39008565254235, 'Motorbike': 70.61659994191245, 'Mug': 91.42375718937247, 'Pistol': 79.33559310486933, 'Rocket': 59.39635174241167, 'Skateboard': 77.79302005871838, 'Table': 82.58201733673296}
==================================================

EPOCH 96 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.377, iteration=0.234, train_acc=94.94, train_loss_seg=0.128, train_macc=91.00, train_miou=85.61]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.64it/s, test_acc=93.01, test_loss_seg=0.104, test_macc=86.40, test_miou=79.90]
==================================================
test_loss_seg = 0.10471294075250626
test_acc = 93.01601794612648
test_macc = 86.40539439988467
test_miou = 79.90635923921728
test_acc_per_class = {'Airplane': 91.14787989186219, 'Bag': 95.6687898089172, 'Cap': 90.82066036443348, 'Car': 91.31835089894606, 'Chair': 94.72913194832154, 'Earphone': 93.31046312178388, 'Guitar': 96.07324825010511, 'Knife': 92.79677319267763, 'Lamp': 90.36074897409341, 'Laptop': 98.15521249329439, 'Motorbike': 84.93359397458747, 'Mug': 99.25112331502746, 'Pistol': 96.00179384572927, 'Rocket': 82.61155163515225, 'Skateboard': 96.3797549424435, 'Table': 94.69721048064922}
test_macc_per_class = {'Airplane': 89.15199990015873, 'Bag': 83.72409789470619, 'Cap': 82.60678299391935, 'Car': 86.84734592529931, 'Chair': 90.99776625112438, 'Earphone': 71.73459304480447, 'Guitar': 94.39935474558658, 'Knife': 92.76173494516907, 'Lamp': 86.11426431378592, 'Laptop': 98.1327597980705, 'Motorbike': 73.53013606817656, 'Mug': 94.58203217985393, 'Pistol': 87.34714620603383, 'Rocket': 71.83576783587885, 'Skateboard': 89.70504726118837, 'Table': 89.0154810343985}
test_miou_per_class = {'Airplane': 81.7091865087779, 'Bag': 78.29044790973634, 'Cap': 76.9400331938652, 'Car': 77.09493054735182, 'Chair': 84.9577131723765, 'Earphone': 66.56869139620404, 'Guitar': 89.81608474985372, 'Knife': 86.53963834431178, 'Lamp': 77.57622605291934, 'Laptop': 96.35694301239293, 'Motorbike': 64.57930510050429, 'Mug': 93.12963810559552, 'Pistol': 82.61282681816687, 'Rocket': 61.393083544394, 'Skateboard': 78.6996877793615, 'Table': 82.23731159166512}
==================================================

EPOCH 97 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.383, iteration=0.237, train_acc=94.98, train_loss_seg=0.118, train_macc=90.68, train_miou=85.16]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.64it/s, test_acc=93.37, test_loss_seg=0.168, test_macc=87.51, test_miou=80.80]
==================================================
test_loss_seg = 0.16858600080013275
test_acc = 93.3726286675049
test_macc = 87.51132029760808
test_miou = 80.8068111630587
test_acc_per_class = {'Airplane': 90.94032997130873, 'Bag': 95.7348748193571, 'Cap': 94.29349101318691, 'Car': 91.43918596508624, 'Chair': 94.88049803686154, 'Earphone': 91.57791822340165, 'Guitar': 96.15990237980051, 'Knife': 92.90471183215662, 'Lamp': 91.01394403261281, 'Laptop': 98.11570346870978, 'Motorbike': 86.96851034495965, 'Mug': 99.32558357802716, 'Pistol': 95.72660352695668, 'Rocket': 83.95226645293437, 'Skateboard': 96.16443877208303, 'Table': 94.76409626263563}
test_macc_per_class = {'Airplane': 90.32767934827336, 'Bag': 82.35025364524621, 'Cap': 89.37579090181377, 'Car': 86.74686852921086, 'Chair': 91.64992023048669, 'Earphone': 71.36439985686746, 'Guitar': 94.81667377787687, 'Knife': 92.88823894778969, 'Lamp': 89.42682353055869, 'Laptop': 98.04649087339428, 'Motorbike': 80.00942476890124, 'Mug': 95.91105419132795, 'Pistol': 84.15181376881749, 'Rocket': 75.44892509140352, 'Skateboard': 89.16510424534566, 'Table': 88.50166305441547}
test_miou_per_class = {'Airplane': 81.48507094181876, 'Bag': 78.56555968423405, 'Cap': 85.16778558088515, 'Car': 77.30538530536867, 'Chair': 85.43198773332949, 'Earphone': 64.61889476435002, 'Guitar': 90.24490767424632, 'Knife': 86.74247081391024, 'Lamp': 80.07133629529048, 'Laptop': 96.27681261649958, 'Motorbike': 68.76280820765497, 'Mug': 93.86945618139819, 'Pistol': 80.14826124120047, 'Rocket': 64.25450041488746, 'Skateboard': 77.86338391474168, 'Table': 82.10035723912368}
==================================================
miou: 80.75561479818244 -> 80.8068111630587

EPOCH 98 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.379, iteration=0.231, train_acc=94.97, train_loss_seg=0.120, train_macc=91.28, train_miou=86.00]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:34<00:00,  2.64it/s, test_acc=93.21, test_loss_seg=0.126, test_macc=86.39, test_miou=80.15]
==================================================
test_loss_seg = 0.12605129182338715
test_acc = 93.21017534107288
test_macc = 86.39076702086211
test_miou = 80.15821616366036
test_acc_per_class = {'Airplane': 91.40600701680953, 'Bag': 95.52687680828384, 'Cap': 92.03519649858667, 'Car': 91.2845079723033, 'Chair': 94.89557004123985, 'Earphone': 93.98256729793117, 'Guitar': 96.06193885633797, 'Knife': 92.02952076112553, 'Lamp': 90.68358079242942, 'Laptop': 98.08393397162357, 'Motorbike': 86.29940257352942, 'Mug': 99.10680757470092, 'Pistol': 95.62704768838734, 'Rocket': 83.1454190340909, 'Skateboard': 96.2145215068925, 'Table': 94.97990706289427}
test_macc_per_class = {'Airplane': 89.6416448623498, 'Bag': 80.24838303963804, 'Cap': 84.96428238920176, 'Car': 87.55697061109828, 'Chair': 90.85785945523126, 'Earphone': 71.765866369458, 'Guitar': 93.50921017134773, 'Knife': 92.02317562649749, 'Lamp': 89.64039281530685, 'Laptop': 97.978564375121, 'Motorbike': 71.97188985305648, 'Mug': 95.53265762336156, 'Pistol': 84.34944412853754, 'Rocket': 76.22019898933345, 'Skateboard': 87.4664340242488, 'Table': 88.5252980000058}
test_miou_per_class = {'Airplane': 81.93198868454705, 'Bag': 76.80253491037557, 'Cap': 79.84888951694897, 'Car': 77.26995028684135, 'Chair': 85.1896855003547, 'Earphone': 68.06403079945648, 'Guitar': 89.61422962282803, 'Knife': 85.23284612302389, 'Lamp': 81.37455891804913, 'Laptop': 96.21059379347913, 'Motorbike': 65.39987975604, 'Mug': 92.11442577262132, 'Pistol': 80.11358169699014, 'Rocket': 63.24042708465375, 'Skateboard': 77.3388542802266, 'Table': 82.78498187212945}
==================================================

EPOCH 99 / 100
100%|█████████████████████████████| 438/438 [05:15<00:00,  1.39it/s, data_loading=0.381, iteration=0.233, train_acc=95.16, train_loss_seg=0.117, train_macc=90.13, train_miou=84.97]
100%|████████████████████████████████████████████████████████████████████████| 90/90 [00:33<00:00,  2.65it/s, test_acc=90.95, test_loss_seg=0.145, test_macc=85.65, test_miou=78.11]
==================================================
test_loss_seg = 0.1452503651380539
test_acc = 90.95297564684472
test_macc = 85.65776756924717
test_miou = 78.11274449046947
test_acc_per_class = {'Airplane': 91.23435971464201, 'Bag': 95.17076683977105, 'Cap': 92.69182796692382, 'Car': 91.27479035199737, 'Chair': 94.95432327375337, 'Earphone': 60.35234606663178, 'Guitar': 96.3230757841226, 'Knife': 91.89853572884333, 'Lamp': 91.12655373722723, 'Laptop': 97.97885387024182, 'Motorbike': 85.44174687544893, 'Mug': 99.07868963188112, 'Pistol': 96.0812690085588, 'Rocket': 80.49982244318183, 'Skateboard': 96.08940430839766, 'Table': 95.05124474789305}
test_macc_per_class = {'Airplane': 88.25942547942684, 'Bag': 77.48168720099096, 'Cap': 86.12101404505337, 'Car': 85.18370125671044, 'Chair': 90.6221513472541, 'Earphone': 58.11280106850963, 'Guitar': 94.66731076596221, 'Knife': 91.89939621009773, 'Lamp': 88.7259710347473, 'Laptop': 97.89071223623822, 'Motorbike': 85.98362183299261, 'Mug': 94.73713875984299, 'Pistol': 88.98826862494582, 'Rocket': 63.14551026531346, 'Skateboard': 89.51093897539162, 'Table': 89.19463200447773}
test_miou_per_class = {'Airplane': 81.49246129952866, 'Bag': 74.3862941399426, 'Cap': 81.30122292984763, 'Car': 76.60484812811767, 'Chair': 85.18206870127682, 'Earphone': 37.40464086586233, 'Guitar': 90.48232783792707, 'Knife': 85.01125471838384, 'Lamp': 80.28907756433219, 'Laptop': 96.009515440567, 'Motorbike': 70.72768977006673, 'Mug': 91.83625047863121, 'Pistol': 83.83845099881488, 'Rocket': 54.208046584411996, 'Skateboard': 77.90017930279588, 'Table': 83.12958308700505}


BEST: 
* loss_seg: 0.08968085050582886
* acc: 93.38952466037554
* miou: 80.8068111630587
* macc: 87.90512718023592
* miou_per_class for best_miou: 
{
    'Airplane': 81.48507094181876, 
    'Bag': 78.56555968423405, 
    'Cap': 85.16778558088515, 
    'Car': 77.30538530536867, 
    'Chair': 85.43198773332949, 
    'Earphone': 64.61889476435002, 
    'Guitar': 90.24490767424632, 
    'Knife': 86.74247081391024, 
    'Lamp': 80.07133629529048, 
    'Laptop': 96.27681261649958, 
    'Motorbike': 68.76280820765497, 
    'Mug': 93.86945618139819, 
    'Pistol': 80.14826124120047, 
    'Rocket': 64.25450041488746, 
    'Skateboard': 77.86338391474168, 
    'Table': 82.10035723912368
}
```