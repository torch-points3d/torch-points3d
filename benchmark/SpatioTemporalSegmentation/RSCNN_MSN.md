# https://github.com/Yochengliu/Relation-Shape-CNN/blob/master/models/rscnn_msn_seg.py
```
RSCNN_MSN(
  (SA_modules): ModuleList(
    (0): PointnetSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
        (2): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): SharedRSConv(
          (RSConvLayer0): RSConvLayer(
            (RS_Conv): RSConv(
              (bn_rsconv): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_channel_raising): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_mapping): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activation): ReLU(inplace)
              (mapping_func1): Conv2d(10, 32, kernel_size=(1, 1), stride=(1, 1))
              (mapping_func2): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
              (cr_mapping): Conv1d(16, 64, kernel_size=(1,), stride=(1,))
              (xyz_raising): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
            )
          )
        )
        (1): SharedRSConv(
          (RSConvLayer0): RSConvLayer(
            (RS_Conv): RSConv(
              (bn_rsconv): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_channel_raising): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_mapping): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activation): ReLU(inplace)
              (mapping_func1): Conv2d(10, 32, kernel_size=(1, 1), stride=(1, 1))
              (mapping_func2): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
              (cr_mapping): Conv1d(16, 64, kernel_size=(1,), stride=(1,))
              (xyz_raising): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
            )
          )
        )
        (2): SharedRSConv(
          (RSConvLayer0): RSConvLayer(
            (RS_Conv): RSConv(
              (bn_rsconv): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_channel_raising): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_mapping): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activation): ReLU(inplace)
              (mapping_func1): Conv2d(10, 32, kernel_size=(1, 1), stride=(1, 1))
              (mapping_func2): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
              (cr_mapping): Conv1d(16, 64, kernel_size=(1,), stride=(1,))
              (xyz_raising): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
            )
          )
        )
      )
    )
    (1): PointnetSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
        (2): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): SharedRSConv(
          (RSConvLayer0): RSConvLayer(
            (RS_Conv): RSConv(
              (bn_rsconv): BatchNorm2d(195, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_channel_raising): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_mapping): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activation): ReLU(inplace)
              (mapping_func1): Conv2d(10, 32, kernel_size=(1, 1), stride=(1, 1))
              (mapping_func2): Conv2d(32, 195, kernel_size=(1, 1), stride=(1, 1))
              (cr_mapping): Conv1d(195, 128, kernel_size=(1,), stride=(1,))
            )
          )
        )
        (1): SharedRSConv(
          (RSConvLayer0): RSConvLayer(
            (RS_Conv): RSConv(
              (bn_rsconv): BatchNorm2d(195, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_channel_raising): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_mapping): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activation): ReLU(inplace)
              (mapping_func1): Conv2d(10, 32, kernel_size=(1, 1), stride=(1, 1))
              (mapping_func2): Conv2d(32, 195, kernel_size=(1, 1), stride=(1, 1))
              (cr_mapping): Conv1d(195, 128, kernel_size=(1,), stride=(1,))
            )
          )
        )
        (2): SharedRSConv(
          (RSConvLayer0): RSConvLayer(
            (RS_Conv): RSConv(
              (bn_rsconv): BatchNorm2d(195, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_channel_raising): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_mapping): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activation): ReLU(inplace)
              (mapping_func1): Conv2d(10, 32, kernel_size=(1, 1), stride=(1, 1))
              (mapping_func2): Conv2d(32, 195, kernel_size=(1, 1), stride=(1, 1))
              (cr_mapping): Conv1d(195, 128, kernel_size=(1,), stride=(1,))
            )
          )
        )
      )
    )
    (2): PointnetSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
        (2): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): SharedRSConv(
          (RSConvLayer0): RSConvLayer(
            (RS_Conv): RSConv(
              (bn_rsconv): BatchNorm2d(387, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_channel_raising): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_mapping): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activation): ReLU(inplace)
              (mapping_func1): Conv2d(10, 64, kernel_size=(1, 1), stride=(1, 1))
              (mapping_func2): Conv2d(64, 387, kernel_size=(1, 1), stride=(1, 1))
              (cr_mapping): Conv1d(387, 256, kernel_size=(1,), stride=(1,))
            )
          )
        )
        (1): SharedRSConv(
          (RSConvLayer0): RSConvLayer(
            (RS_Conv): RSConv(
              (bn_rsconv): BatchNorm2d(387, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_channel_raising): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_mapping): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activation): ReLU(inplace)
              (mapping_func1): Conv2d(10, 64, kernel_size=(1, 1), stride=(1, 1))
              (mapping_func2): Conv2d(64, 387, kernel_size=(1, 1), stride=(1, 1))
              (cr_mapping): Conv1d(387, 256, kernel_size=(1,), stride=(1,))
            )
          )
        )
        (2): SharedRSConv(
          (RSConvLayer0): RSConvLayer(
            (RS_Conv): RSConv(
              (bn_rsconv): BatchNorm2d(387, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_channel_raising): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_mapping): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activation): ReLU(inplace)
              (mapping_func1): Conv2d(10, 64, kernel_size=(1, 1), stride=(1, 1))
              (mapping_func2): Conv2d(64, 387, kernel_size=(1, 1), stride=(1, 1))
              (cr_mapping): Conv1d(387, 256, kernel_size=(1,), stride=(1,))
            )
          )
        )
      )
    )
    (3): PointnetSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
        (2): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): SharedRSConv(
          (RSConvLayer0): RSConvLayer(
            (RS_Conv): RSConv(
              (bn_rsconv): BatchNorm2d(771, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_channel_raising): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_mapping): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activation): ReLU(inplace)
              (mapping_func1): Conv2d(10, 128, kernel_size=(1, 1), stride=(1, 1))
              (mapping_func2): Conv2d(128, 771, kernel_size=(1, 1), stride=(1, 1))
              (cr_mapping): Conv1d(771, 512, kernel_size=(1,), stride=(1,))
            )
          )
        )
        (1): SharedRSConv(
          (RSConvLayer0): RSConvLayer(
            (RS_Conv): RSConv(
              (bn_rsconv): BatchNorm2d(771, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_channel_raising): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_mapping): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activation): ReLU(inplace)
              (mapping_func1): Conv2d(10, 128, kernel_size=(1, 1), stride=(1, 1))
              (mapping_func2): Conv2d(128, 771, kernel_size=(1, 1), stride=(1, 1))
              (cr_mapping): Conv1d(771, 512, kernel_size=(1,), stride=(1,))
            )
          )
        )
        (2): SharedRSConv(
          (RSConvLayer0): RSConvLayer(
            (RS_Conv): RSConv(
              (bn_rsconv): BatchNorm2d(771, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_channel_raising): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (bn_mapping): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activation): ReLU(inplace)
              (mapping_func1): Conv2d(10, 128, kernel_size=(1, 1), stride=(1, 1))
              (mapping_func2): Conv2d(128, 771, kernel_size=(1, 1), stride=(1, 1))
              (cr_mapping): Conv1d(771, 512, kernel_size=(1,), stride=(1,))
            )
          )
        )
      )
    )
    (4): PointnetSAModule(
      (groupers): ModuleList(
        (0): GroupAll()
      )
      (mlps): ModuleList(
        (0): GloAvgConv(
          (conv_avg): Conv2d(1539, 128, kernel_size=(1, 1), stride=(1, 1))
          (bn_avg): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
        )
      )
    )
    (5): PointnetSAModule(
      (groupers): ModuleList(
        (0): GroupAll()
      )
      (mlps): ModuleList(
        (0): GloAvgConv(
          (conv_avg): Conv2d(771, 128, kernel_size=(1, 1), stride=(1, 1))
          (bn_avg): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
        )
      )
    )
  )
  (FP_modules): ModuleList(
    (0): PointnetFPModule(
      (mlp): SharedMLP(
        (layer0): Conv2d(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU(inplace)
        )
        (layer1): Conv2d(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU(inplace)
        )
      )
    )
    (1): PointnetFPModule(
      (mlp): SharedMLP(
        (layer0): Conv2d(
          (conv): Conv2d(704, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU(inplace)
        )
        (layer1): Conv2d(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU(inplace)
        )
      )
    )
    (2): PointnetFPModule(
      (mlp): SharedMLP(
        (layer0): Conv2d(
          (conv): Conv2d(896, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU(inplace)
        )
        (layer1): Conv2d(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU(inplace)
        )
      )
    )
    (3): PointnetFPModule(
      (mlp): SharedMLP(
        (layer0): Conv2d(
          (conv): Conv2d(2304, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU(inplace)
        )
        (layer1): Conv2d(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU(inplace)
        )
      )
    )
  )
  (FC_layer): Sequential(
    (0): Conv1d(
      (conv): Conv1d(400, 128, kernel_size=(1,), stride=(1,), bias=False)
      (bn): BatchNorm1d(
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
Model size = %i 3,488,705

###########################################################################

-1 -3 -2
torch.Size([12, 64, 3]) torch.Size([12, 16, 3]) torch.Size([12, 768, 64]) torch.Size([12, 1536, 16]) SharedMLP(
  (layer0): Conv2d(
    (conv): Conv2d(2304, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(
      (bn): BatchNorm2d(512, eps=1e-05, momentum=1.8, affine=True, track_running_stats=True)
    )
    (activation): ReLU(inplace)
  )
  (layer1): Conv2d(
    (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(
      (bn): BatchNorm2d(512, eps=1e-05, momentum=1.8, affine=True, track_running_stats=True)
    )
    (activation): ReLU(inplace)
  )
)
torch.Size([12, 512, 64, 1])



-2 -4 -3
torch.Size([12, 256, 3]) torch.Size([12, 64, 3]) torch.Size([12, 384, 256]) torch.Size([12, 512, 64]) SharedMLP(
  (layer0): Conv2d(
    (conv): Conv2d(896, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(
      (bn): BatchNorm2d(512, eps=1e-05, momentum=1.8, affine=True, track_running_stats=True)
    )
    (activation): ReLU(inplace)
  )
  (layer1): Conv2d(
    (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(
      (bn): BatchNorm2d(512, eps=1e-05, momentum=1.8, affine=True, track_running_stats=True)
    )
    (activation): ReLU(inplace)
  )
)
torch.Size([12, 512, 256, 1])



-3 -5 -4
torch.Size([12, 1024, 3]) torch.Size([12, 256, 3]) torch.Size([12, 192, 1024]) torch.Size([12, 512, 256]) SharedMLP(
  (layer0): Conv2d(
    (conv): Conv2d(704, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(
      (bn): BatchNorm2d(256, eps=1e-05, momentum=1.8, affine=True, track_running_stats=True)
    )
    (activation): ReLU(inplace)
  )
  (layer1): Conv2d(
    (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(
      (bn): BatchNorm2d(256, eps=1e-05, momentum=1.8, affine=True, track_running_stats=True)
    )
    (activation): ReLU(inplace)
  )
)
torch.Size([12, 256, 1024, 1])



-4 -6 -5

###################################################################################

PointnetSAModuleMSG(
  (groupers): ModuleList(
    (0): QueryAndGroup()
    (1): QueryAndGroup()
    (2): QueryAndGroup()
  )
  (mlps): ModuleList(
    (0): SharedRSConv(
      (RSConvLayer0): RSConvLayer(
        (RS_Conv): RSConv(
          (bn_rsconv): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_channel_raising): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_mapping): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
          (mapping_func1): Conv2d(10, 32, kernel_size=(1, 1), stride=(1, 1))
          (mapping_func2): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
          (cr_mapping): Conv1d(16, 64, kernel_size=(1,), stride=(1,))
          (xyz_raising): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    (1): SharedRSConv(
      (RSConvLayer0): RSConvLayer(
        (RS_Conv): RSConv(
          (bn_rsconv): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_channel_raising): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_mapping): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
          (mapping_func1): Conv2d(10, 32, kernel_size=(1, 1), stride=(1, 1))
          (mapping_func2): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
          (cr_mapping): Conv1d(16, 64, kernel_size=(1,), stride=(1,))
          (xyz_raising): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    (2): SharedRSConv(
      (RSConvLayer0): RSConvLayer(
        (RS_Conv): RSConv(
          (bn_rsconv): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_channel_raising): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_mapping): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
          (mapping_func1): Conv2d(10, 32, kernel_size=(1, 1), stride=(1, 1))
          (mapping_func2): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
          (cr_mapping): Conv1d(16, 64, kernel_size=(1,), stride=(1,))
          (xyz_raising): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
  )
) size = 2800
PointnetSAModuleMSG(
  (groupers): ModuleList(
    (0): QueryAndGroup()
    (1): QueryAndGroup()
    (2): QueryAndGroup()
  )
  (mlps): ModuleList(
    (0): SharedRSConv(
      (RSConvLayer0): RSConvLayer(
        (RS_Conv): RSConv(
          (bn_rsconv): BatchNorm2d(195, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_channel_raising): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_mapping): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
          (mapping_func1): Conv2d(10, 32, kernel_size=(1, 1), stride=(1, 1))
          (mapping_func2): Conv2d(32, 195, kernel_size=(1, 1), stride=(1, 1))
          (cr_mapping): Conv1d(195, 128, kernel_size=(1,), stride=(1,))
        )
      )
    )
    (1): SharedRSConv(
      (RSConvLayer0): RSConvLayer(
        (RS_Conv): RSConv(
          (bn_rsconv): BatchNorm2d(195, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_channel_raising): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_mapping): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
          (mapping_func1): Conv2d(10, 32, kernel_size=(1, 1), stride=(1, 1))
          (mapping_func2): Conv2d(32, 195, kernel_size=(1, 1), stride=(1, 1))
          (cr_mapping): Conv1d(195, 128, kernel_size=(1,), stride=(1,))
        )
      )
    )
    (2): SharedRSConv(
      (RSConvLayer0): RSConvLayer(
        (RS_Conv): RSConv(
          (bn_rsconv): BatchNorm2d(195, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_channel_raising): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_mapping): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
          (mapping_func1): Conv2d(10, 32, kernel_size=(1, 1), stride=(1, 1))
          (mapping_func2): Conv2d(32, 195, kernel_size=(1, 1), stride=(1, 1))
          (cr_mapping): Conv1d(195, 128, kernel_size=(1,), stride=(1,))
        )
      )
    )
  )
) size = 34101
PointnetSAModuleMSG(
  (groupers): ModuleList(
    (0): QueryAndGroup()
    (1): QueryAndGroup()
    (2): QueryAndGroup()
  )
  (mlps): ModuleList(
    (0): SharedRSConv(
      (RSConvLayer0): RSConvLayer(
        (RS_Conv): RSConv(
          (bn_rsconv): BatchNorm2d(387, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_channel_raising): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_mapping): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
          (mapping_func1): Conv2d(10, 64, kernel_size=(1, 1), stride=(1, 1))
          (mapping_func2): Conv2d(64, 387, kernel_size=(1, 1), stride=(1, 1))
          (cr_mapping): Conv1d(387, 256, kernel_size=(1,), stride=(1,))
        )
      )
    )
    (1): SharedRSConv(
      (RSConvLayer0): RSConvLayer(
        (RS_Conv): RSConv(
          (bn_rsconv): BatchNorm2d(387, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_channel_raising): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_mapping): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
          (mapping_func1): Conv2d(10, 64, kernel_size=(1, 1), stride=(1, 1))
          (mapping_func2): Conv2d(64, 387, kernel_size=(1, 1), stride=(1, 1))
          (cr_mapping): Conv1d(387, 256, kernel_size=(1,), stride=(1,))
        )
      )
    )
    (2): SharedRSConv(
      (RSConvLayer0): RSConvLayer(
        (RS_Conv): RSConv(
          (bn_rsconv): BatchNorm2d(387, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_channel_raising): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_mapping): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
          (mapping_func1): Conv2d(10, 64, kernel_size=(1, 1), stride=(1, 1))
          (mapping_func2): Conv2d(64, 387, kernel_size=(1, 1), stride=(1, 1))
          (cr_mapping): Conv1d(387, 256, kernel_size=(1,), stride=(1,))
        )
      )
    )
  )
) size = 129525
PointnetSAModuleMSG(
  (groupers): ModuleList(
    (0): QueryAndGroup()
    (1): QueryAndGroup()
    (2): QueryAndGroup()
  )
  (mlps): ModuleList(
    (0): SharedRSConv(
      (RSConvLayer0): RSConvLayer(
        (RS_Conv): RSConv(
          (bn_rsconv): BatchNorm2d(771, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_channel_raising): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_mapping): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
          (mapping_func1): Conv2d(10, 128, kernel_size=(1, 1), stride=(1, 1))
          (mapping_func2): Conv2d(128, 771, kernel_size=(1, 1), stride=(1, 1))
          (cr_mapping): Conv1d(771, 512, kernel_size=(1,), stride=(1,))
        )
      )
    )
    (1): SharedRSConv(
      (RSConvLayer0): RSConvLayer(
        (RS_Conv): RSConv(
          (bn_rsconv): BatchNorm2d(771, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_channel_raising): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_mapping): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
          (mapping_func1): Conv2d(10, 128, kernel_size=(1, 1), stride=(1, 1))
          (mapping_func2): Conv2d(128, 771, kernel_size=(1, 1), stride=(1, 1))
          (cr_mapping): Conv1d(771, 512, kernel_size=(1,), stride=(1,))
        )
      )
    )
    (2): SharedRSConv(
      (RSConvLayer0): RSConvLayer(
        (RS_Conv): RSConv(
          (bn_rsconv): BatchNorm2d(771, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_channel_raising): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_xyz_raising): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn_mapping): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU(inplace)
          (mapping_func1): Conv2d(10, 128, kernel_size=(1, 1), stride=(1, 1))
          (mapping_func2): Conv2d(128, 771, kernel_size=(1, 1), stride=(1, 1))
          (cr_mapping): Conv1d(771, 512, kernel_size=(1,), stride=(1,))
        )
      )
    )
  )
) size = 504693
PointnetSAModule(
  (groupers): ModuleList(
    (0): GroupAll()
  )
  (mlps): ModuleList(
    (0): GloAvgConv(
      (conv_avg): Conv2d(1539, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn_avg): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation): ReLU(inplace)
    )
  )
) size = 197376
PointnetSAModule(
  (groupers): ModuleList(
    (0): GroupAll()
  )
  (mlps): ModuleList(
    (0): GloAvgConv(
      (conv_avg): Conv2d(771, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn_avg): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation): ReLU(inplace)
    )
  )
) size = 99072
PointnetFPModule(
  (mlp): SharedMLP(
    (layer0): Conv2d(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (activation): ReLU(inplace)
    )
    (layer1): Conv2d(
      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (activation): ReLU(inplace)
    )
  )
) size = 49664
PointnetFPModule(
  (mlp): SharedMLP(
    (layer0): Conv2d(
      (conv): Conv2d(704, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (activation): ReLU(inplace)
    )
    (layer1): Conv2d(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (activation): ReLU(inplace)
    )
  )
) size = 246784
PointnetFPModule(
  (mlp): SharedMLP(
    (layer0): Conv2d(
      (conv): Conv2d(896, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (activation): ReLU(inplace)
    )
    (layer1): Conv2d(
      (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (activation): ReLU(inplace)
    )
  )
) size = 722944
PointnetFPModule(
  (mlp): SharedMLP(
    (layer0): Conv2d(
      (conv): Conv2d(2304, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (activation): ReLU(inplace)
    )
    (layer1): Conv2d(
      (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (activation): ReLU(inplace)
    )
  )
) size = 1443840
Model size = %i 3488705
```
