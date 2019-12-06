import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import functional as FPModule
from models.base_model import MLP, FPModule
from .modules import KPConv

class UnetBasedModel(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, opt, num_classes):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetBasedModel, self).__init__()
        
        assert len(opt.convs) == len(opt.fpms)

        # construct unet structure
        contains_global = hasattr(opt, "global")
        if contains_global:
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        if len(opt.convs) > (2 if contains_global else 1):
            for _ in range(len(opt.convs) - 2):
                unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(num_classes, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        if outermost:
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class SegmentationModel(torch.nn.Module):
    def __init__(self, opt, num_classes):
        super(SegmentationModel, self).__init__()
        
        self.convs = nn.ModuleList()
        self.fpm_nn = nn.ModuleList()

        for conv in opt.convs:
            conv_opt = getattr(opt.convs, conv)
            self.convs.append(KPConv(**conv_opt))
        
        import pdb; pdb.set_trace()

        for fpm in opt.fpms: 
            fpm_opt = getattr(opt.fpms, fpm)
            self.fpm_nn.append(FPModule(fpm_opt.dim_in, MLP([fpm_opt.mlp_in, fpm_opt.mlp_out])))

        self.mlp_cls = MLP([opt.mlp_cls.dim_in] + opt.mlp_cls.fcs + [num_classes])

    def get_from_data(self, datas, index):
        return out

    def forward(self, data):
        input = (data.x, data.pos, data.batch)
        datas = [input]

        for conv in self.convs:
            input = conv(*input)
            datas.append(input)

        for fpn in fpm_nn[::-1]:
             = fpn(self.get_from_data())

        fp2_out = self.fp2_module(*kp2_out, *kp1_out)
    
        x = self.mlp_cls(x[0])
        return F.log_softmax(x, dim=-1)
    
    def __name__(self):
        return "KPConv SEGMENTATION MODEL"