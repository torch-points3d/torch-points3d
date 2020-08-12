
import logging
import torch.nn.functional as F
from typing import Any
import torch
import numpy as np
from torch_points3d.models.base_model import BaseModel
from torch_points3d.modules.DGCN.main_modules import PointGenerator, Discriminator
from torch_points3d.modules.DGCN.helpers import ChamferLoss
from torch_points3d.utils.model_utils import freeze_params, unfreeze_params
from torch_points3d.core.spatial_ops import KNNNeighbourFinder
import torch_geometric
from torch_points3d.datasets.batch import SimpleBatch
import itertools
import torch.nn as nn
from torch_points3d.core.schedulers.lr_schedulers import instantiate_scheduler
from torch_points3d.core.schedulers.bn_schedulers import instantiate_bn_scheduler
from torch_points3d.core.common_modules.gathering import gather
from torch_geometric.data import DataLoader, InMemoryDataset, extract_zip, Data

log = logging.getLogger(__name__)

def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in tensors.items():
        print('{}\t{}'.format(*line))

class DGCN(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        super(DGCN, self).__init__(option)

        self.scales = option.scales

        self.latent_space = 2**dataset.num_classes
        self.generator = PointGenerator(self.scales, self.latent_space, option.generator)
        
        self.discriminators = nn.ModuleList()
        for i in range(len(self.scales)):
            self.discriminators.append(Discriminator(option.discriminator.fc_layers[i], option.discriminator.classifier))

        #self.sampler = DenseFPSSampler(num_to_sample=20)
        self.neighbour_finder = KNNNeighbourFinder(k=20)
        self.chamfer_loss = ChamferLoss()

        losses = ["g_similar_loss", "g_loss", "g_discriminator_loss"]
        visuals = []
        for i in range(len(self.scales)):
            losses.append("d_loss_%d" % (self.scales[i]))
            visuals.append("data_visual_%d" % (self.scales[i]))
        #print(losses)
        self.loss_names = losses

        self.visual_names = visuals


    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.batch_size = len(data[data.keys[0]])
        #print("batch size: %s" % (self.batch_size))
        self.input = data.to(device)

    def compute_mean_covariance(self, points):
        bs, ch, nump = points.size()
        # ----------------------------------------------------------------
        mu = points.mean(dim=-1, keepdim=True)  # Bx3xN -> Bx3x1
        # ----------------------------------------------------------------
        tmp = points - mu.repeat(1, 1, nump)    # Bx3xN - Bx3xN -> Bx3xN
        tmp_transpose = tmp.transpose(1, 2)     # Bx3xN -> BxNx3
        covariance = torch.bmm(tmp, tmp_transpose)
        covariance = covariance / nump
        return mu, covariance   # Bx3x1 Bx3x3

    def get_neighbors(self, pt1, pt2):
        #BxMx3 -> BMx3
        pt1_batch = np.indices(pt1.shape[0:2])[0].reshape(-1)
        pt1_batch = torch.from_numpy(pt1_batch).cuda()
        pt1 = pt1.view(-1, 3)

        pt2_batch = np.indices(pt2.shape[0:2])[0].reshape(-1)
        pt2_batch = torch.from_numpy(pt2_batch).cuda()
        pt2 = pt2.view(-1, 3)

        idx = self.neighbour_finder(pt1, pt2, pt1_batch, pt2_batch)


        pt1 = gather(pt1, idx)
        pt1 = pt1.reshape(pt1.shape[0], pt2_batch.shape[-1], 3, self.neighbour_finder.k)
        pt1 = pt1.transpose(2, 1)
        
        #print(pt1.shape)  # Bx3xMxK
        return pt1

    def get_local_pair(self, pt1, pt2):
        pt1_batch,pt1_N,pt1_M = pt1.size()
        pt2_batch,pt2_N,pt2_M = pt2.size()
        # pt1: Bx3xM    pt2: Bx3XN      (N > M)
        #print('pt1: {}      pt2: {}'.format(pt1.size(), pt2.size()))
        new_xyz = pt1.transpose(1, 2).contiguous()      # Bx3xM -> BxMx3
        pt1_trans = pt1.transpose(1, 2).contiguous()    # Bx3xM -> BxMx3
        pt2_trans = pt2.transpose(1, 2).contiguous()    # Bx3xN -> BxNx3
        
        #g_xyz1 = self.group(pt1_trans, new_xyz)     # Bx3xMxK
        g_xyz1 = self.get_neighbors(pt1_trans, new_xyz)
        #g_xyz1 = self.neighbour_finder(pt1_trans, new_xyz)#, batch_x=batch1, batch_y=batch1,)

        #print('g_xyz1: {}'.format(g_xyz1.size()))   
        #g_xyz2 = self.group(pt2_trans, new_xyz)     # Bx3xMxK
        g_xyz2 = self.get_neighbors(pt2_trans, new_xyz)
        #g_xyz2 = self.neighbour_finder(pt2_trans, new_xyz) #, batch_x=batch2, batch_y=batch1,)
        #print('g_xyz2: {}'.format(g_xyz2.size()))

        
        g_xyz1 = g_xyz1.transpose(1, 2).contiguous().view(-1, 3, 20)    # Bx3xMxK -> BxMx3xK -> (BM)x3xK
        #print('g_xyz1: {}'.format(g_xyz1.size()))   
        g_xyz2 = g_xyz2.transpose(1, 2).contiguous().view(-1, 3, 20)    # Bx3xMxK -> BxMx3xK -> (BM)x3xK
        #print('g_xyz2: {}'.format(g_xyz2.size()))   
        # print('====================== FPS ========================')
        # print(pt1.shape,g_xyz1.shape)
        # print(pt2.shape,g_xyz2.shape)
        mu1, var1 = self.compute_mean_covariance(g_xyz1) 
        mu2, var2 = self.compute_mean_covariance(g_xyz2) 
        #print('mu1: {} var1: {}'.format(mu1.size(), var1.size())) 
        #print('mu2: {} var2: {}'.format(mu2.size(), var2.size()))
        

        #--------------------------------------------------
        # like_mu12 = self.shape_loss_fn(mu1, mu2)
        # like_var12 = self.shape_loss_fn(var1, var2)
        #----------------------------------------------------
        #=========$$$  CD loss   $$$===============
        
        # print("p1,p2:",pt1.shape,pt2.shape)
        # print("mu2:",mu1.shape,mu2.shape,pt1_batch,pt1_N,pt1_M)
        mu1 = mu1.view(pt1_batch,-1,3)
        mu2 = mu2.view(pt2_batch,-1,3)

        var1 = var1.view(pt1_batch,-1,9)
        var2 = var2.view(pt2_batch,-1,9)

        like_mu12 = self.chamfer_loss(mu1,mu2) / float(pt1_M)

        like_var12 = self.chamfer_loss(var1,var2) / float(pt1_M)
        # import pdb
        # pdb.set_trace()


        #print('mu: {} var: {}'.format(like_mu12.item(), like_var12.item())) 
              
        return like_mu12, like_var12

    def get_pairs(self, data1, data2):
        #idx = np.arange(len(data1.pos()))

        pos1, batch1 = data1.pos, data1.batch
        pos2, batch2 = data2.pos, data2.batch
        g_xyz1 = self.neighbour_finder(pos1, pos1, batch_x=batch1, batch_y=batch1,)
        g_xyz2 = self.neighbour_finder(pos2, pos1, batch_x=batch2, batch_y=batch1,)

        mu1, var1 = self.compute_mean_covariance(g_xyz1) 
        mu2, var2 = self.compute_mean_covariance(g_xyz2) 

    def instantiate_optimizers(self, config):
        # Optimiser
        optimizer_opt = self.get_from_opt(
            config,
            ["training", "optim", "optimizer"],
            msg_err="optimizer needs to be defined within the training config",
        )       
        optmizer_cls_name = optimizer_opt.get("class")
        optimizer_cls = getattr(torch.optim, optmizer_cls_name)
        optimizer_params = {}
        if hasattr(optimizer_opt, "params"):
            optimizer_params = optimizer_opt.params
        self._optimizer = optimizer_cls(self.parameters(), **optimizer_params)
        self.g_optimizer = self._optimizer
        self.d_optimizers = []
        for i in range(len(self.scales)):
            self.d_optimizers.append(optimizer_cls(self.parameters(), **optimizer_params))

        # LR Scheduler
        scheduler_opt = self.get_from_opt(config, ["training", "optim", "lr_scheduler"])
        if scheduler_opt:
            update_lr_scheduler_on = config.update_lr_scheduler_on
            if update_lr_scheduler_on:
                self._update_lr_scheduler_on = update_lr_scheduler_on
            scheduler_opt.update_scheduler_on = self._update_lr_scheduler_on
            lr_scheduler = instantiate_scheduler(self._optimizer, scheduler_opt)
            self._add_scheduler("lr_scheduler", lr_scheduler)

        # BN Scheduler
        bn_scheduler_opt = self.get_from_opt(config, ["training", "optim", "bn_scheduler"])
        if bn_scheduler_opt:
            update_bn_scheduler_on = config.update_bn_scheduler_on
            if update_bn_scheduler_on:
                self._update_bn_scheduler_on = update_bn_scheduler_on
            bn_scheduler_opt.update_scheduler_on = self._update_bn_scheduler_on
            bn_scheduler = instantiate_bn_scheduler(self, bn_scheduler_opt)
            self._add_scheduler("bn_scheduler", bn_scheduler)

        # Accumulated gradients
        self._accumulated_gradient_step = self.get_from_opt(config, ["training", "optim", "accumulated_gradient"])
        if self._accumulated_gradient_step:
            if self._accumulated_gradient_step > 1:
                self._accumulated_gradient_count = 0
            else:
                raise Exception("When set, accumulated_gradient option should be an integer greater than 1")

        # Gradient clipping
        self._grad_clip = self.get_from_opt(config, ["training", "optim", "grad_clip"], default_value=-1)

    def optimize_parameters(self, epoch, batch_size):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self._num_epochs = epoch
        self._num_batches += 1
        self._num_samples += batch_size

        #debug_memory()
        print(torch.cuda.memory_allocated(0))
        self.d_step(epoch=epoch)
        print(torch.cuda.memory_allocated(0))
        
        self.g_step(epoch=epoch)
        print(torch.cuda.memory_allocated(0))

        # self.forward(epoch=epoch)  # first call forward to calculate intermediate results
        # make_optimizer_step = self._manage_optimizer_zero_grad()  # Accumulate gradient if option is up
        # self.backward()  # calculate gradients

        # if self._grad_clip > 0:
        #     torch.nn.utils.clip_grad_value_(self.parameters(), self._grad_clip)

        # # if make_optimizer_step:
        # #     self._optimizer.step()  # update parameters

        # if self._lr_scheduler:
        #     lr_scheduler_step = self._collect_scheduler_step("_update_lr_scheduler_on")
        #     self._lr_scheduler.step(lr_scheduler_step)

        # if self._bn_scheduler:
        #     bn_scheduler_step = self._collect_scheduler_step("_update_bn_scheduler_on")
        #     self._bn_scheduler.step(bn_scheduler_step)

    def d_step(self, *args, **kwargs):
        generator_input = torch.Tensor(np.random.normal(0, 0.2, (self.batch_size, self.latent_space))).cuda()
        fake_points_all = self.generator(generator_input)
        fake_target = torch.from_numpy(np.zeros(self.batch_size,).astype(np.int64)).float().reshape(self.batch_size, 1).cuda().detach() # should be all 0's since they fake
        real_target = torch.from_numpy(np.ones(self.batch_size,).astype(np.int64)).float().reshape(self.batch_size, 1).cuda().detach() # should be all 0's since they fake

        fake_loss = []
        for i in range(len(self.scales)):
            optimizer = self.d_optimizers[i]
            optimizer.zero_grad()

            real_points = self.input["scale_" + str(self.scales[i])]
            real_points = SimpleBatch.from_data_list(real_points).contiguous()
            fake_points = fake_points_all[i].transpose(2,1).detach()
            fake_points = SimpleBatch(pos=fake_points).contiguous()

            fake_preds = self.discriminators[i](fake_points)
            real_preds = self.discriminators[i](real_points)

            d_real = F.mse_loss(real_preds, real_target)
            d_fake = F.mse_loss(fake_preds, fake_target)
            fake_loss.append(d_fake)
            d_tot = (d_real + d_fake) / 2

            setattr(self, "d_loss_%d" % (self.scales[i]), d_tot)

            d_tot.backward()
            optimizer.step()


    def g_step(self, *args, **kwargs):
        optimizer = self.g_optimizer
        optimizer.zero_grad()

        generator_input = torch.Tensor(np.random.normal(0, 0.2, (self.batch_size, self.latent_space))).cuda()
        gen_points_all = self.generator(generator_input)
        real_target = torch.from_numpy(np.ones(self.batch_size,).astype(np.int64)).float().reshape(self.batch_size, 1).cuda() # should be all 0's since they fake

        with torch.no_grad():
            g_loss = []
            for i in range(len(self.scales)):
                gen_points = gen_points_all[i].transpose(2,1)        
                setattr(self, "data_visual_%d" % (self.scales[i]), Data(pos=gen_points))
                gen_points = SimpleBatch(pos=gen_points).contiguous()

                gen_preds = self.discriminators[i](gen_points)

                g_loss.append(F.mse_loss(gen_preds, real_target))

        point_combos = itertools.combinations(gen_points_all, r=2)
        mu = 0
        cov = 0
        for combo in point_combos:
            m, c = self.get_local_pair(combo[0], combo[1])
            mu += m
            cov += c

        weight = 30
        self.g_similar_loss = weight * 1.0 * mu + \
                            weight * 5.0 * cov
        self.g_discriminator_loss = 1.2 * sum(g_loss)
        self.g_loss = self.g_discriminator_loss + 0.5 * self.g_similar_loss

        self.g_loss.backward()
        optimizer.step()


#     def forward(self, *args, **kwargs):
#         generator_input = torch.Tensor(np.random.normal(0, 0.2, (self.batch_size, self.latent_space))).cuda()
#         fake_points_all = self.generator(generator_input)
#         fake_target = torch.from_numpy(np.zeros(self.batch_size,).astype(np.int64)).float().reshape(self.batch_size, 1).cuda() # should be all 0's since they fake
#         real_target = torch.from_numpy(np.ones(self.batch_size,).astype(np.int64)).float().reshape(self.batch_size, 1).cuda() # should be all 0's since they fake

#         fake_loss = []
#         self.d_losses = []
#         for i in range(len(self.scales)):
#             real_points = self.input["scale_" + str(self.scales[i])]
#             real_points = SimpleBatch.from_data_list(real_points).contiguous()
#             fake_points = fake_points_all[i].transpose(2,1)
#             fake_points = SimpleBatch(pos=fake_points).contiguous()

#             fake_preds = self.discriminators[i](fake_points)
#             real_preds = self.discriminators[i](real_points)

#             print('discriminator scale: %d' % (self.scales[i]))

#             d_real = F.mse_loss(real_preds, real_target)
#             d_fake = F.mse_loss(fake_preds, fake_target)
#             fake_loss.append(d_fake)
#             d_tot = (d_real + d_fake) / 2

#             self.freeze_all_params()
#             unfreeze_params(self.discriminators[i])
#             d_tot.backward()
        
#         self.d_loss = sum(self.d_losses)
            
#         point_combos = itertools.combinations(fake_points, r=2)
#         mu = 0
#         cov = 0
#         for combo in point_combos:
#             m, c = self.get_local_pair(combo[0], combo[1])
#             mu += m
#             cov += c

#         weight = 30
#         self.g_similar_loss = weight * 1.0 * mu + \
#                             weight * 5.0 * cov
#         self.g_real_loss = 1.2 * sum(fake_loss)
#         self.g_loss = self.g_real_loss + 0.5 * self.g_similar_loss

#     def freeze_all_params(self):
#         for i in range(len(self.scales)):
#             freeze_params(self.discriminators[i])
#         freeze_params(self.generator)

#     def backward(self):
#         #self.g_loss.backward()
#         #print('done')
#         for i in range(len(self.scales)):
#             self.freeze_all_params()
#             unfreeze_params(self.discriminators[i])
#             for name, param in self.named_parameters():
#                 print(name + ": " + str(param.requires_grad))

#             loss = self.d_losses[i]
#             print('backwards on %d' % (i))
# #           print(self.d_losses[i])
# #            self.discriminators[i].backward(loss)
#             loss.backward(retain_graph=True)

#         # self.freeze_all_params()
#         # unfreeze_params(self.discriminators[0])
#         # self.d_losses[0].backward(retain_graph=True)

#         self.freeze_all_params()
#         unfreeze_params(self.generator)
#         self.g_loss.backward()