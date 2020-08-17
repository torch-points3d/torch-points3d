
import logging
import torch.nn.functional as F
from typing import Any
import torch
import numpy as np
from torch_points3d.models.generative.gan_base_model import BaseGanModel
from torch_points3d.modules.DGCN.gan_modules import PointGenerator, Discriminator
from torch_points3d.core.losses.chamfer_loss import ChamferLoss
from torch_points3d.core.spatial_ops import KNNNeighbourFinder
import torch_geometric
from torch_points3d.datasets.batch import SimpleBatch
import itertools
import torch.nn as nn
from torch_points3d.core.common_modules.gathering import gather
from torch_geometric.data import DataLoader, InMemoryDataset, extract_zip, Data

log = logging.getLogger(__name__)

class DGCN(BaseGanModel):
    def __init__(self, option, model_type, dataset, modules):
        super(DGCN, self).__init__(option)

        self.generator = PointGenerator(self.scales, self.latent_space, option.generator)
        
        self.discriminators = nn.ModuleList()
        for i in range(len(self.scales)):
            self.discriminators.append(Discriminator(option.discriminator.fc_layers[i], option.discriminator.classifier))

        self.neighbour_finder = KNNNeighbourFinder(k=20)
        self.chamfer_loss = ChamferLoss()

        losses = ["g_similar_loss", "g_loss", "g_discriminator_loss"]
        visuals = []
        for i in range(len(self.scales)):
            losses.append("d_loss_%d" % (self.scales[i]))
            visuals.append("data_visual_%d" % (self.scales[i]))
        self.loss_names = losses

        self.visual_names = visuals


    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        
        self.batch_size = len(data[data.keys[0]])
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
        pt1_batch = torch.from_numpy(pt1_batch).to(self.device).long()
        pt1 = pt1.view(-1, 3)

        pt2_batch = np.indices(pt2.shape[0:2])[0].reshape(-1)
        pt2_batch = torch.from_numpy(pt2_batch).to(self.device).long()
        pt2 = pt2.view(-1, 3)

        idx = self.neighbour_finder(pt1, pt2, pt1_batch, pt2_batch)


        pt1 = gather(pt1, idx)
        pt1 = pt1.reshape(pt1.shape[0], pt2_batch.shape[-1], 3, self.neighbour_finder.k)
        pt1 = pt1.transpose(2, 1)
        return pt1

    def get_local_pair(self, pt1, pt2):
        pt1_batch,pt1_N,pt1_M = pt1.size()
        pt2_batch,pt2_N,pt2_M = pt2.size()
        
        new_xyz = pt1.transpose(1, 2).contiguous()      # Bx3xM -> BxMx3
        pt1_trans = pt1.transpose(1, 2).contiguous()    # Bx3xM -> BxMx3
        pt2_trans = pt2.transpose(1, 2).contiguous()    # Bx3xN -> BxNx3
        
        g_xyz1 = self.get_neighbors(pt1_trans, new_xyz)
        g_xyz2 = self.get_neighbors(pt2_trans, new_xyz)
        
        g_xyz1 = g_xyz1.transpose(1, 2).contiguous().view(-1, 3, 20)    # Bx3xMxK -> BxMx3xK -> (BM)x3xK
        g_xyz2 = g_xyz2.transpose(1, 2).contiguous().view(-1, 3, 20)    # Bx3xMxK -> BxMx3xK -> (BM)x3xK

        mu1, var1 = self.compute_mean_covariance(g_xyz1) 
        mu2, var2 = self.compute_mean_covariance(g_xyz2) 
        
        mu1 = mu1.view(pt1_batch,-1,3)
        mu2 = mu2.view(pt2_batch,-1,3)

        var1 = var1.view(pt1_batch,-1,9)
        var2 = var2.view(pt2_batch,-1,9)

        like_mu12 = self.chamfer_loss(mu1,mu2) / float(pt1_M)
        like_var12 = self.chamfer_loss(var1,var2) / float(pt1_M)
              
        return like_mu12, like_var12

    def d_step(self, *args, **kwargs):
        generator_input = torch.Tensor(np.random.normal(0, 0.2, (self.batch_size, self.latent_space))).to(self.device)
        fake_points_all = self.generator(generator_input)
        fake_target = torch.from_numpy(np.zeros(self.batch_size,).astype(np.int64)).float().reshape(self.batch_size, 1).to(self.device).detach() # should be all 0's since they fake
        real_target = torch.from_numpy(np.ones(self.batch_size,).astype(np.int64)).float().reshape(self.batch_size, 1).to(self.device).detach() # should be all 0's since they fake

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

        generator_input = torch.Tensor(np.random.normal(0, 0.2, (self.batch_size, self.latent_space))).to(self.device)
        gen_points_all = self.generator(generator_input)
        real_target = torch.from_numpy(np.ones(self.batch_size,).astype(np.int64)).float().reshape(self.batch_size, 1).to(self.device) # should be all 0's since they fake

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

    def forward(self, data):
        # forward-only for interface
        generator_input = torch.Tensor(np.random.normal(0, 0.2, (self.batch_size, self.latent_space))).to(self.device)
        gen_points_all = self.generator(generator_input)
        
        for i in range(len(self.scales)):
            gen_points = gen_points_all[i].transpose(2,1)        
            setattr(self, "data_visual_%d" % (self.scales[i]), Data(pos=gen_points))