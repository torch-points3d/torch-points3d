import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from pdb import set_trace as brk

def get_edge_features_xyz(x, pc, k, num=-1):
    """
    Args:
        x: point cloud [B, dims, N]
        k: kNN neighbours
    Return:
        [B, 2dims, N, k]
        idx
    """
    B, dims, N = x.shape

    # ----------------------------------------------------------------
    # batched pair-wise distance in feature space maybe is can be changed to coordinate space
    # ----------------------------------------------------------------
    xt = x.permute(0, 2, 1)
    xi = -2 * torch.bmm(xt, x)
    xs = torch.sum(xt**2, dim=2, keepdim=True)
    xst = xs.permute(0, 2, 1)
    dist = xi + xs + xst # [B, N, N]

    # get k NN id    
    _, idx_o = torch.sort(dist, dim=2)
    idx = idx_o[: ,: ,1:k+1] # [B, N, k]
    idx = idx.contiguous().view(B, N*k)

    
    # gather
    neighbors = []
    xyz =[]
    for b in range(B):
        tmp = torch.index_select(x[b], 1, idx[b]) # [d, N*k] <- [d, N], 0, [N*k]
        tmp = tmp.view(dims, N, k)
        neighbors.append(tmp)

        tp = torch.index_select(pc[b], 1, idx[b])
        tp = tp.view(3, N, k)
        xyz.append(tp)

    neighbors = torch.stack(neighbors)  # [B, d, N, k]
    xyz = torch.stack(xyz)              # [B, 3, N, k]
    
    # centralize
    central = x.unsqueeze(3).repeat(1, 1, 1, k)         # [B, d, N, 1] -> [B, d, N, k]
    central_xyz = pc.unsqueeze(3).repeat(1, 1, 1, k)    # [B, 3, N, 1] -> [B, 3, N, k]
    
    e_fea = torch.cat([central, neighbors-central], dim=1)
    e_xyz = torch.cat([central_xyz, xyz-central_xyz], dim=1)
    
    assert e_fea.size() == (B, 2*dims, N, k) and e_xyz.size() == (B, 2*3, N, k)
    return e_fea, e_xyz
    
def get_edge_features(x, k, num=-1):
    """
    Args:
        x: point cloud [B, dims, N]
        k: kNN neighbours
    Return:
        [B, 2dims, N, k]    
    """
    B, dims, N = x.shape

    # batched pair-wise distance
    xt = x.permute(0, 2, 1)
    xi = -2 * torch.bmm(xt, x)
    xs = torch.sum(xt**2, dim=2, keepdim=True)
    xst = xs.permute(0, 2, 1)
    dist = xi + xs + xst # [B, N, N]

    # get k NN id    
    _, idx_o = torch.sort(dist, dim=2)
    idx = idx_o[: ,: ,1:k+1] # [B, N, k]
    idx = idx.contiguous().view(B, N*k)


    # gather
    neighbors = []
    for b in range(B):
        tmp = torch.index_select(x[b], 1, idx[b]) # [d, N*k] <- [d, N], 0, [N*k]
        tmp = tmp.view(dims, N, k)
        neighbors.append(tmp)

    neighbors = torch.stack(neighbors) # [B, d, N, k]

    # centralize
    central = x.unsqueeze(3) # [B, d, N, 1]
    central = central.repeat(1, 1, 1, k) # [B, d, N, k]

    ee = torch.cat([central, neighbors-central], dim=1)
    assert ee.shape == (B, 2*dims, N, k)
    return ee

class conv2dbr(nn.Module):
    """ Conv2d-bn-relu
    [B, Fin, H, W] -> [B, Fout, H, W]
    """
    def __init__(self, Fin, Fout, kernel_size, stride=1):
        super(conv2dbr, self).__init__()
        self.conv = nn.Conv2d(Fin, Fout, kernel_size, stride)
        self.bn = nn.BatchNorm2d(Fout)
        self.ac = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x) # [B, Fout, H, W]
        x = self.bn(x)
        x = self.ac(x)
        return x


class ChamferLoss(nn.Module):

	def __init__(self):
		super(ChamferLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()        

	def forward(self,preds,gts):
		P = self.batch_pairwise_dist(gts, preds)
		mins, _ = torch.min(P, 1)
		loss_1 = torch.sum(mins)
		mins, _ = torch.min(P, 2)
		loss_2 = torch.sum(mins)

		return loss_1 + loss_2


	def batch_pairwise_dist(self,x,y):
		bs, num_points_x, points_dim = x.size()
		_, num_points_y, _ = y.size()
		xx = torch.bmm(x, x.transpose(2,1))
		yy = torch.bmm(y, y.transpose(2,1))
		zz = torch.bmm(x, y.transpose(2,1))
		if self.use_cuda:
			dtype = torch.cuda.LongTensor
		else:
			dtype = torch.LongTensor
		diag_ind_x = torch.arange(0, num_points_x).type(dtype)
		diag_ind_y = torch.arange(0, num_points_y).type(dtype)
		#brk()
		rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2,1))
		ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
		P = (rx.transpose(2,1) + ry - 2*zz)
		return P