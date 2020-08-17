import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def get_edge_features_xyz(x, pc, k, num=-1):
    """
    Args:
        x: point features [B, dims, N]
        pc: point cloud [B, dims, M]
        k: kNN neighbours
    Return:
        ([B, 2dims, N, k], [B, 2dims, M, k])
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
        x: point features [B, dims, N]
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