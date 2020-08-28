"""
loss on transformations(quaternions, rotations matrix, ...)
"""
import torch
import torch.nn as nn

# matrix loss


class MatrixMSELoss(nn.Module):
    def forward(self, est_T, T_gt):
        """ |g*igt - I| (should be 0) """
        assert est_T.size(0) == T_gt.size(0)
        assert est_T.size(1) == T_gt.size(1) == 4
        assert est_T.size(2) == T_gt.size(2) == 4

        T_gt_inv = torch.inverse(T_gt)
        A = est_T.matmul(T_gt_inv)
        Id = torch.eye(4).to(A).view(1, 4, 4).expand(A.size(0), 4, 4)
        return torch.nn.functional.mse_loss(A, Id, size_average=True)


# feature based loss


class RSQLoss(nn.Module):
    def forward(self, r):
        """
        minimize the norm of ||r||
        """
        z = torch.zeros_like(r)
        return torch.nn.functional.mse_loss(r, z, size_average=False)
