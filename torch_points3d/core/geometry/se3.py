""" 3-d rigid body transfomation group and corresponding Lie algebra. """
import torch
from .sinc import sinc1, sinc2, sinc3
from . import so3


def twist_prod(x, y):
    x_ = x.view(-1, 6)
    y_ = y.view(-1, 6)

    xw, xv = x_[:, 0:3], x_[:, 3:6]
    yw, yv = y_[:, 0:3], y_[:, 3:6]

    zw = so3.cross_prod(xw, yw)
    zv = so3.cross_prod(xw, yv) + so3.cross_prod(xv, yw)

    z = torch.cat((zw, zv), dim=1)

    return z.view_as(x)


def liebracket(x, y):
    return twist_prod(x, y)


def mat(x):
    # size: [*, 6] -> [*, 4, 4]
    x_ = x.view(-1, 6)
    w1, w2, w3 = x_[:, 0], x_[:, 1], x_[:, 2]
    v1, v2, v3 = x_[:, 3], x_[:, 4], x_[:, 5]
    O = torch.zeros_like(w1)

    X = torch.stack(
        (
            torch.stack((O, -w3, w2, v1), dim=1),
            torch.stack((w3, O, -w1, v2), dim=1),
            torch.stack((-w2, w1, O, v3), dim=1),
            torch.stack((O, O, O, O), dim=1),
        ),
        dim=1,
    )
    return X.view(*(x.size()[0:-1]), 4, 4)


def vec(X):
    X_ = X.view(-1, 4, 4)
    w1, w2, w3 = X_[:, 2, 1], X_[:, 0, 2], X_[:, 1, 0]
    v1, v2, v3 = X_[:, 0, 3], X_[:, 1, 3], X_[:, 2, 3]
    x = torch.stack((w1, w2, w3, v1, v2, v3), dim=1)
    return x.view(*X.size()[0:-2], 6)


def genvec():
    return torch.eye(6)


def genmat():
    return mat(genvec())


def exp(x):
    x_ = x.view(-1, 6)
    w, v = x_[:, 0:3], x_[:, 3:6]
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    W = so3.mat(w)
    S = W.bmm(W)
    I = torch.eye(3).to(w)

    # Rodrigues' rotation formula.
    # R = cos(t)*eye(3) + sinc1(t)*W + sinc2(t)*(w*w');
    #  = eye(3) + sinc1(t)*W + sinc2(t)*S
    R = I + sinc1(t) * W + sinc2(t) * S

    # V = sinc1(t)*eye(3) + sinc2(t)*W + sinc3(t)*(w*w')
    #  = eye(3) + sinc2(t)*W + sinc3(t)*S
    V = I + sinc2(t) * W + sinc3(t) * S

    p = V.bmm(v.contiguous().view(-1, 3, 1))

    z = torch.Tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(x_.size(0), 1, 1).to(x)
    Rp = torch.cat((R, p), dim=2)
    g = torch.cat((Rp, z), dim=1)

    return g.view(*(x.size()[0:-1]), 4, 4)


def inverse(g):
    g_ = g.view(-1, 4, 4)
    R = g_[:, 0:3, 0:3]
    p = g_[:, 0:3, 3]
    Q = R.transpose(1, 2)
    q = -Q.matmul(p.unsqueeze(-1))

    z = torch.Tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(g_.size(0), 1, 1).to(g)
    Qq = torch.cat((Q, q), dim=2)
    ig = torch.cat((Qq, z), dim=1)

    return ig.view(*(g.size()[0:-2]), 4, 4)


def log(g):
    g_ = g.view(-1, 4, 4)
    R = g_[:, 0:3, 0:3]
    p = g_[:, 0:3, 3]

    w = so3.log(R)
    H = so3.inv_vecs_Xg_ig(w)
    v = H.bmm(p.contiguous().view(-1, 3, 1)).view(-1, 3)

    x = torch.cat((w, v), dim=1)
    return x.view(*(g.size()[0:-2]), 6)


def group_prod(g, h):
    # g, h : SE(3)
    g1 = g.matmul(h)
    return g1


class ExpMap(torch.autograd.Function):
    """ Exp: se(3) -> SE(3)
    """

    @staticmethod
    def forward(ctx, x):
        """ Exp: R^6 -> M(4),
            size: [B, 6] -> [B, 4, 4],
              or  [B, 1, 6] -> [B, 1, 4, 4]
        """
        ctx.save_for_backward(x)
        g = exp(x)
        return g

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        g = exp(x)
        gen_k = genmat().to(x)

        # Let z = f(g) = f(exp(x))
        # dz = df/dgij * dgij/dxk * dxk
        #    = df/dgij * (d/dxk)[exp(x)]_ij * dxk
        #    = df/dgij * [gen_k*g]_ij * dxk

        dg = gen_k.matmul(g.view(-1, 1, 4, 4))
        # (k, i, j)
        dg = dg.to(grad_output)

        go = grad_output.contiguous().view(-1, 1, 4, 4)
        dd = go * dg
        grad_input = dd.sum(-1).sum(-1)

        return grad_input


Exp = ExpMap.apply


class SE3Transform(torch.nn.Module):
    def __init__(self, conv_type="DENSE", trans_x=True):
        super(SE3Transform, self).__init__()
        self.conv_type = conv_type
        self.trans_x = trans_x

    @staticmethod
    def batch_transform(trans, xyz, norm=None):
        # trans : SE(3),  * x 4 x 4
        # xyz : R^3,    * x 3[x N]
        trans_ = trans.view(-1, 4, 4)
        R = trans_[:, 0:3, 0:3].contiguous().view(*(trans.size()[0:-2]), 3, 3)
        p = trans_[:, 0:3, 3].contiguous().view(*(trans.size()[0:-2]), 3)
        if len(trans.size()) == len(xyz.size()):
            res = R.matmul(xyz) + p.unsqueeze(-1)
        else:
            res = R.matmul(xyz.unsqueeze(-1)).squeeze(-1) + p
        return res

    @staticmethod
    def partial_transform(trans, xyz, batch, norm=None):
        """
        trans: B x 4 x 4
        xyz : N x 3
        batch : N
        """
        num_batch = batch.max().item() + 1
        assert num_batch == trans.size(0)
        assert xyz.size(0) == batch.size(0)
        assert trans.size(1) == trans.size(2) == 4
        for i in range(num_batch):
            T = trans[i]
            xyz[batch == i] = xyz[batch == i] @ T[:3, :3].T + T[:3, 3]
        return xyz

    @staticmethod
    def multi_partial_transform(trans, xyz, batch, norm=None):

        # trans of size B x M x 4 x 4
        new_trans = trans.view(-1, 4, 4)  # size BM x 4 x 4
        num_batch = batch.max().item() + 1
        num_multi = new_trans.size(0) // num_batch  # M
        # new_size = xyz.size(0) * num_multi # MN

        new_xyz = xyz.unsqueeze(0).expand(num_multi, *xyz.shape).reshape(-1, 3)  # MN
        new_batch = batch.unsqueeze(0).expand(num_multi, *batch.shape).reshape(-1)  # MN
        rang = (torch.arange(num_multi) * num_batch).repeat(xyz.size(0), 1).T.reshape(-1).to(new_batch)
        new_batch = new_batch + rang  # size MN
        new_xyz = SE3Transform.partial_transform(new_trans, new_xyz, new_batch)
        return new_xyz, new_batch

    def forward(self, trans, data):

        if self.conv_type.lower() == "dense":
            data.pos = SE3Transform.batch_transform(trans, data.pos)
            if self.trans_x:
                assert data.x.size(-1) == 3
                data.x = SE3Transform.batch_transform(trans, data.x)
        else:
            assert hasattr(data, "batch")
            data.pos, b = SE3Transform.multi_partial_transform(trans, data.pos, data.batch)
            if self.trans_x:
                if data.x.size(-1) == 3:
                    data.x, _ = SE3Transform.multi_partial_transform(trans, data.pos, data.batch)
            data.batch = b
        return data
