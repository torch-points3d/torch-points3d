"""
implementation of pointnet LK
taken from https://github.com/vinits5/learning3d
taken from https://github.com/hmgoforth/PointNetLK
"""
import torch


from torch_points3d.models.registration.base import End2EndBasedModel
import torch_points3d.core.geometry.se3 as se3
import torch_points3d.core.geometry.invmat as invmat
from torch_points3d.applications import models
from torch_points3d.utils.registration import estimate_transfo

from torch_points3d.core.losses import MatrixMSELoss, RSQLoss
from torch_points3d.core.geometry.se3 import SE3Transform


class PointnetLK(End2EndBasedModel):
    """
    class of PointnetLK to estimate transformation using Pointnet.
    """

    def __init__(self, option, model_type, dataset, modules):
        End2EndBasedModel.__init__(self, option)
        self.inverse = invmat.InvMatrix.apply
        self.exp = se3.Exp  # [B, 6] -> [B, 4, 4]
        self.transform = SE3Transform(conv_type=option.conv_type, trans_x=option.trans_options.trans_x)

        backbone_option = option.backbone
        backbone_cls = getattr(models, backbone_option.model_type)
        # self.encoder = ?  # TODO define abstraction
        # in case we update x by position every time
        self.encoder = backbone_cls(
            architecture="encoder", input_nc=dataset.feature_dimension, config=backbone_option.config
        )
        # losses
        self.lambda_T = option.loss_options.lambda_T
        self.lambda_r = option.loss_options.lambda_r
        self.use_prev_r = option.loss_options.use_prev_r
        self.trans_loss_fn = MatrixMSELoss()
        self.feat_loss_fn = RSQLoss()

        w1, w2, w3, v1, v2, v3 = [option.delta for _ in range(6)]
        twist = torch.Tensor([w1, w2, w3, v1, v2, v3])
        self.dt = torch.nn.Parameter(twist.view(1, 6), requires_grad=option.learn_delta)

        # results
        self.last_err = None
        self.g_series = None  # for debug purpose
        self.g = None  # estimation result
        self.itr = 0
        self.xtol = option.xtol

        self.max_iter = option.max_iter

    def set_input(self, data, device):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        """
        assert hasattr(data, "pos_target")

        inp, inp_target = data.to_data()
        self.input = inp.to(device)
        self.input_target = inp_target.to(device)
        if hasattr(data, "pair_ind"):
            self.match = data.pair_ind.to(torch.long)
            self.size_match = data.size_pair_ind.to(torch.long)
            self.set_transfo_gt()
            self.match = self.match.to(device)
            self.size_match = self.size_match.to(device)
            self.trans_gt = self.trans_gt.to(device)
        else:
            self.trans_gt = None

    def compute_loss(self, result):

        self.loss_T = self.trans_loss_fn(result["est_T"], self.trans_gt)
        if not self.use_prev_r:
            self.loss_r = self.feat_loss_fn(result["r"])
        else:
            pr = result["prev_r"]
            if pr is not None:
                self.loss_r = self.feat_loss_fn(result["r"] - pr)
            else:
                self.loss_r = self.feat_loss_fn(result["r"])
        self.loss = self.lambda_T * self.loss_T + self.lambda_r * self.loss_r

    def forward(self, *args, **kwargs):

        result = self.iclk()
        self.output = result["est_T"]

        if self.trans_gt is not None:
            self.compute_loss(result)

    def iclk(self):
        """ initialize
         approximate the jacobian
         compute the inverse jacobian
        iteration"""

        # Initialize

        batch_size = self.get_batch_size()
        est_T0 = torch.eye(4).to(self.input.pos).view(1, 4, 4).expand(batch_size, 4, 4).contiguous()
        est_T = est_T0
        est_T_series = torch.zeros(self.max_iter + 1, *est_T0.size(), dtype=est_T0.dtype)
        est_T_series[0] = est_T0
        data_s = self.input.clone()

        data_t = self.transform(est_T, self.input_target.clone()).contiguous()
        feat_t = self.encoder.forward(data_t)
        dt = self.dt.to(data_t.pos).expand(batch_size, 6)
        J = self.approx_Jic(self.input_target, feat_t.x.view(batch_size, -1), dt)
        pinv = self.compute_inverse_jacobian(J, feat_t.x.view(batch_size, -1), data_s.pos)
        prev_r = None

        for itr in range(self.max_iter):
            # Dense [B, 4, 4] x [B, N, 3] -> [B, N, 3]
            # Other [B, 4, 4] x [N, 3] -> [N, 3]

            data_s = self.transform(est_T, data_s).contiguous()

            feat_s = self.encoder.forward(data_s)
            r = (feat_s.x - feat_t.x).view(batch_size, -1)

            prev_r = r
            pose = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)
            check = pose.norm(p=2, dim=1, keepdim=True).max()
            if float(check) < self.xtol:
                if itr == 0:
                    self.last_err = 0  # no update.
                break
            est_T = self.update(est_T, pose)
            est_T_series[itr + 1] = est_T.clone()

        return {"est_T": est_T, "r": r, "prev_r": prev_r, "est_T_series": est_T_series}

    def compute_inverse_jacobian(self, J, target_features, data_source):
        # compute pinv(J) to solve J*x = -r
        try:
            Jt = J.transpose(1, 2)  # [B, 6, K]
            H = Jt.bmm(J)  # [B, 6, 6]
            B = self.inverse(H)
            pinv = B.bmm(Jt)  # [B, 6, K]
            return pinv
        except RuntimeError as err:
            # singular...?
            self.last_err = err
            # torch.eye(4).to(p0).view(1, 4, 4).expand(p0.size(0), 4, 4).contiguous()
            # Perhaps we can use MP-inverse, but,...
            # probably, self.dt is way too small...
            # source_features = self.encoder(data_source)  # [B, N, 3] -> [B, K]
            # source_features - target_features
            # self.encoder.train(training)
            return {}

    def approx_Jic(self, data_target, target_features, dt):
        # p0: [B, N, 3], Variable
        # f0: [B, K], corresponding feature vector
        # dt: [B, 6], Variable
        # Jk = (feature_model(p(-delta[k], p0)) - f0) / delta[k]
        batch_size = self.get_batch_size()
        # compute transforms
        transf = torch.zeros(6, batch_size, 4, 4).to(data_target.pos)
        for b in range(batch_size):
            d = torch.diag(dt[b, :])  # [6, 6]
            D = self.exp(-d)  # [6, 4, 4]
            transf[:, b, :, :] = D[:, :, :]
        # WARNING change the size of data_target
        data_p = self.transform(transf, data_target).contiguous()  # x [B, 1, N, 3] -> [B, 6, N, 3]
        # f0 = self.feature_model(p0).unsqueeze(-1) # [B, K, 1]
        target_features = target_features.unsqueeze(-1)  # [B, K, 1]
        f = self.encoder.forward(data_p).x.view(6, batch_size, -1).transpose(0, 2).transpose(0, 1)
        # [B, K, 6]
        df = target_features - f  # [B, K, 6]
        J = df / dt.unsqueeze(1)
        return J

    def update(self, g, dx):
        dg = self.exp(dx)  # [B, 6] - > [B, 4, 4]
        return dg.matmul(g)
