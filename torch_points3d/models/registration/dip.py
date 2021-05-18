import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(3, 256, 1), nn.BatchNorm1d(256), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv1d(512, 1024, 1), nn.BatchNorm1d(1024))

        self.fc1 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU())

        self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU())

        self.fc3 = nn.Sequential(nn.Linear(256, 9))

    def forward(self, x):

        batchsize = x.size()[0]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.Tensor([1, 0, 0, 0, 1, 0, 0, 0, 1]).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden

        x = x.view(-1, 3, 3)

        return x


class PointNetFeature(nn.Module):
    def __init__(self, dim=32, l2norm=True, tnet=True):
        super(PointNetFeature, self).__init__()

        self.l2norm = l2norm
        self.tnet = tnet

        self.stn3d = STN3d()

        self.conv1 = nn.Sequential(nn.Conv1d(3, 256, 1), nn.BatchNorm1d(256), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv1d(512, 1024, 1), nn.BatchNorm1d(1024))

        self.fc1 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU())

        self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.Dropout(p=0.3), nn.BatchNorm1d(256), nn.ReLU())

        self.fc3 = nn.Sequential(nn.Linear(256, dim))

    def _forward(self, x):

        if self.tnet:
            trans = self.stn3d(x)
            xtrans = torch.bmm(trans, x)
        else:
            xtrans = x

        x = self.conv1(xtrans)
        x = self.conv2(x)
        x = self.conv3(x)

        mx, amx = torch.max(x, 2, keepdim=True)
        x = mx.view(-1, 1024)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        if self.l2norm:
            if self.tnet:
                return F.normalize(x, p=2, dim=1), xtrans, trans, mx, amx
            else:
                return F.normalize(x, p=2, dim=1), mx, amx
        else:
            return x, xtrans, trans, mx, amx

    def forward(self, xa, xp=torch.Tensor([]), trans=False):

        if xp.nelement() == 0:
            if trans or not self.tnet:
                out, mx, amx = self._forward(xa)
                return out, mx, amx
            else:
                out, _, _, mx, amx = self._forward(xa)
                return out, mx, amx
        else:
            if self.tnet:
                out1a, out1b, out1c, _, _ = self._forward(xa)
                out2a, out2b, out2c, _, _ = self._forward(xp)
                return out1a, out1b, out1c, out2a, out2b, out2c

            else:
                return self._forward(xa), self._forward(xp)


class DIP:
    def __init__(self):
        pass
