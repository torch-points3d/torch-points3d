from torch_points3d.models.base_model import BaseModel
from torch import nn


class MockModelConfig(object):
    def __init__(self):
        pass

    def keys(self):
        return []


class MockModel(BaseModel):
    def __init__(self, opt):
        super(MockModel, self).__init__(opt)

    def set_input(self, data, device):
        pass

    def forward(self, **kwargs):
        pass

    def backward(self):
        pass


class DifferentiableMockModel(BaseModel):
    def __init__(self, opt):
        super(DifferentiableMockModel, self).__init__(opt)

        self.nn = nn.Linear(3, 3)

    def set_input(self, data, device):
        self.pos = data.pos

    def forward(self, **kwargs):
        self.output = self.nn(self.pos)
        self.loss = self.output.sum()

    def backward(self):
        self.loss.backward()
