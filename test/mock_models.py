from src.models.base_model import BaseModel


class MockModelConfig(object):
    def __init__(self):
        pass

    def keys(self):
        return []


class MockModel(BaseModel):
    def __init__(self, opt):
        super(MockModel, self).__init__(opt)

    def set_input(self, data):
        pass

    def forward(self):
        pass

    def backward(self):
        pass
