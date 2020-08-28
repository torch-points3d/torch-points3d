""" sinc(t) := sin(t) / t """
import torch
from torch import sin, cos


def sinc1(t):
    """ sinc1: t -> sin(t)/t """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = s == 0
    t2 = t[s] ** 2
    r[s] = 1 - t2 / 6 * (1 - t2 / 20 * (1 - t2 / 42))  # Taylor series O(t^8)
    r[c] = sin(t[c]) / t[c]

    return r


def sinc1_dt(t):
    """ d/dt(sinc1) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = s == 0
    t2 = t ** 2
    r[s] = -t[s] / 3 * (1 - t2[s] / 10 * (1 - t2[s] / 28 * (1 - t2[s] / 54)))  # Taylor series O(t^8)
    r[c] = cos(t[c]) / t[c] - sin(t[c]) / t2[c]

    return r


def sinc1_dt_rt(t):
    """ d/dt(sinc1) / t """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = s == 0
    t2 = t ** 2
    r[s] = -1 / 3 * (1 - t2[s] / 10 * (1 - t2[s] / 28 * (1 - t2[s] / 54)))  # Taylor series O(t^8)
    r[c] = (cos(t[c]) / t[c] - sin(t[c]) / t2[c]) / t[c]

    return r


def rsinc1(t):
    """ rsinc1: t -> t/sinc1(t) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = s == 0
    t2 = t[s] ** 2
    r[s] = (((31 * t2) / 42 + 7) * t2 / 60 + 1) * t2 / 6 + 1  # Taylor series O(t^8)
    r[c] = t[c] / sin(t[c])

    return r


def rsinc1_dt(t):
    """ d/dt(rsinc1) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = s == 0
    t2 = t[s] ** 2
    r[s] = ((((127 * t2) / 30 + 31) * t2 / 28 + 7) * t2 / 30 + 1) * t[s] / 3  # Taylor series O(t^8)
    r[c] = 1 / sin(t[c]) - (t[c] * cos(t[c])) / (sin(t[c]) * sin(t[c]))

    return r


def rsinc1_dt_csc(t):
    """ d/dt(rsinc1) / sin(t) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = s == 0
    t2 = t[s] ** 2
    r[s] = t2 * (t2 * ((4 * t2) / 675 + 2 / 63) + 2 / 15) + 1 / 3  # Taylor series O(t^8)
    r[c] = (1 / sin(t[c]) - (t[c] * cos(t[c])) / (sin(t[c]) * sin(t[c]))) / sin(t[c])

    return r


def sinc2(t):
    """ sinc2: t -> (1 - cos(t)) / (t**2) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = s == 0
    t2 = t ** 2
    r[s] = 1 / 2 * (1 - t2[s] / 12 * (1 - t2[s] / 30 * (1 - t2[s] / 56)))  # Taylor series O(t^8)
    r[c] = (1 - cos(t[c])) / t2[c]

    return r


def sinc2_dt(t):
    """ d/dt(sinc2) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = s == 0
    t2 = t ** 2
    r[s] = -t[s] / 12 * (1 - t2[s] / 5 * (1.0 / 3 - t2[s] / 56 * (1.0 / 2 - t2[s] / 135)))  # Taylor series O(t^8)
    r[c] = sin(t[c]) / t2[c] - 2 * (1 - cos(t[c])) / (t2[c] * t[c])

    return r


def sinc3(t):
    """ sinc3: t -> (t - sin(t)) / (t**3) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = s == 0
    t2 = t[s] ** 2
    r[s] = 1 / 6 * (1 - t2 / 20 * (1 - t2 / 42 * (1 - t2 / 72)))  # Taylor series O(t^8)
    r[c] = (t[c] - sin(t[c])) / (t[c] ** 3)

    return r


def sinc3_dt(t):
    """ d/dt(sinc3) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = s == 0
    t2 = t[s] ** 2
    r[s] = -t[s] / 60 * (1 - t2 / 21 * (1 - t2 / 24 * (1.0 / 2 - t2 / 165)))  # Taylor series O(t^8)
    r[c] = (3 * sin(t[c]) - t[c] * (cos(t[c]) + 2)) / (t[c] ** 4)

    return r


def sinc4(t):
    """ sinc4: t -> 1/t^2 * (1/2 - sinc2(t))
                  = 1/t^2 * (1/2 - (1 - cos(t))/t^2)
    """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = s == 0
    t2 = t ** 2
    r[s] = 1 / 24 * (1 - t2 / 30 * (1 - t2 / 56 * (1 - t2 / 90)))  # Taylor series O(t^8)
    r[c] = (0.5 - (1 - cos(t)) / t2) / t2


class Sinc1_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return sinc1(theta)

    @staticmethod
    def backward(ctx, grad_output):
        (theta,) = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * sinc1_dt(theta).to(grad_output)
        return grad_theta


Sinc1 = Sinc1_autograd.apply


class RSinc1_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return rsinc1(theta)

    @staticmethod
    def backward(ctx, grad_output):
        (theta,) = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * rsinc1_dt(theta).to(grad_output)
        return grad_theta


RSinc1 = RSinc1_autograd.apply


class Sinc2_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return sinc2(theta)

    @staticmethod
    def backward(ctx, grad_output):
        (theta,) = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * sinc2_dt(theta).to(grad_output)
        return grad_theta


Sinc2 = Sinc2_autograd.apply


class Sinc3_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return sinc3(theta)

    @staticmethod
    def backward(ctx, grad_output):
        (theta,) = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * sinc3_dt(theta).to(grad_output)
        return grad_theta


Sinc3 = Sinc3_autograd.apply


# EOF
