""" inverse matrix """

import torch


def batch_inverse(x):
    """ M(n) -> M(n); x -> x^-1 """
    batch_size, h, w = x.size()
    assert h == w
    y = torch.zeros_like(x)
    for i in range(batch_size):
        y[i, :, :] = x[i, :, :].inverse()
    return y


def batch_inverse_dx(y):
    """ backward """
    # Let y(x) = x^-1.
    # compute dy
    #   dy = dy(j,k)
    #      = - y(j,m) * dx(m,n) * y(n,k)
    #      = - y(j,m) * y(n,k) * dx(m,n)
    # therefore,
    #   dy(j,k)/dx(m,n) = - y(j,m) * y(n,k)
    batch_size, h, w = y.size()
    assert h == w
    # compute dy(j,k,m,n) = dy(j,k)/dx(m,n) = - y(j,m) * y(n,k)
    #   = - (y(j,:))' * y'(k,:)
    yl = y.repeat(1, 1, h).view(batch_size * h * h, h, 1)
    yr = y.transpose(1, 2).repeat(1, h, 1).view(batch_size * h * h, 1, h)
    dy = -yl.bmm(yr).view(batch_size, h, h, h, h)

    # compute dy(m,n,j,k) = dy(j,k)/dx(m,n) = - y(j,m) * y(n,k)
    #   = - (y'(m,:))' * y(n,:)
    # yl = y.transpose(1, 2).repeat(1, 1, h).view(batch_size*h*h, h, 1)
    # yr = y.repeat(1, h, 1).view(batch_size*h*h, 1, h)
    # dy = - yl.bmm(yr).view(batch_size, h, h, h, h)

    return dy


def batch_pinv_dx(x):
    """ returns y = (x'*x)^-1 * x' and dy/dx. """
    # y = (x'*x)^-1 * x'
    #   = s^-1 * x'
    #   = b * x'
    # d{y(j,k)}/d{x(m,n)}
    #   = d{b(j,i) * x(k,i)}/d{x(m,n)}
    #   = d{b(j,i)}/d{x(m,n)} * x(k,i) + b(j,i) * d{x(k,i)}/d{x(m,n)}
    # d{b(j,i)}/d{x(m,n)}
    #   = d{b(j,i)}/d{s(p,q)} * d{s(p,q)}/d{x(m,n)}
    #   = -b(j,p)*b(q,i) * d{s(p,q)}/d{x(m,n)}
    # d{s(p,q)}/d{x(m,n)}
    #   = d{x(t,p)*x(t,q)}/d{x(m,n)}
    #   = d{x(t,p)}/d{x(m,n)} * x(t,q) + x(t,p) * d{x(t,q)}/d{x(m,n)}
    batch_size, h, w = x.size()
    xt = x.transpose(1, 2)
    s = xt.bmm(x)
    b = batch_inverse(s)
    y = b.bmm(xt)

    # dx/dx
    ex = torch.eye(h * w).to(x).unsqueeze(0).view(1, h, w, h, w)
    # ds/dx = dx(t,_)/dx * x(t,_) + x(t,_) * dx(t,_)/dx
    ex1 = ex.view(1, h, w * h * w)  # [t, p*m*n]
    dx1 = x.transpose(1, 2).matmul(ex1).view(batch_size, w, w, h, w)  # [q, p,m,n]
    ds_dx = dx1.transpose(1, 2) + dx1  # [p, q, m, n]
    # db/ds
    db_ds = batch_inverse_dx(b)  # [j, i, p, q]
    # db/dx = db/d{s(p,q)} * d{s(p,q)}/dx
    db1 = db_ds.view(batch_size, w * w, w * w).bmm(ds_dx.view(batch_size, w * w, h * w))
    db_dx = db1.view(batch_size, w, w, h, w)  # [j, i, m, n]
    # dy/dx = db(_,i)/dx * x(_,i) + b(_,i) * dx(_,i)/dx
    dy1 = db_dx.transpose(1, 2).contiguous().view(batch_size, w, w * h * w)
    dy1 = x.matmul(dy1).view(batch_size, h, w, h, w)  # [k, j, m, n]
    ext = ex.transpose(1, 2).contiguous().view(1, w, h * h * w)
    dy2 = b.matmul(ext).view(batch_size, w, h, h, w)  # [j, k, m, n]
    dy_dx = dy1.transpose(1, 2) + dy2

    return y, dy_dx


class InvMatrix(torch.autograd.Function):
    """ M(n) -> M(n); x -> x^-1.
    """

    @staticmethod
    def forward(ctx, x):
        y = batch_inverse(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors  # v0.4
        # y, = ctx.saved_variables  # v0.3.1
        batch_size, h, w = y.size()
        assert h == w

        # Let y(x) = x^-1 and assume any function f(y(x)).
        # compute df/dx(m,n)...
        #   df/dx(m,n) = df/dy(j,k) * dy(j,k)/dx(m,n)
        # well, df/dy is 'grad_output'
        # and so we will return 'grad_input = df/dy(j,k) * dy(j,k)/dx(m,n)'

        dy = batch_inverse_dx(y)  # dy(j,k,m,n) = dy(j,k)/dx(m,n)
        go = grad_output.contiguous().view(batch_size, 1, h * h)  # [1, (j*k)]
        ym = dy.view(batch_size, h * h, h * h)  # [(j*k), (m*n)]
        r = go.bmm(ym)  # [1, (m*n)]
        grad_input = r.view(batch_size, h, h)  # [m, n]

        return grad_input


# TODO test the procedure
if __name__ == "__main__":

    def test():
        x = torch.randn(2, 3, 2)
        x_val = x.requires_grad_()

        s_val = x_val.transpose(1, 2).bmm(x_val)
        s_inv = InvMatrix.apply(s_val)
        y_val = s_inv.bmm(x_val.transpose(1, 2))
        y_val.sum().backward()
        t1 = x_val.grad

        y, dy_dx = batch_pinv_dx(x)
        t2 = dy_dx.sum(1).sum(1)

        print(t1)
        print(t2)
        print(t1 - t2)

    test()

# EOF
