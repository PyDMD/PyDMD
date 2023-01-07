import torch
from torch.autograd import Function
import scipy
import numpy as np


class MatrixSquareRoot(Function):
    """
    Square root matrix.
    """

    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy()
        if m.ndim == 2:
            sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m)).to(input.device)
        elif m.ndim == 3:
            sqrtm = np.stack(tuple(scipy.linalg.sqrtm(mi) for mi in m))
            sqrtm = torch.from_numpy(sqrtm).to(input.device)
        else:
            raise ValueError(f"Unsupported n. of dimensions {m.ndim}")

        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy()
            gm = grad_output.data.cpu().numpy()
            if sqrtm.ndim == 2:
                grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)
                grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
            else:
                grad_sqrtm = np.stack(
                    tuple(
                        scipy.linalg.solve_sylvester(sqrtm_i, sqrtm_i, gm_i)
                        for sqrtm_i, gm_i in zip(sqrtm, gm)
                    )
                )
                grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input
