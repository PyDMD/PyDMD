import logging

import numpy as np
import scipy
import torch
from torch.autograd import Function

logging.basicConfig(
    format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
)


class MatrixSquareRoot(Function):
    """
    Square root matrix.
    """

    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy()
        if m.ndim == 2:
            m = m[None]
        elif m.ndim != 3:
            raise ValueError(f"Unsupported number of axes: {m.ndim}")

        sqrtm = np.stack(tuple(scipy.linalg.sqrtm(mi) for mi in m))
        if hasattr(np, "complex256") and sqrtm.dtype == np.complex256:
            sqrtm = sqrtm.astype(np.complex128)
            msg = "Casting atilde from np.complex256 to np.complex128"
            logging.info(msg)

        sqrtm = torch.from_numpy(sqrtm).to(input.device)
        if input.ndim == 2:
            sqrtm = sqrtm[0]

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
