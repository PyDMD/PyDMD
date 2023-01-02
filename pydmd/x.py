import logging
logging.getLogger().setLevel(logging.INFO)

from torch.nn import Sequential, ReLU, Linear, Module

import torch
from pydmd import DLDMD, DMD

from data import data_maker_fluid_flow_full

torch.autograd.set_detect_anomaly(True)

def allocate_dldmd(input_size, immersion_size, rweight=1, pweight=1, phweight=1):
    encoder = MLP(input_size, immersion_size)
    decoder = MLP(immersion_size, input_size)
    dmd = DMD(svd_rank=-1, opt=True)
    return DLDMD(encoder=encoder, decoder=decoder, reconstruction_weight=rweight,
                 prediction_weight=pweight, phase_space_weight=phweight, dmd=dmd)


class MLP(Module):
    def __init__(self, input_size, output_size, hidden_layer_size=128):
        super().__init__()
        self.layers = Sequential(
            Linear(input_size, hidden_layer_size),
            ReLU(),
            Linear(hidden_layer_size, hidden_layer_size),
            ReLU(),
            Linear(hidden_layer_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

data = data_maker_fluid_flow_full(x1_lower=-1.1, x1_upper=1.1, x2_lower=-1.1, x2_upper=1.1,
                                  x3_lower=0.0, x3_upper=2, n_ic=50, dt=0.01, tf=6)
data = torch.from_numpy(data)

dldmd = allocate_dldmd(3, 5).double()
dldmd.fit({'training_data': data[:40], 'test_data': data[40:]})