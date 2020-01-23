import torch
import torch.nn as nn
import torchdiffeq
from src.utils.models import Hamiltonian, Christoffel


def torchdiffeq_torch_integrator(GP_model, t_eval, y0, method='rk4', adjoint_boolean=False, create_graph=True,
                                 retain_graph=True):
    assert str.upper(method) in ['EULER', 'MIDPOINT', 'RK4', 'DOPRI5']

    class dummy_integrator(nn.Module):

        def __init__(self, fitted_model, create_graph, retain_graph):
            super(dummy_integrator, self).__init__()
            self.hamiltonian_nn = Hamiltonian(
                Christoffel(fitted_model, xmin=torch.Tensor([1.]), xmax=torch.Tensor([2.])))
            self.retain_graph = retain_graph
            self.create_graph = create_graph

        def forward(self, t, x):
            return self.hamiltonian_nn.get_Rgrad(x, create_graph=self.create_graph, retain_graph=self.retain_graph)

    dummy_nn = dummy_integrator(GP_model, create_graph, retain_graph)
    if adjoint_boolean:
        z_out = torchdiffeq.odeint_adjoint(dummy_nn, y0, t_eval, method=str.lower(method))
    else:
        z_out = torchdiffeq.odeint(dummy_nn, y0, t_eval, method=str.lower(method))
    return z_out