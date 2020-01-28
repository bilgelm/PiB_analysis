import torch
import torch.nn as nn
import torchdiffeq
from scipy.integrate import solve_ivp
from src.utils.op import gpu_numpy_detach
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
    dummy_nn.eval()
    if adjoint_boolean:
        z_out = torchdiffeq.odeint_adjoint(dummy_nn, y0, t_eval, method=str.lower(method))
    else:
        z_out = torchdiffeq.odeint(dummy_nn, y0, t_eval, method=str.lower(method))
    return z_out


def numpy_ivp_integrate_model(model, t_eval, y0):
    """
    Integration with ivp solver :
    * builds on Torch derivation dH with numpy conversion
    * no batch
    """

    def model_numpy_integrator(t, np_x):
        x = torch.from_numpy(np_x).float()
        x = x.view(1, np.size(np_x))
        x.requires_grad = True
        dx = gpu_numpy_detach(model.get_Rgrad(x)).reshape(-1)
        return dx
    t_eval = gpu_numpy_detach(t_eval)
    y0 = gpu_numpy_detach(y0)
    sequence = solve_ivp(fun=model_numpy_integrator, t_span=(t_eval[0], t_eval[-1]), y0=y0, t_eval=t_eval, rtol=1e-10)
    return np.split(sequence['y'], 2, axis=0)


def ivp_integrate_GP(model, t_eval, y0):

    def modelGP_numpy_integrator(t, np_x):
        x = torch.from_numpy(np_x).float()
        x = x.view(1, np.size(np_x))  # batch size of 1
        x.requires_grad = True
        return gpu_numpy_detach(model(x).mean)

    sequence = solve_ivp(fun=modelGP_numpy_integrator, t_span=(t_eval[0], t_eval[-1]), y0=y0, t_eval=t_eval, rtol=1e-10)
    return sequence['y']