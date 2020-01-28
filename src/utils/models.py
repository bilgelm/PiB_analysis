import torch
import gpytorch
import torch.nn as nn


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Mean : Constant
    Kernel : RBF | Cosine | Polynomial | Linear | Matern
    Likelihood : Gaussian
    """
    def __init__(self, kernel_name, train_x, train_y, likelihood, **kwargs):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        assert kernel_name in ['RBF', 'linear', 'polynomial', 'cosine', 'Matern']
        self.name = kernel_name
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel_name == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(**kwargs))
        elif kernel_name == 'Matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(**kwargs))
        elif kernel_name == 'cosine':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel(**kwargs))
        elif kernel_name == 'linear':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(**kwargs))
        elif kernel_name == 'polynomial':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(**kwargs))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class Christoffel(nn.Module):

    def __init__(self, fitted_model, xmin, xmax):
        super(Christoffel, self).__init__()
        self.model_GP = fitted_model
        self.xref = xmin.clone().detach()
        self.norm = 1.
        self.xmin = xmin.view(-1, 1).clone().detach()
        self.xmax = xmax.view(-1, 1).clone().detach()
        self.normalization_call()

    def forward(self, data):
        """
        Christoffel symbol computation for given data
        """
        return self.model_GP(data).mean

    def integrate(self, x, precision=100):
        """
        Returns normalized inverse metric
        """
        assert len(x.size()) == 2, "must be of shape (n, 1)"
        x = x.repeat(1, precision)
        n = len(x)
        x0 = self.xref.repeat(n, precision)
        line = torch.linspace(0., 1., precision).view(1, -1).repeat(n, 1)
        x_evals = ((x - x0) * line + x0).float()
        if not x_evals.requires_grad:
            x_evals.requires_grad = True
        evals = self.forward(x_evals.reshape(-1, 1)).reshape(n, precision)
        trapezoidal_integration = 1. / (2 * precision) * torch.sum(evals[:, 1:] + evals[:, :-1], dim=-1)
        integrated_x = torch.exp(- 2. * trapezoidal_integration).view(-1, 1)
        rescaled_integrated_x = self.mref * integrated_x
        return rescaled_integrated_x

    def normalization_call(self):
        precision_1 = 100
        precision_2 = 25
        line_1 = (self.xmin + (self.xmax - self.xmin) * torch.linspace(0., 1., precision_1)).view(-1, 1).repeat(1,
                                                                                                                precision_2)
        grid = (self.xref + (line_1 - self.xref) * torch.linspace(0., 1., precision_2).view(1, -1).repeat(precision_1,
                                                                                                          1))
        evals = self.forward(grid.reshape(-1, 1)).reshape(precision_1, precision_2)
        trapezoidal_integration = 1. / (2 * precision_2) * torch.sum(evals[:, 1:] + evals[:, :-1], dim=-1)
        integrated_out = torch.exp(- 2. * trapezoidal_integration)
        integral = 1. / (2 * precision_1) * torch.sum(integrated_out[1:] + integrated_out[:-1])
        self.mref = (self.norm / integral).view(-1, 1)


class Hamiltonian(object):
    """
    Hamiltonian class allowing for autodiff computation of geodesics and field observation from Christoffel symbol
    /!\ Implementation restricted to dimension 1 and BATCH, not extensible to higher dim as is
    """

    def __init__(self, christoffel):
        super(Hamiltonian, self).__init__()
        self.christoffel = christoffel
        self.dimension = 1
        self.J = torch.cat((torch.cat((torch.zeros(self.dimension, self.dimension),
                                       torch.eye(self.dimension)), dim=1),
                            torch.cat((torch.diag(torch.Tensor([-1] * self.dimension)),
                                       torch.zeros(self.dimension, self.dimension)), dim=1)), dim=0)

    def inverse_metric(self, data, precision=100):
        return self.christoffel.integrate(data, precision)

    def metric(self, q, precision=100):
        return 1. / self.inverse_metric(q, precision)

    def velocity_to_momenta(self, q, v):
        return self.metric(q) * v

    def momenta_to_velocity(self, q, p):
        return self.inverse_metric(q) * p

    def dHdq(self, q, p):
        return 2 * p ** 2 * self.inverse_metric(q) * self.christoffel(q).view(-1, 1)

    def dHdp(self, q, p):
        return 2 * p * self.inverse_metric(q)

    def hamiltonian(self, z):
        """
        Hamiltonian from inverse metric computations : H(q,p) = .5 * <p|Q^{-1}p>
        """
        bts = z.size(0)
        q, p = torch.split(z, split_size_or_sections=z.shape[1] // 2, dim=1)
        Q_batched = self.inverse_metric(q)
        Q_batched = Q_batched.unsqueeze(-1) if len(Q_batched.size()) < 3 else Q_batched
        return .5 * torch.bmm(p.unsqueeze(1), torch.bmm(Q_batched, p.unsqueeze(-1))).squeeze(-1)

    def get_grad(self, z, create_graph=False, retain_graph=False, allow_unused=False):
        bts, dim = z.size()
        H = self.hamiltonian(z)
        dH = torch.autograd.grad(outputs=H, inputs=z, grad_outputs=torch.ones((bts, 1)),
                                 create_graph=create_graph, retain_graph=retain_graph,
                                 allow_unused=allow_unused)[0]
        return dH

    def get_Rgrad(self, z, create_graph=False, retain_graph=False):
        """
        Returns rotated Hamiltonian, driving geodesic equations
        """
        return self.get_grad(z, create_graph, retain_graph) @ self.J.t()

    def require_grad_field(self, status):
        for e in self.christoffel.parameters():
            e.requires_grad = status









class MetricFromChristoffel(nn.Module):
    """
    Specific class of inverse metric : requires integration steps
    """

    def __init__(self, fitted_model, xmin, xmax, precision=100, norm=1.):
        nn.Module.__init__(self)
        self.model_GP = fitted_model
        self.xref = xmin.clone().detach()
        self.xmin = xmin.view(-1, 1).clone().detach()
        self.xmax = xmax.view(-1, 1).clone().detach()
        self.norm = norm
        self.precision = precision
        self.normalization_call()

    def set_precision(self, precision):
        self.precision = precision

    def Gamma(self, data):
        """
        Christoffel symbol computation for given data
        """
        return self.model_GP(data).mean

    def forward(self, x):
        """
        Returns normalized inverse metric
        """
        assert len(x.size()) == 2, "must be of shape (n, 1)"
        x = x.repeat(1, self.precision)
        n = len(x)
        x0 = self.xref.repeat(n, self.precision)
        line = torch.linspace(0., 1., self.precision).view(1, -1).repeat(n, 1)
        x_evals = ((x - x0) * line + x0).float()
        if not x_evals.requires_grad:
            x_evals.requires_grad = True
        evals = self.Gamma(x_evals.reshape(-1, 1)).reshape(n, self.precision)
        trapezoidal_integration = 1. / (2 * self.precision) * torch.sum(evals[:, 1:] + evals[:, :-1], dim=-1)
        integrated_x = torch.exp(- 2. * trapezoidal_integration).view(-1, 1)
        rescaled_integrated_x = self.mref * integrated_x
        return rescaled_integrated_x

    def normalization_call(self):
        precision_1 = 100
        precision_2 = 25
        line_1 = (self.xmin + (self.xmax - self.xmin) * torch.linspace(0., 1., precision_1)).view(-1, 1).repeat(1,
                                                                                                                precision_2)
        grid = (self.xref + (line_1 - self.xref) * torch.linspace(0., 1., precision_2).view(1, -1).repeat(precision_1,
                                                                                                          1))
        evals = self.Gamma(grid.reshape(-1, 1)).reshape(precision_1, precision_2)
        trapezoidal_integration = 1. / (2 * precision_2) * torch.sum(evals[:, 1:] + evals[:, :-1], dim=-1)
        integrated_out = torch.exp(- 2. * trapezoidal_integration)
        integral = 1. / (2 * precision_1) * torch.sum(integrated_out[1:] + integrated_out[:-1])
        self.mref = (self.norm / integral).view(-1, 1)


class MetricHamiltonian(nn.Module):

    def __init__(self, dimension, nn_inverse_metric, create_graph, retain_graph):
        nn.Module.__init__(self)
        self.dimension = dimension
        self.nn_inverse_metric = nn_inverse_metric
        self.create_graph = create_graph
        self.retain_graph = retain_graph
        self.J = torch.cat((torch.cat((torch.zeros(self.dimension, self.dimension),
                                       torch.eye(self.dimension)), dim=1),
                            torch.cat((torch.diag(torch.Tensor([-1] * self.dimension)),
                                       torch.zeros(self.dimension, self.dimension)), dim=1)), dim=0)

    def set_create_graph(self, boolean):
        self.create_graph = boolean

    def set_retain_graph(self, boolean):
        self.retain_graph = boolean

    def inverse_metric(self, q):
        return self.nn_inverse_metric(q)

    def metric(self, q):
        """
        ONLY WORKS IN DIM = 1
        ELSE : torch.inverse
        """
        return 1. / self.inverse_metric(q)

    def forward(self, z):
        q, p = torch.split(z, split_size_or_sections=self.dimension, dim=1)
        return .5 * p @ self.inverse_metric(q) @ p.t()

    def momenta_to_velocity(self, q, p):
        return self.inverse_metric(q) * p

    def velocity_to_momenta(self, q, v):
        return self.metric(q) * v

    def get_grad(self, z):
        H = self.forward(z)
        dH = torch.autograd.grad(H, z, create_graph=self.create_graph, retain_graph=self.retain_graph)[0]
        return dH

    def get_Rgrad(self, z):
        H = self.forward(z)
        dH = torch.autograd.grad(H, z, create_graph=self.create_graph, retain_graph=self.retain_graph)[0]
        return dH @ self.J.t()

    def require_grad_field(self, status):
        for e in self.nn_hamiltonian.parameters():
            e.requires_grad = status


class MetricGradHamiltonian(nn.Module):

    def __init__(self, dimension, nn_inverse_metric, create_graph, retain_graph):
        super(MetricGradHamiltonian, self).__init__()
        self.dimension = dimension
        self.nn_inverse_metric = nn_inverse_metric
        self.create_graph = create_graph
        self.retain_graph = retain_graph
        self.J = torch.cat((torch.cat((torch.zeros(self.dimension, self.dimension),
                                       torch.eye(self.dimension)), dim=1),
                            torch.cat((torch.diag(torch.Tensor([-1] * self.dimension)),
                                       torch.zeros(self.dimension, self.dimension)), dim=1)), dim=0)

    def set_create_graph(self, boolean):
        self.create_graph = boolean

    def set_retain_graph(self, boolean):
        self.retain_graph = boolean

    def inverse_metric(self, q):
        return self.nn_inverse_metric(q)

    def metric(self, q):
        """
        ONLY WORKS IN DIM = 1
        ELSE : torch.inverse
        """
        return 1. / self.inverse_metric(q)

    def hamiltonian(self, z):
        q, p = torch.split(z, split_size_or_sections=self.dimension, dim=1)
        return .5 * p @ self.inverse_metric(q) @ p.t()

    def forward(self, t, z):
        return self.get_Rgrad(z)

    def momenta_to_velocity(self, q, p):
        return self.inverse_metric(q) * p

    def velocity_to_momenta(self, q, v):
        return self.metric(q) * v

    def get_grad(self, z):
        H = self.hamiltonian(z)
        dH = torch.autograd.grad(H, z, create_graph=self.create_graph, retain_graph=self.retain_graph)[0]
        return dH

    def get_Rgrad(self, z):
        H = self.hamiltonian(z)
        dH = torch.autograd.grad(H, z, create_graph=self.create_graph, retain_graph=self.retain_graph)[0]
        return dH @ self.J.t()

    def require_grad_field(self, status):
        for e in self.nn_hamiltonian.parameters():
            e.requires_grad = status
