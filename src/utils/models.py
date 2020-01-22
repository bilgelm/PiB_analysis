import gpytorch


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
