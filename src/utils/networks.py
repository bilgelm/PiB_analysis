import torch
import torch.nn as nn


class MLP_potential(nn.Module):

    def __init__(self, input_dim, intermediate_dim, output_dim, nb_layers, nonlinearity='tanh', lastnonlinearity=None,
                 dropout=0., must_be_positive=True):
        super(MLP_potential, self).__init__()

        # Params
        self.input_dim = input_dim
        self.d = intermediate_dim
        self.output_dim = output_dim
        self.n = nb_layers
        assert self.n >= 1
        self.dropout = dropout
        self.must_be_positive = must_be_positive
        self.minimal_epsilon = float(1e-6)
        # Network
        modules = []
        if self.n == 1:
            modules.append(Linear_nonlin(self.input_dim, self.output_dim, lastnonlinearity))
        elif self.n == 2:
            modules.append(Linear_nonlin(self.input_dim, self.d, nonlinearity))
            modules.append(nn.Dropout(self.dropout))
            modules.append(Linear_nonlin(self.d, self.output_dim, lastnonlinearity))
        else:
            modules.append(Linear_nonlin(self.input_dim, self.d, nonlinearity))
            for i in range(self.n-2):
                modules.append(Linear_nonlin(self.d, self.d, nonlinearity))
                modules.append(nn.Dropout(self.dropout))
            modules.append(Linear_nonlin(self.d, self.output_dim, lastnonlinearity))
        self.net = nn.Sequential(*modules)
        print('>> MLP potential function has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, z):
        net = self.net(z)
        if self.must_be_positive:
            net = torch.clamp(self.a, min=self.minimal_epsilon)
        return net

    def norm(self):
        norm = []
        for name, param in self.named_parameters():
            if 'bias' not in name:
                norm.append(param.norm(p='fro', dim=None, keepdim=False).view(1))
        return torch.mean(torch.cat(norm))


class Linear_nonlin(nn.Module):
    def __init__(self, in_ch, out_ch, nonlinearity=None, bias=True):
        nn.Module.__init__(self)
        assert nonlinearity in ['cos', 'tanh', 'softplus', 'lrelu', 'sigmoid', None]
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.linear = nn.Linear(in_ch, out_ch, bias=bias)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        x = self.linear(x.view(-1, self.in_ch))
        if not self.nonlinearity:
            pass
        elif self.nonlinearity == 'cos':
            x = torch.cos(x)
        elif self.nonlinearity == 'tanh':
            x = nn.Tanh()(x)
        elif self.nonlinearity == 'softplus':
            x = nn.Softplus()(x)
        elif self.nonlinearity == 'lrelu':
            x = nn.LeakyReLU()(x)
        elif self.nonlinearity == 'sigmoid':
            x = nn.Sigmoid()(x)
        return x.view(-1, self.out_ch)


class euclidean_nn(nn.Module):

    def __init__(self, dimension):
        super(euclidean_nn, self).__init__()
        self.dimension = dimension
        self.Q = nn.Parameter(torch.rand(self.dimension, self.dimension).float())

    def forward(self, z):
        return self.Q.t() @ self.Q


class logistic_nn(nn.Module):

    def __init__(self, dimension, centers, scalings):
        super(logistic_nn, self).__init__()
        self.dimension = dimension
        self.centers = nn.Parameter(torch.from_numpy(centers).float())
        self.scalings = nn.Parameter(torch.from_numpy(scalings).float())

    def forward(self, z):
        z_scaled = (z - self.centers) * self.scalings
        z_ = torch.pow(z_scaled, 2) * (1. - torch.pow(z_scaled, 2))
        return z_ * torch.eye(self.dimension, dtype=torch.float32)






