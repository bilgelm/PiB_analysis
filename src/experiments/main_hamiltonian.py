import os
import sys

# Setting paths to directory roots | >> PiB_analysis directory
parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(parent)
sys.path.insert(0, parent)
os.chdir(parent)
print('Setting root path to : {}'.format(parent))

# Generic
import math
import argparse
import datetime
import json
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.integrate import solve_ivp
from scipy.stats import ranksums

# Torch
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

# utils
import gpytorch
import torchdiffeq
from src.utils.op import gpu_numpy_detach
from src.utils.datasets import get_fromdataframe
from src.utils.models import ExactGPModel, Christoffel, Hamiltonian
from src.utils.integrators import torchdiffeq_torch_integrator


parser = argparse.ArgumentParser(description='Hamiltonian fit with GP on PiB data.')
# action parameters
parser.add_argument('--output_dir', type=str, default='./results/hamiltonian', help='Output directory.')
parser.add_argument('--data_dir', type=str, default='./', help='Data directory.')
parser.add_argument('--cuda', action='store_true', help='Whether CUDA is available on GPUs.')
parser.add_argument('--num_gpu', type=int, default=0, help='Which GPU to run on.')
parser.add_argument('--num_threads', type=int, default=36, help='Number of threads to use if cuda not available')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
# dataset parameters
parser.add_argument('--min_visits', type=int, default=3, help='Minimal number of visits.')
parser.add_argument('--preprocessing', action='store_true', help='Standardization of not.')
parser.add_argument('--noise_std', type=float, default=.07, help='Std of additive noise on observations.')
parser.add_argument('--filter_quantiles', action='store_true', help='Filter ratio data of not.')
parser.add_argument('--eps_quantiles', type=float, default=.05, help='Fraction of left out quantiles.')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
# model parameters
parser.add_argument('--model_type', type=str, default='POLY', choices=['RBF', 'LINEAR', 'POLY'], help='GP model type.')
parser.add_argument('--tuning', type=int, default=50, help='Tuning of GP hyperparameters.')
parser.add_argument('--pib_threshold', type=float, default=1.2, help='Onset for PiB positivity (data dependant).')
# optimization parameters
parser.add_argument('--method', type=str, default='midpoint', choices=['midpoint', 'rk4'], help='Integration scheme.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
parser.add_argument('--lr', type=float, default=.1, help='initial learning rate.')
parser.add_argument('--lr_min', type=float, default=1e-5, help='Minimal learning rate.')
parser.add_argument('--lr_decay', type=float, default=.9, help='Learning rate decay.')
parser.add_argument('--patience', type=int, default=5, help='Patience before scheduling learning rate decay.')


args = parser.parse_args()

# CPU/GPU settings || random seeds
args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    print('>> GPU available.')
    DEVICE = torch.device('cuda')
    torch.cuda.set_device(args.num_gpu)
    torch.cuda.manual_seed(args.seed)
else:
    DEVICE = torch.device('cpu')
    print('>> CUDA is not available. Overridding with device = "cpu".')
    print('>> OMP_NUM_THREADS will be set to ' + str(args.num_threads))
    os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
    torch.set_num_threads(args.num_threads)


# Custom functions definitions

def get_dataframe(data_path, min_visits):
    """
    :param data_path: absolute path to data
    :return: dataframe with columns (time (list) | values (list) | apoe_all1 (int) | apoe_all2 (int))
    """
    # Get data (as pandas dataframes)
    df_pib = pd.read_sas(os.path.join(data_path, 'pib.sas7bdat'), format='sas7bdat')
    # df_mri = pd.read_sas(os.path.join(data_path, 'mri.sas7bdat'), format='sas7bdat')
    df_demo = pd.read_sas(os.path.join(data_path, 'demo.sas7bdat'), format='sas7bdat')
    # df_pacc = pd.read_sas(os.path.join(data_path, 'pacc.sas7bdat'), format='sas7bdat')

    # Preprocess data
    df_demo['reggieid'] = df_demo.reggieid.astype(int)
    df_pib['reggieid'] = df_pib.reggieid.astype(int)
    df_pib.head()
    df_time = df_pib.groupby(['reggieid'])['pib_age'].apply(list)
    df_values = df_pib.groupby(['reggieid'])['pib_index'].apply(list)
    df_merged = pd.concat([df_time, df_values], axis=1)
    assert len(df_time) == len(df_values) == len(df_merged)
    print('Number of patients : {:d}'.format(len(df_merged)))
    df_merged = df_merged[df_merged['pib_age'].apply(lambda x: len(x)) >= min_visits]
    print('Number of patients with visits > {} time : {:d}'.format(min_visits - 1, len(df_merged)))

    # Final merge
    df = df_merged.join(df_demo.set_index('reggieid')[['apoe_all1', 'apoe_all2']])
    return df


def ivp_integrate_GP(model, t_eval, y0):

    def modelGP_numpy_integrator(t, np_x):
        x = torch.from_numpy(np_x).float()
        x = x.view(1, np.size(np_x))  # batch size of 1
        x.requires_grad = True
        return gpu_numpy_detach(model(x).mean)

    sequence = solve_ivp(fun=modelGP_numpy_integrator, t_span=(t_eval[0], t_eval[-1]), y0=y0, t_eval=t_eval, rtol=1e-10)
    return sequence['y']


def data_to_derivatives(data_loader, preprocessing, var, alpha_default=10.):
    """
    Ridge regression on data to summarize trajectories with first derivatives
    """
    # Ridge regression for longitudinal reduction
    polynomial_features = PolynomialFeatures(degree=2, include_bias=True)
    alpha_ridge = 1e-10 if preprocessing else alpha_default
    ridge_regression = Ridge(alpha=alpha_ridge, fit_intercept=False)
    stats_derivatives = {'t_bar': [], 'means': [], 'bias': [], 'covars': []}

    for batch_idx, (positions, maskers, times, sequences, _) in enumerate(data_loader):
        for position, masker, time, sequence in zip(positions, maskers, times, sequences):

            # ------ Read STANDARDIZED data (as Ridge regression is not scale invariant)
            t = gpu_numpy_detach(time[masker == True])
            s = gpu_numpy_detach(sequence[masker == True])
            n = len(t)
            assert n > 2, "At least 3 points required for a 2nd order derivative estimate"
            t_bar = np.mean(t)

            # i) Fit Ridge regression
            t_poly = polynomial_features.fit_transform(t.reshape(-1, 1))
            ridge_regression.fit(t_poly, s)
            theta = np.array(ridge_regression.coef_)
            A_func = lambda t_ref: np.array([[1, t_ref, t_ref ** 2], [0., 1., 2 * t_ref], [0., 0., 2]])
            A_bar = A_func(t_bar)

            # ii) Regress fitted data at mean time point (ie t_bar)
            s_hat = A_bar.dot(theta)

            # iii) Store bias and variances on (biased) estimator
            H = np.linalg.inv(np.transpose(t_poly).dot(t_poly) + alpha_ridge * n * np.eye(3)).dot(
                np.transpose(t_poly).dot(t_poly))
            bias_theta = H.dot(theta)
            covar_theta = var * H.dot(
                np.linalg.inv(np.transpose(t_poly).dot(t_poly) + alpha_ridge * n * np.eye(3)))
            bias_derivatives = A_bar.dot(bias_theta)
            covars_derivatives = A_bar.dot(covar_theta).dot(np.transpose(A_bar))

            stats_derivatives['t_bar'].append(t_bar)
            stats_derivatives['means'].append(s_hat)
            stats_derivatives['bias'].append(bias_derivatives)
            stats_derivatives['covars'].append(covars_derivatives)

    x_derivatives = np.transpose(np.stack(stats_derivatives['means']))
    return x_derivatives


def estimate_initialconditions(data_loader, GP_model, model_hamiltonian, preprocessing, pib_threshold, timesteps=200,
                               nb_var=3, epochs=100, lr=1e-1, min_lr=1e-5, lr_decay=.9, patience=5, method='midpoint'):
    """
    Compute initial conditions at reference age (via Gradient Descent on initial conditions)
    """
    assert method in ['midpoint', 'rk4']
    # First pass on data to get time interval
    t_data = []
    for batch_idx, (positions, maskers, times, sequences, labels) in enumerate(data_loader):
        for position, masker, time, sequence, label in zip(positions, maskers, times, sequences, labels):
            # ------ REDO PROCESSING STEP IN THE SAME FASHION
            t = gpu_numpy_detach(time[masker == True])
            assert len(t) > 2, "At least 3 points required for a 2nd order derivative estimate"
            t_data.append(t)
    t_data = np.concatenate(t_data).ravel()
    t_line = np.linspace(t_data.min() - nb_var * t_data.var(),
                         t_data.max() + nb_var * t_data.var(), timesteps)
    t_ref = np.mean(t_data)

    model_hamiltonian.require_grad_field(True)    # set model gradients to True | allows gradients to flow

    # Ridge regression for longitudinal reduction
    polynomial_features = PolynomialFeatures(degree=2, include_bias=True)
    alpha_ridge = 1e-10 if preprocessing else 1e1
    ridge_regression = Ridge(alpha=alpha_ridge, fit_intercept=False)
    labeler_func = lambda arr: int(np.sum(arr))
    y_lab = []
    initial_conditions = []
    tref_initial_conditions = []
    time_values = []
    data_values = []
    i_undone = 0
    i_total = 0
    for batch_idx, (positions, maskers, times, sequences, labels) in enumerate(data_loader):
        for position, masker, time, sequence, label in tqdm(zip(positions, maskers, times, sequences, labels)):
            # ------ REDO PROCESSING STEP IN THE SAME FASHION
            t = gpu_numpy_detach(time[masker==True])
            s = gpu_numpy_detach(sequence[masker==True])
            t_bar = np.mean(t)

            # i) Fit Ridge regression
            t_poly = polynomial_features.fit_transform(t.reshape(-1, 1))
            ridge_regression.fit(t_poly, s)
            theta = np.array(ridge_regression.coef_)
            A_func = lambda t_ref: np.array([[1, t_ref, t_ref ** 2], [0., 1., 2 * t_ref], [0., 0., 2]])
            A_bar = A_func(t_bar)

            # ii) Regress fitted data at mean time point (ie t_bar) | compute estimated initial momentum
            s_bar = A_bar.dot(theta)[0]
            sdot_bar = A_bar.dot(theta)[1]
            p_bar = gpu_numpy_detach(
                model_hamiltonian.velocity_to_momenta(q=torch.from_numpy(np.array(A_bar.dot(theta)[0])).view(-1, 1),
                                                      v=torch.from_numpy(np.array(sdot_bar)).view(-1, 1)))
            initial_condition = torch.Tensor([s_bar, p_bar]).view(1, -1)  # (batch, 2*dim)

            # ------ Geodesic regression via gradient descent | initial conditions estimated at t_bar
            torch_t_eval = torch.from_numpy(t)
            torch_y = torch.from_numpy(s)
            torch_t_eval.requires_grad = False
            torch_y.requires_grad = False
            optimizer = Adam([initial_condition], lr=lr, amsgrad=True)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay, patience=patience,
                                          min_lr=min_lr, verbose=False, threshold_mode='abs')
            loss_ = 0
            initial_condition.requires_grad = True
            regressed_flow = 0
            best_loss_ = np.inf
            for epoch in range(epochs):
                optimizer.zero_grad()
                regressed_flow = torchdiffeq_torch_integrator(GP_model=GP_model, t_eval=torch_t_eval,
                                                              y0=initial_condition,
                                                              method=method, adjoint_boolean=False,
                                                              create_graph=True, retain_graph=True)
                delta = regressed_flow.squeeze()[:, 0] - torch_y
                loss_ = torch.mean(delta ** 2, dim=0, keepdim=False)
                loss_.backward(retain_graph=True)
                optimizer.step()
                scheduler.step(loss_)
                # Check NaN
                if gpu_numpy_detach(torch.isnan(initial_condition).sum()):
                    print('Nan encountered : breaking loop at epoch {}'.format(epoch))
                    i_undone += 1
                    break
                # Retrieve best regression parameters
                if gpu_numpy_detach(loss_) < best_loss_:
                    best_loss_ = gpu_numpy_detach(loss_)
                    best_init_ = gpu_numpy_detach(initial_condition)

            # ------ Shoot best registration to reference time
            t_evaluator = torch.from_numpy(np.array([t[0], t_ref]))
            best_y0s = torch.from_numpy(best_init_)
            best_y0s.requires_grad = True
            tref_regressed_flow = torchdiffeq_torch_integrator(GP_model=GP_model, t_eval=t_evaluator, y0=best_y0s,
                                                               method=method, adjoint_boolean=False,
                                                               create_graph=True, retain_graph=False)
            tref_initial_conditions.append(gpu_numpy_detach(tref_regressed_flow[-1]))
            initial_conditions.append(best_init_)
            time_values.append(t)
            data_values.append(s)
            y_lab.append(labeler_func(gpu_numpy_detach(label)))
            i_total += 1
    print('>> Rejected rate = {:.1f}% ({} / {})\n'.format(100 * (i_undone / float(i_total)), i_undone, i_total))

    model_hamiltonian.require_grad_field(False)     # reset model gradients to False

    # Indices for APOE pairs: (3,3) | (3,4) | (4,4)
    idx_33 = np.argwhere(np.array(y_lab) == 6)
    idx_34 = np.argwhere(np.array(y_lab) == 7)
    idx_44 = np.argwhere(np.array(y_lab) == 8)

    # inital conditions : at observation point (t_bar individual) | at t_ref (mean of dataset time observed)
    np_ic = np.concatenate([ic.reshape(-1, 1) for ic in initial_conditions], axis=-1)
    np_ic_ref = np.concatenate([ic.reshape(-1, 1) for ic in tref_initial_conditions], axis=-1)

    return idx_33, idx_34, idx_44, t_line, t_ref, np_ic, np_ic_ref, time_values, data_values


def run(args, device):

    # ==================================================================================================================
    # SNAPSHOTS
    # ==================================================================================================================

    log = ''
    args.model_signature = str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')
    args.snapshots_path = os.path.join(args.output_dir, 'HAMILTONIAN_{}'.format(args.model_signature))
    if not os.path.exists(args.snapshots_path):
        os.makedirs(args.snapshots_path)

    with open(os.path.join(args.snapshots_path, 'args.json'), 'w') as f:
        args_wo_device = deepcopy(args.__dict__)
        args_wo_device.pop('device', None)
        json.dump(args_wo_device, f, indent=4, sort_keys=True)

    args.device = device

    # ==================================================================================================================
    # LOAD DATA | PERFORM REGRESSION
    # ==================================================================================================================

    # Get raw data
    df = get_dataframe(data_path=args.data_dir, min_visits=args.min_visits)
    train_loader, val_loader, test_loader, all_loader, (times_mean, times_std), \
        (data_mean, data_std) = get_fromdataframe(df=df, batch_size=args.batch_size,
                                                  standardize=args.preprocessing, seed=args.seed)

    # Helper functions
    destandardize_time = lambda time: time * times_std + times_mean if args.preprocessing else time
    destandardize_data = lambda data: data * data_std + data_mean if args.preprocessing else data
    restandardize_data = lambda data: (data - data_mean) / data_std if args.preprocessing else data

    # Ridge regression for longitudinal reduction
    x_derivatives = data_to_derivatives(data_loader=all_loader, preprocessing=args.preprocessing, var=args.noise_std**2)
    u = x_derivatives[0]
    v = x_derivatives[1]
    v_squ = x_derivatives[1] ** 2
    w = x_derivatives[2]
    ratio = -1. * w / v_squ
    filter_quantiles = args.filter_quantiles
    eps_quantiles = args.eps_quantiles
    if filter_quantiles:
        nb_before = len(ratio)
        filtered_index = np.where(
            (ratio > np.quantile(ratio, eps_quantiles)) & (ratio < np.quantile(ratio, 1 - eps_quantiles)))
        u = u[filtered_index]
        v = v[filtered_index]
        v_squ = v_squ[filtered_index]
        w = w[filtered_index]
        ratio = ratio[filtered_index]
        print('---\nOriginal nb points : {} \nRetained nb points : {}\n---\n'.format(nb_before, len(ratio)))
    else:
        print('---\nNo filtering of extremal values\n---\n')

    # Plot figures
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.scatter(x=u, y=ratio, marker='o', color='k', s=10.)
    ax.set_xlim(np.min(u) - 1e-4, np.max(u) + 1e-4)
    ax.set_xlabel('x')
    ax.set_ylabel('-x_dotdot / x_dot ^2')
    ax.set_title('Estimates for Christoffel relationship')
    plt.savefig(os.path.join(args.snapshots_path, 'christoffel_relationship.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

    # ==================================================================================================================
    # GAUSSIAN PROCESS FIT
    # ==================================================================================================================

    # hyperparameter tuning for scale parameter | should be kept low to prevent overfitting
    training_iter = args.tuning     # for standardized, prefer low 5 | for raw, can go up to 50 with linear / poly
    train_x = torch.from_numpy(u).float()
    train_y = torch.from_numpy(ratio).float()
    likelihood_GP = gpytorch.likelihoods.GaussianLikelihood()       # default noise : can be initialized wrt priors

    # list of ODE GP models
    if args.model_type == 'RBF':
        model_GP_H = ExactGPModel('RBF', train_x, train_y, likelihood_GP)
    elif args.model_type == 'LINEAR':
        model_GP_H = ExactGPModel('linear', train_x, train_y, likelihood_GP)
    elif args.model_type == 'POLY':
        model_GP_H = ExactGPModel('polynomial', train_x, train_y, likelihood_GP, power=2)
    else:
        raise ValueError("model not accounted for yet ...")

    # Find optimal GP model hyperparameters - akin to tuning | should be kept low to prevent overfitting
    model_GP_H.train()
    likelihood_GP.train()
    print('\n---')
    print('GP hyperparameters : pre-tuning\n')
    for name, param in model_GP_H.named_parameters():
        print('>> {}'.format(name))
        print('      {}\n'.format(param))
    print('---\n')

    optimizer = torch.optim.Adam([
        {'params': model_GP_H.parameters()}], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_GP, model_GP_H)
    for i in range(training_iter):
        optimizer.zero_grad()           # Zero gradients from previous iteration
        output = model_GP_H(train_x)      # Output from model
        loss = -mll(output, train_y)    # Calc loss and backprop gradients
        loss.backward()
        optimizer.step()

    model_GP_H.eval()
    likelihood_GP.eval()

    # ------ Plots
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.scatter(x=destandardize_data(torch.from_numpy(u)), y=ratio, marker='o', color='k', s=10.)
    with torch.no_grad():
        u_line = np.linspace(u.min(), u.max(), 200)
        f_preds = model_GP_H(torch.from_numpy(u_line).float())
        f_mean = f_preds.mean
        f_var = f_preds.variance
        ax.plot(destandardize_data(torch.from_numpy(u_line)), gpu_numpy_detach(f_mean), label=model_GP_H.name)
        lower, upper = f_mean - 2. * f_var, f_mean + 2. * f_var
        ax.fill_between(destandardize_data(torch.from_numpy(u_line)), gpu_numpy_detach(lower), gpu_numpy_detach(upper), alpha=0.2)
    # ax.set_ylim(np.min(ratio) - 1e-4, np.max(ratio) + 1e-4)
    ax.set_xlabel('x')
    ax.set_ylabel('- x_dotdot / x_dot^2')
    ax.set_title('GP regression on Christoffel relationship')
    ax.legend()
    plt.gray()
    plt.savefig(os.path.join(args.snapshots_path, 'GP_fit.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

    # ==================================================================================================================
    # FIT CHRISTOFFEL SYMBOL, COMPUTE INVERSE METRIC, INITIALIZE HAMILTONIAN OBJECT
    # ==================================================================================================================

    # I) Christoffel symbol estimate (trapezoidal integration)
    u_line = np.linspace(u.min(), u.max(), 500)
    ful_torch = torch.from_numpy(u_line).float().view(-1, 1)
    Gamma = Christoffel(fitted_model=model_GP_H, xmin=ful_torch.min().clone().detach(), xmax=ful_torch.max().clone().detach())

    # II) Hamiltonian object derivation
    hamiltonian_fn = Hamiltonian(Gamma)
    inverse_estimate = hamiltonian_fn.inverse_metric(ful_torch, precision=1000)
    metric_estimate = hamiltonian_fn.metric(ful_torch)

    # ------ Plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 18), sharex=True)
    ax1.plot(destandardize_data(torch.from_numpy(u_line)), np.squeeze(gpu_numpy_detach(inverse_estimate)),
             '-.r', label='inverse metric')
    ax1bis = ax1.twinx()
    ax1bis.plot(destandardize_data(torch.from_numpy(u_line)), np.squeeze(gpu_numpy_detach(metric_estimate)),
                '-.b', label='metric')
    ax1.set_xlabel('PiB index')
    ax1.set_ylabel('inverse metric')
    ax1bis.set_ylabel('metric')
    ax1.set_title('Estimated metric (normalized) for {} GP'.format(model_GP_H.name))
    ax1.legend(loc=0)
    ax1bis.legend(loc=2)

    # -------- compute associated Hamiltonian flow (rotated gradient)
    momentum_p = gpu_numpy_detach(hamiltonian_fn.velocity_to_momenta(torch.from_numpy(u).float().view(-1, 1),
                                                                     torch.from_numpy(v).float().view(-1, 1)))
    ax2.scatter(x=destandardize_data(torch.from_numpy(u)), y=momentum_p, marker='o', color='r', s=40.)
    Y, X = np.mgrid[momentum_p.min():momentum_p.max():150j,
           u.min():u.max():150j]
    _, X_std = np.mgrid[momentum_p.min():
                        momentum_p.max():150j,
               gpu_numpy_detach(destandardize_data(torch.from_numpy(u))).min():
               gpu_numpy_detach(destandardize_data(torch.from_numpy(u))).max():150j]
    Z = torch.from_numpy(np.stack([X, Y])).reshape(2, len(X) * len(Y)).t().type(torch.float)
    Z.requires_grad = True   # required for automatic gradient computation
    Field_grad = hamiltonian_fn.get_Rgrad(Z, retain_graph=False, create_graph=False)
    Field_grid = Field_grad.reshape(len(X), len(Y), 2).permute(2, 0, 1)
    F_speed = gpu_numpy_detach(torch.norm(Field_grid, p=2, dim=0))
    Field_grid = gpu_numpy_detach(Field_grid)
    lw = 10 * F_speed / F_speed.max()

    #  Varying density along a streamline
    _ = ax2.streamplot(X_std, Y, Field_grid[0], Field_grid[1], linewidth=1., color=lw, cmap='plasma')
    ax2.set_title('Vector field')
    ax2.set_ylim(np.min(momentum_p) - 1e-4, np.max(momentum_p) + 1e-4)
    ax2.set_xlabel('PiB')
    ax2.set_ylabel('momenta')
    ax2.set_title('GP regression {} on ODE function'.format(model_GP_H.name))

    # Zooming on outliers excluded
    eps_filter = .05
    fIDX = np.where(
        (momentum_p > np.quantile(momentum_p, eps_filter)) & (momentum_p < np.quantile(momentum_p, 1 - eps_filter)))[0]
    Y_zoom, X_zoom = np.mgrid[momentum_p[fIDX].min():momentum_p[fIDX].max():150j,
           u[fIDX].min():u[fIDX].max():150j]
    _, X_zoom_std = np.mgrid[momentum_p[fIDX].min():momentum_p[fIDX].max():150j,
               gpu_numpy_detach(destandardize_data(torch.from_numpy(u[fIDX]))).min():
               gpu_numpy_detach(destandardize_data(torch.from_numpy(u[fIDX]))).max():150j]
    Z_zoom = torch.from_numpy(np.stack([X_zoom, Y_zoom])).reshape(2, len(X_zoom) * len(Y_zoom)).t().type(torch.float)
    Z_zoom.requires_grad = True
    # Hamiltonian integral curves
    Field_grad_zoom = hamiltonian_fn.get_Rgrad(Z_zoom, retain_graph=False, create_graph=False)
    Field_grid_zoom = Field_grad_zoom.reshape(len(X_zoom), len(Y_zoom), 2).permute(2, 0, 1)
    F_speed_zoom = gpu_numpy_detach(torch.norm(Field_grid_zoom, p=2, dim=0))
    Field_grid_zoom = gpu_numpy_detach(Field_grid_zoom)
    lw_zoom = 10 * F_speed_zoom / F_speed_zoom.max()
    ax3.scatter(x=destandardize_data(torch.from_numpy(u[fIDX])), y=momentum_p[fIDX], marker='o', color='r',
                s=40.)
    _ = ax3.streamplot(X_zoom_std, Y_zoom, Field_grid_zoom[0], Field_grid_zoom[1], linewidth=1., color=lw_zoom, cmap='plasma')
    ax3.set_title('Vector field (zoom)')
    ax3.set_ylim(np.min(momentum_p[fIDX]) - 1e-4, np.max(momentum_p[fIDX]) + 1e-4)
    ax3.set_xlabel('PiB')
    ax3.set_ylabel('momenta')
    ax3.set_title('GP regression {} on ODE function (zoom)'.format(model_GP_H.name))

    plt.savefig(os.path.join(args.snapshots_path, 'hamiltonian_field.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

    # ==================================================================================================================
    # GEODESIC REGRESSION OF INITIAL CONDITIONS AT REFERENCE AGE (TIME CONSUMING STEP)
    # ==================================================================================================================

    timesteps = 200
    idx_33, idx_34, idx_44, t_line, t_ref, np_ic, \
    np_ic_ref, time_values, data_values = estimate_initialconditions(all_loader, model_GP_H, hamiltonian_fn,
                                                                     args.preprocessing,
                                                                     restandardize_data(args.pib_threshold),
                                                                     timesteps=timesteps, nb_var=2, epochs=args.epochs,
                                                                     lr=args.lr, min_lr=args.lr_min,
                                                                     lr_decay=args.lr_decay, patience=args.patience,
                                                                     method=args.method)

    # ------ Plot initial conditions at reference age
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    # Initial conditions regression at reference age
    ax1.scatter(x=destandardize_data(torch.from_numpy(np_ic_ref[0, idx_33])), y=np_ic_ref[1, idx_33], marker='o',
                color='r', s=20., label='APOE 33')
    ax1.scatter(x=destandardize_data(torch.from_numpy(np_ic_ref[0, idx_34])), y=np_ic_ref[1, idx_34], marker='x',
                color='b', s=20., label='APOE 34')
    ax1.scatter(x=destandardize_data(torch.from_numpy(np_ic_ref[0, idx_44])), y=np_ic_ref[1, idx_44], marker='d',
                color='g', s=20., label='APOE 44')
    ax1.set_xlabel('PiB')
    ax1.set_ylabel('momentum')
    ax1.set_yscale('symlog')
    ax1.set_title('Initial conditions distribution at age {:d}'.format(
        int(destandardize_time(torch.from_numpy(np.array(t_ref))))))
    ax1.legend()

    # ------ Plot random geodesics
    ymin, ymax = np.inf, - np.inf
    idx1, idx2, idx3 = np.random.choice(idx_33.squeeze(), 2, replace=False), \
                       np.random.choice(idx_34.squeeze(), 2, replace=False), \
                       np.random.choice(idx_44.squeeze(), 2, replace=False)
    for idds, name, color in zip([idx1, idx2, idx3], ['APOE 33', 'APOE 34', 'APOE 44'], ['r', 'b', 'g']):
        for i, idd in enumerate(idds):
            idd = int(idd)
            # real observed data
            if i == 0:
                ax2.plot(destandardize_time(torch.from_numpy(time_values[idd])),
                         destandardize_data(torch.from_numpy(data_values[idd])),
                         linestyle='-', marker='o', color=color, label=name)
            else:
                ax2.plot(destandardize_time(torch.from_numpy(time_values[idd])),
                         destandardize_data(torch.from_numpy(data_values[idd])),
                         linestyle='-', marker='o', color=color)
            ax2.scatter(x=destandardize_time(torch.from_numpy(np.array([t_ref]))),
                        y=destandardize_data(torch.from_numpy(np_ic_ref)[0, idd]),
                        marker='*', s=75, color='k')
            # Compute real geodesic onward and backward of reference age
            t_backward = torch.linspace(t_ref,  t_line.min(), 50)
            t_forward = torch.linspace(t_ref,  t_line.max(), 50)
            t_all = np.concatenate((gpu_numpy_detach(t_backward)[::-1][:-1], gpu_numpy_detach(t_forward)))
            best_y0s = torch.from_numpy(np_ic_ref[:, idd]).view(1, -1)
            best_y0s.requires_grad = True
            forward_geo = torchdiffeq_torch_integrator(GP_model=model_GP_H, t_eval=t_forward, y0=best_y0s,
                                                       method=args.method, adjoint_boolean=False, create_graph=False,
                                                       retain_graph=False)
            backward_geo = torchdiffeq_torch_integrator(GP_model=model_GP_H, t_eval=t_backward, y0=best_y0s,
                                                        method=args.method, adjoint_boolean=False, create_graph=False,
                                                        retain_graph=False)
            all_geo = np.concatenate(
                (gpu_numpy_detach(backward_geo)[::-1][:-1, 0, 0], gpu_numpy_detach(forward_geo)[:, 0, 0]))
            ax2.plot(destandardize_time(torch.from_numpy(t_all)), destandardize_data(torch.from_numpy(all_geo)),
                     linestyle='-.', color=color)
            ymin = min(ymin, gpu_numpy_detach(destandardize_data(torch.from_numpy(data_values[idd]))).min())
            ymax = max(ymax, gpu_numpy_detach(destandardize_data(torch.from_numpy(data_values[idd]))).max())

    ax2.set_ylim(ymin - .1 * (y_max - y_min), ymax + .1 * (y_max - y_min))
    ax2.set_xlabel('Age')
    ax2.set_ylabel('PiB')
    ax2.set_title('Geodesic reconstruction for random observations')
    ax2.legend()

    plt.savefig(os.path.join(args.snapshots_path, 'random_geodesics.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

    # ==================================================================================================================
    # AGE AT ONSET COMPUTATION | OBSERVE REGRESSED GEODESICS IN PHASE SPACE (TIME CONSUMING STEP)
    # ==================================================================================================================

    print('Age at onset computation and plot [time consuming step]...')
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(x=destandardize_data(torch.from_numpy(np_ic_ref[0, idx_33])), y=np_ic_ref[1, idx_33], marker='*',
               color='r', s=50., label='APOE 33')
    ax.scatter(x=destandardize_data(torch.from_numpy(np_ic_ref[0, idx_34])), y=np_ic_ref[1, idx_34], marker='*',
               color='b', s=50., label='APOE 34')
    ax.scatter(x=destandardize_data(torch.from_numpy(np_ic_ref[0, idx_44])), y=np_ic_ref[1, idx_44], marker='*',
               color='g', s=50., label='APOE 44')

    # Reconstructed trajectories | estimated age at onset (aka PiB positivity)
    pib_at_t_ref = []  # storing estimated age of crossing PiB positivity
    for idds, name, color in zip([idx_33.squeeze(), idx_34.squeeze(), idx_44.squeeze()],
                                 ['APOE 33', 'APOE 34', 'APOE 44'], ['r', 'b', 'g']):
        for i, idd in tqdm(enumerate(idds)):
            idd = int(idd)
            # Time stretch : intrapolation | extrapolation
            t_obs_min, t_obs_max = time_values[idd][0], time_values[idd][-1]
            t_backward = torch.linspace(t_ref, t_line.min(), 50)
            t_forward = torch.linspace(t_ref, t_line.max(), 50)
            t_all = np.concatenate((gpu_numpy_detach(t_backward)[::-1][:-1], gpu_numpy_detach(t_forward)))
            intrapolation_indexes = (t_all > t_obs_min) & (t_all < t_obs_max)

            best_y0s = torch.from_numpy(np_ic_ref[:, idd]).view(1, -1)
            best_y0s.requires_grad = True
            forward_geo = torchdiffeq_torch_integrator(GP_model=model_GP_H, t_eval=t_forward, y0=best_y0s,
                                                       method=args.method, adjoint_boolean=False, create_graph=True,
                                                       retain_graph=False)
            backward_geo = torchdiffeq_torch_integrator(GP_model=model_GP_H, t_eval=t_backward, y0=best_y0s,
                                                        method=args.method, adjoint_boolean=False, create_graph=True,
                                                        retain_graph=False)
            all_geo = np.concatenate(
                (gpu_numpy_detach(backward_geo)[::-1][:-1, 0, :], gpu_numpy_detach(forward_geo)[:, 0, :]), axis=0)

            # Plot trajectories in phase space | emphasis on observed time sequence
            ax.plot(destandardize_data(torch.from_numpy(all_geo[:, 0])), all_geo[:, 1], linestyle='-.', linewidth=.5,
                    color=color)
            ax.plot(destandardize_data(torch.from_numpy(all_geo[intrapolation_indexes, 0])),
                    all_geo[intrapolation_indexes, 1], linestyle='-', linewidth=2., color=color)
            posidx = np.where(all_geo[:, 0] > restandardize_data(args.pib_threshold))[0]
            pib_at_t_ref.append(t_all[posidx[0]] if len(posidx) else None)

    ax.set_xlabel('PiB')
    ax.set_xlim(0.5, 2.5)      # ad hoc values, could be changed
    ax.set_ylabel('momentum')
    ax.set_yscale('symlog')
    ax.set_title('Phase space trajectories from reference age {:d}'.format(
        int(destandardize_time(torch.from_numpy(np.array(t_ref))))))
    ax.legend()
    plt.savefig(os.path.join(args.snapshots_path, 'phase_space_regressions.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

    # ==================================================================================================================
    # (WILCOXON) RANK TEST ON APOE PAIRS
    # ==================================================================================================================

    boxplot_stock_pib = []
    # boxplot_stock_age = []
    for i_, name in zip([idx_33, idx_34, idx_44], ['APOE 33', 'APOE 34', 'APOE 44']):
        age_tref = int(gpu_numpy_detach(destandardize_time(torch.from_numpy(np.array([t_ref]))))[0])
        pib_tref = gpu_numpy_detach(destandardize_data(torch.from_numpy(np_ic_ref[0, i_])))
        p_tref = gpu_numpy_detach(destandardize_data(torch.from_numpy(np_ic_ref[1, i_])))
        # trueage_tref = int(gpu_numpy_detach(np.array([pib_at_t_ref[int(i)] for i in i_])))
        boxplot_stock_pib.append(pib_tref)
        # boxplot_stock_age.append(trueage_tref)
        print('{}'.format(name))
        print('          PiB at reference age {} : {:.2f} +- {:.2f}'.format(age_tref, np.mean(pib_tref),
                                                                            np.std(pib_tref)))
        print('          Mom at reference age {} : {:.2f} +- {:.2f}'.format(age_tref, np.mean(p_tref),
                                                                            np.std(p_tref)))

    # -------- compute wilcowon vectorial scores
    wilcox_A_01 = ranksums(x=np_ic[0, idx_33].squeeze(),
                           y=np_ic[0, idx_34].squeeze())
    wilcox_A_02 = ranksums(x=np_ic[0, idx_33].squeeze(),
                           y=np_ic[0, idx_44].squeeze())
    wilcox_A_12 = ranksums(x=np_ic[0, idx_34].squeeze(),
                           y=np_ic[0, idx_44].squeeze())
    wilcox_B_01 = ranksums(x=np_ic[1, idx_33].squeeze(),
                           y=np_ic[1, idx_34].squeeze())
    wilcox_B_02 = ranksums(x=np_ic[1, idx_33].squeeze(),
                           y=np_ic[1, idx_44].squeeze())
    wilcox_B_12 = ranksums(x=np_ic[1, idx_34].squeeze(),
                           y=np_ic[1, idx_44].squeeze())

    print('\n')
    print('Wilcoxon test p-value for difference in PIB (at ref age) between {} and {} = {:.3f}'.format('APOE 33',
                                                                                                       'APOE 34',
                                                                                                       wilcox_A_01.pvalue))
    print('Wilcoxon test p-value for difference in vel (at ref age) between {} and {} = {:.3f}'.format('APOE 33',
                                                                                                       'APOE 34',
                                                                                                       wilcox_B_01.pvalue))
    print('-----')
    print('Wilcoxon test p-value for difference in age at PIB positive between {} and {} = {:.3f}'.format('APOE 33',
                                                                                                          'APOE 44',
                                                                                                          wilcox_A_02.pvalue))
    print('Wilcoxon test p-value for difference in vel (at ref age) between {} and {} = {:.3f}'.format('APOE 33',
                                                                                                       'APOE 44',
                                                                                                       wilcox_B_02.pvalue))
    print('-----')
    print('Wilcoxon test p-value for difference in age at PIB positive between {} and {} = {:.3f}'.format('APOE 34',
                                                                                                          'APOE 44',
                                                                                                          wilcox_A_12.pvalue))
    print('Wilcoxon test p-value for difference in vel (at ref age) between {} and {} = {:.3f}'.format('APOE 34',
                                                                                                       'APOE 44',
                                                                                                       wilcox_B_12.pvalue))
    print('\n')

    fig = plt.subplots(figsize=(10, 5))
    plt.boxplot(boxplot_stock_pib)
    plt.xticks([i for i in range(1, 4)], ['APOE 33', 'APOE 34', 'APOE 44'], rotation=60)
    plt.ylabel('PiB')
    plt.title('PiB score at reference age : {} years'.format(age_tref))
    plt.savefig(os.path.join(args.snapshots_path, 'boxplots_pib.png'), bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    run(args, device=DEVICE)

