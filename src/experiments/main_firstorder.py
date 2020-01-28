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

# sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.integrate import solve_ivp
from scipy.stats import ranksums

# Torch
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

# utils
import gpytorch
from src.utils.op import gpu_numpy_detach
from src.utils.datasets import get_fromdataframe
from src.utils.models import ExactGPModel


parser = argparse.ArgumentParser(description='ODE fit with GP on PiB data.')
# action parameters
parser.add_argument('--output_dir', type=str, default='./results/ODE', help='Output directory.')
parser.add_argument('--data_dir', type=str, default='./', help='Data directory.')
parser.add_argument('--cuda', action='store_true', help='Whether CUDA is available on GPUs.')
parser.add_argument('--num_gpu', type=int, default=0, help='Which GPU to run on.')
parser.add_argument('--num_threads', type=int, default=36, help='Number of threads to use if cuda not available')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
# dataset parameters
parser.add_argument('--min_visits', type=int, default=3, help='Minimal number of visits.')
parser.add_argument('--preprocessing', action='store_true', help='Standardization of not.')
parser.add_argument('--noise_std', type=float, default=.07, help='Std of additive noise on observations.')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
# model parameters
parser.add_argument('--model_type', type=str, default='POLY', choices=['RBF', 'LINEAR', 'POLY'], help='GP model type.')
parser.add_argument('--tuning', type=int, default=50, help='Tuning of GP hyperparameters.')
parser.add_argument('--pib_threshold', type=float, default=1.2, help='Onset for PiB positivity (data dependant).')
parser.add_argument('--use_vars', action='store_true', help='Wether to use variance information or not.')


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
    print('Total number of patients : {:d}'.format(len(df_merged)))
    print('Mean number of visits : {:.2f}'.format(np.mean(df_merged['pib_age'].apply(lambda x: len(x)))))
    print('Mean span of visits : {:.2f}'.format(np.mean(df_merged['pib_age'].apply(lambda x: max(x) - min(x)))))
    df_merged = df_merged[df_merged['pib_age'].apply(lambda x: len(x)) >= min_visits]
    print('Now filtering patients with visits > {}'.format(min_visits - 1))
    print('>> Number of patients : {:d}'.format(len(df_merged)))
    print('>> Number of visits in average : {:.2f}'.format(np.mean(df_merged['pib_age'].apply(lambda x: len(x)))))
    print('>> Number of span in average : {:.2f}'.format(np.mean(df_merged['pib_age'].apply(lambda x: max(x) - min(x)))))

    # Final merge
    df = df_merged.join(df_demo.set_index('reggieid')[['apoe_all1', 'apoe_all2']])
    assert len(df) == len(df_merged)
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
    polynomial_features = PolynomialFeatures(degree=1, include_bias=True)
    alpha_ridge = 1e-10 if preprocessing else alpha_default
    ridge_regression = Ridge(alpha=alpha_ridge, fit_intercept=False)
    stats_derivatives = {'t_bar': [], 'means': [], 'bias': [], 'covars': []}

    for batch_idx, (positions, maskers, times, sequences, _) in enumerate(data_loader):
        for position, masker, time, sequence in zip(positions, maskers, times, sequences):

            # ------ Read STANDARDIZED data (as Ridge regression is not scale invariant)
            t = gpu_numpy_detach(time[masker == True])
            s = gpu_numpy_detach(sequence[masker == True])
            n = len(t)
            assert len(t) > 1, "At least 2 points required for a 1st order derivative estimate"
            t_bar = np.mean(t)

            # i) Fit LINEAR Ridge regression
            t_ = polynomial_features.fit_transform(t.reshape(-1, 1))
            ridge_regression.fit(t_, s)
            theta = np.array(ridge_regression.coef_)
            A_func = lambda t_ref: np.array([[1, t_ref], [0., 1.]])
            A_bar = A_func(t_bar)

            # ii) Regress fitted data at mean time point (ie t_bar)
            s_hat = A_bar.dot(theta)

            # iii) Store bias and variances on (biased) estimator
            H = np.linalg.inv(np.transpose(t_).dot(t_) + alpha_ridge * n * np.eye(2)).dot(
                np.transpose(t_).dot(t_))
            bias_theta = H.dot(theta)
            covar_theta = var * H.dot(
                np.linalg.inv(np.transpose(t_).dot(t_) + alpha_ridge * n * np.eye(2)))
            bias_derivatives = A_bar.dot(bias_theta)
            covars_derivatives = A_bar.dot(covar_theta).dot(np.transpose(A_bar))

            stats_derivatives['t_bar'].append(t_bar)
            stats_derivatives['means'].append(s_hat)
            stats_derivatives['bias'].append(bias_derivatives)
            stats_derivatives['covars'].append(covars_derivatives)

    x_derivatives = np.transpose(np.stack(stats_derivatives['means']))
    x_uncertainties = np.transpose(np.stack(stats_derivatives['covars']))
    return x_derivatives, x_uncertainties


def align_trajectories(data_loader, model_GP, preprocessing, pib_threshold, timesteps=200, nb_var=3):
    """
    Aligns trajectories to common reference (such that x(t_ref) = pib_threshold)
    """
    # First pass on data to get time interval
    t_data = []
    for batch_idx, (positions, maskers, times, sequences, labels) in enumerate(data_loader):
        for position, masker, time, sequence, label in zip(positions, maskers, times, sequences, labels):
            # ------ REDO PROCESSING STEP IN THE SAME FASHION
            t = gpu_numpy_detach(time[masker == True])
            assert len(t) > 1, "At least 2 points required for a 1st order derivative estimate"
            t_data.append(t)
    t_data = np.concatenate(t_data).ravel()
    t_line = np.linspace(t_data.min() - nb_var * t_data.var(),
                         t_data.max() + nb_var * t_data.var(), timesteps)
    t_ref = np.mean(t_data)

    # Reference trajectory
    i_start = np.argmin(np.abs(t_line - t_ref))
    x_init = np.array([pib_threshold])
    forward_x = ivp_integrate_GP(model=model_GP, t_eval=t_line[i_start:], y0=x_init)
    backward_x = ivp_integrate_GP(model=model_GP, t_eval=t_line[:i_start + 1][::-1], y0=x_init)
    reference_trajectory = np.concatenate(
        [np.concatenate(backward_x).ravel()[::-1][:-1], np.concatenate(forward_x).ravel()])

    # Ridge regression for longitudinal reduction
    polynomial_features = PolynomialFeatures(degree=1, include_bias=True)
    alpha_ridge = 1e-10 if preprocessing else 1e1
    ridge_regression = Ridge(alpha=alpha_ridge, fit_intercept=False)
    labeler_func = lambda arr: int(np.sum(arr))
    y_lab = []
    initial_conditions = []
    initial_dtau = []
    time_values = []
    data_values = []
    i_undone = 0
    i_total = 0
    for batch_idx, (positions, maskers, times, sequences, labels) in enumerate(data_loader):
        for position, masker, time, sequence, label in zip(positions, maskers, times, sequences, labels):
            # ------ REDO PROCESSING STEP IN THE SAME FASHION
            t = gpu_numpy_detach(time[masker==True])
            s = gpu_numpy_detach(sequence[masker==True])
            t_bar = np.mean(t)

            # i) Fit Ridge regression
            t_ = polynomial_features.fit_transform(t.reshape(-1, 1))
            ridge_regression.fit(t_, s)
            theta = np.array(ridge_regression.coef_)
            A_func = lambda t_ref: np.array([[1, t_ref], [0., 1.]])
            A_bar = A_func(t_bar)

            # ii) Regress fitted data at mean time point (ie t_bar)
            s_hat = A_bar.dot(theta)

            # ------ Registration of curves by time translation wrt observed point
            i_relative = np.argmin(np.abs(reference_trajectory - s_hat[0]))
            if 0 < i_relative < len(reference_trajectory):
                tau_relative = t_line[i_relative] - t_bar
                initial_dtau.append(tau_relative)
                i_ic = np.argmin(np.abs(t_line - (t_ref + tau_relative)))
                initial_conditions.append(reference_trajectory[i_ic])
                time_values.append(t)
                data_values.append(s)
                y_lab.append(labeler_func(gpu_numpy_detach(label)))
            else:
                i_undone += 1
            i_total += 1
    print('>> Rejected rate = {:.1f}% ({} / {})\n'.format(100 * (i_undone / float(i_total)), i_undone, i_total))

    # Indices for APOE pairs: (3,3) | (3,4) | (4,4)
    idx_33 = np.argwhere(np.array(y_lab) == 6)
    idx_34 = np.argwhere(np.array(y_lab) == 7)
    idx_44 = np.argwhere(np.array(y_lab) == 8)

    return idx_33, idx_34, idx_44, t_line, reference_trajectory, t_ref, initial_conditions, initial_dtau, time_values, data_values


def run(args, device):

    # ==================================================================================================================
    # SNAPSHOTS
    # ==================================================================================================================

    log = ''
    args.model_signature = str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')
    args.snapshots_path = os.path.join(args.output_dir, 'ODE_{}'.format(args.model_signature))
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

    # Plot dataset
    xpltmin, xpltmax = np.inf, -np.inf
    fig, ax = plt.subplots(figsize=(5, 5))
    for batch_idx, (positions, maskers, times, sequences, _) in enumerate(all_loader):
        for position, masker, time, sequence in zip(positions, maskers, times, sequences):
            t = gpu_numpy_detach(destandardize_time(time[masker == True]))
            s = gpu_numpy_detach(destandardize_data(sequence[masker == True]))
            ax.plot(t, s, '--o', linewidth=1., markersize=5)
            xpltmin = min(xpltmin, min(t))
            xpltmax = max(xpltmax, max(t))
    ax.hlines(y=1.2, xmin=xpltmin, xmax=xpltmax, linestyle='-', color='k', alpha=.5, label='positivity threshold')
    # ax.set_title('WRAP dataset')
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('PiB cDVR', fontsize=12)
    ax.legend()
    plt.savefig(os.path.join(args.snapshots_path, 'dataset.png'), pad_inches=.5)  # bbox_inches='tight',
    plt.close()

    # Ridge regression for longitudinal reduction
    x_derivatives, x_uncertainties = data_to_derivatives(data_loader=all_loader, preprocessing=args.preprocessing,
                                                         var=gpu_numpy_detach(destandardize_data(
                                                             torch.from_numpy(np.array([args.noise_std])) ** 2)))

    u = x_derivatives[0]
    v = x_derivatives[1]

    # Plot figures
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x=u, y=v, marker='o', color='k', s=10.)
    ax.set_xlim(np.min(u) - 1e-4, np.max(u) + 1e-4)
    ax.set_xlabel('PiV cDVR', fontsize=12)
    ax.set_ylabel('Slope of PiB cDVR (1/year)', fontsize=10)
    # ax.set_title('Estimates for ODE')
    plt.savefig(os.path.join(args.snapshots_path, 'first_derivative_relationship.png'), pad_inches=0.1, bbox_inches='tight')
    plt.close()

    # ==================================================================================================================
    # GAUSSIAN PROCESS FIT
    # ==================================================================================================================

    # hyperparameter tuning for scale parameter | should be kept low to prevent overfitting
    training_iter = args.tuning     # for standardized, prefer low 5 | for raw, can go up to 50 with linear / poly
    train_x = torch.from_numpy(u).float()
    train_y = torch.from_numpy(v).float()

    if args.use_vars:
        vardot_noises = torch.from_numpy(x_uncertainties[1, 1]).squeeze().float()  # uncertainties on x_dot
        likelihood_GP = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=vardot_noises, learn_additional_noise=False)
    else:
        likelihood_GP = gpytorch.likelihoods.GaussianLikelihood()

    # list of ODE GP models
    if args.model_type == 'RBF':
        model_GP_ODE = ExactGPModel('RBF', train_x, train_y, likelihood_GP)
    elif args.model_type == 'LINEAR':
        model_GP_ODE = ExactGPModel('linear', train_x, train_y, likelihood_GP)
    elif args.model_type == 'POLY':
        model_GP_ODE = ExactGPModel('polynomial', train_x, train_y, likelihood_GP, power=2)
    else:
        raise ValueError("model not accounted for yet ...")

    # Find optimal GP model hyperparameters - akin to tuning | should be kept low to prevent overfitting
    model_GP_ODE.train()
    likelihood_GP.train()
    print('\n---')
    print('GP hyperparameters : pre-tuning\n')
    for name, param in model_GP_ODE.named_parameters():
        print('>> {}'.format(name))
        print('      {}\n'.format(param))
    print('---\n')

    optimizer = torch.optim.Adam([
        {'params': model_GP_ODE.parameters()}], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_GP, model_GP_ODE)
    for i in range(training_iter):
        optimizer.zero_grad()           # Zero gradients from previous iteration
        output = model_GP_ODE(train_x)      # Output from model
        loss = -mll(output, train_y)    # Calc loss and backprop gradients
        loss.backward()
        optimizer.step()

    model_GP_ODE.eval()
    likelihood_GP.eval()

    # ------ Plots
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x=destandardize_data(torch.from_numpy(u)), y=v, marker='o', color='k', s=10.)
    with torch.no_grad():
        u_line = np.linspace(u.min(), u.max(), 200)
        f_preds = model_GP_ODE(torch.from_numpy(u_line).float())
        f_mean = f_preds.mean
        f_var = f_preds.variance
        ax.plot(destandardize_data(torch.from_numpy(u_line)), gpu_numpy_detach(f_mean), label=model_GP_ODE.name)
        lower, upper = f_mean - 2. * f_var, f_mean + 2. * f_var
        ax.fill_between(destandardize_data(torch.from_numpy(u_line)), gpu_numpy_detach(lower), gpu_numpy_detach(upper), alpha=0.2)
    ax.set_ylim(np.min(v) - 1e-4, np.max(v) + 1e-4)
    ax.set_xlabel('PiB cDVR', fontsize=12)
    ax.set_ylabel('Slope of PiB cDVR (1/year)', fontsize=10)
    # ax.set_title('GP regression on ODE function')
    ax.legend()
    plt.gray()
    plt.savefig(os.path.join(args.snapshots_path, 'GP_fit.png'), pad_inches=0.1, bbox_inches='tight')
    plt.close()

    # ==================================================================================================================
    # ALIGN TRAJECTORIES
    # ==================================================================================================================

    timesteps = 200
    idx_33, idx_34, idx_44, t_line, reference_trajectory, t_ref, initial_conditions, initial_dtau, \
    time_values, data_values = align_trajectories(all_loader, model_GP_ODE, preprocessing=args.preprocessing,
                                                  pib_threshold=restandardize_data(args.pib_threshold),
                                                  timesteps=timesteps, nb_var=5.)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(destandardize_time(torch.from_numpy(t_line)) - destandardize_time(torch.from_numpy(np.array([t_ref]))),
            destandardize_data(torch.from_numpy(reference_trajectory)),
             '-', c='r', linewidth=1., label='reference trajectory')
    ax.axhline(y=args.pib_threshold, label='reference threshold')
    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf
    for data, time, tau in zip(data_values, time_values, initial_dtau):
        timeaxe = destandardize_time(torch.from_numpy(time + tau)) - destandardize_time(torch.from_numpy(np.array([t_ref])))
        ax.plot(timeaxe, destandardize_data(torch.from_numpy(data)), '-.', linewidth=.8)
        xmin = min(xmin, timeaxe.min())
        xmax = max(xmax, timeaxe.max())
        ymin = min(ymin, destandardize_data(torch.from_numpy(data)).min())
        ymax = max(ymax, destandardize_data(torch.from_numpy(data)).max())
    ax.set_xlim(xmin - .1 * (xmax - xmin), xmax + .1 * (xmax - xmin))
    ax.set_ylim(ymin - .1 * (ymax - ymin), ymax + .1 * (ymax - ymin))
    ax.set_xlabel('Time from PiB positivity (years)', fontsize=12)
    ax.set_ylabel('PiB cDVR', fontsize=12)
    ax.legend()
    # ax.set_title('Aligned trajectories on common evolution')
    plt.savefig(os.path.join(args.snapshots_path, 'Sequences_aligned.png'), pad_inches=0.5)
    plt.close()

    # ==================================================================================================================
    # (WILCOXON) RANK TEST ON APOE PAIRS
    # ==================================================================================================================

    boxplot_stock_pib = []
    boxplot_stock_age = []
    for ic, name in zip([idx_33, idx_34, idx_44], ['APOE 33', 'APOE 34', 'APOE 44']):
        age_tref = int(gpu_numpy_detach(destandardize_time(torch.from_numpy(np.array([t_ref]))))[0])
        age_pibpos = gpu_numpy_detach(destandardize_time(torch.from_numpy(t_ref - np.array(initial_dtau)[ic])))
        pib_tref = gpu_numpy_detach(destandardize_data(torch.from_numpy(np.array(initial_conditions)[ic])))
        boxplot_stock_pib.append(pib_tref)
        boxplot_stock_age.append(age_pibpos)
        print('{}'.format(name))
        print('          age at PiB positive :     {:.2f} +- {:.2f}'.format(np.mean(age_pibpos), np.std(age_pibpos)))
        print('          PiB at reference age {} : {:.2f} +- {:.2f}'.format(age_tref, np.mean(pib_tref),
                                                                            np.std(pib_tref)))

    # -------- compute wilcowon scores
    wilcox_age_01 = ranksums(x=np.array(initial_dtau)[idx_33].squeeze(),
                             y=np.array(initial_dtau)[idx_34].squeeze())
    wilcox_age_02 = ranksums(x=np.array(initial_dtau)[idx_33].squeeze(),
                             y=np.array(initial_dtau)[idx_44].squeeze())
    wilcox_age_12 = ranksums(x=np.array(initial_dtau)[idx_34].squeeze(),
                             y=np.array(initial_dtau)[idx_44].squeeze())
    wilcox_pib_01 = ranksums(x=np.array(initial_conditions)[idx_33].squeeze(),
                             y=np.array(initial_conditions)[idx_34].squeeze())
    wilcox_pib_02 = ranksums(x=np.array(initial_conditions)[idx_33].squeeze(),
                             y=np.array(initial_conditions)[idx_44].squeeze())
    wilcox_pib_12 = ranksums(x=np.array(initial_conditions)[idx_34].squeeze(),
                             y=np.array(initial_conditions)[idx_44].squeeze())

    print('\n')
    print('Wilcoxon test p-value for difference in age at PIB positive between {} and {} = {:.3f}'.format('APOE 33',
                                                                                                          'APOE 34',
                                                                                                          wilcox_age_01.pvalue))
    print('Wilcoxon test p-value for difference in PIB (at ref age) between {} and {} = {:.3f}'.format('APOE 33',
                                                                                                       'APOE 34',
                                                                                                       wilcox_pib_01.pvalue))

    print('-----')
    print('Wilcoxon test p-value for difference in age at PIB positive between {} and {} = {:.3f}'.format('APOE 33',
                                                                                                          'APOE 44',
                                                                                                          wilcox_age_02.pvalue))
    print('Wilcoxon test p-value for difference in PIB (at ref age) between {} and {} = {:.3f}'.format('APOE 33',
                                                                                                       'APOE 44',
                                                                                                       wilcox_pib_02.pvalue))

    print('-----')
    print('Wilcoxon test p-value for difference in age at PIB positive between {} and {} = {:.3f}'.format('APOE 34',
                                                                                                          'APOE 44',
                                                                                                          wilcox_age_12.pvalue))
    print('Wilcoxon test p-value for difference in PIB (at ref age) between {} and {} = {:.3f}'.format('APOE 34',
                                                                                                       'APOE 44',
                                                                                                       wilcox_pib_12.pvalue))

    print('\n')

    fig = plt.subplots(figsize=(5, 5))
    plt.boxplot(boxplot_stock_age)
    plt.xticks([i for i in range(1, 4)], ['APOE 33', 'APOE 34', 'APOE 44'], rotation=60, fontsize=8)
    plt.ylabel('Estimated age at onset of PiB positivity by APOE genotype', fontsize=10)
    plt.savefig(os.path.join(args.snapshots_path, 'boxplots_age.png'), pad_inches=1.)
    plt.close()


if __name__ == '__main__':
    run(args, device=DEVICE)

