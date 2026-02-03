import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def empirical_cross_cov(A_list, B_list):
    """
    A_list: list of flattened arrays (N_samples, D1)
    B_list: list of flattened arrays (N_samples, D2)
    Returns: cross-covariance matrix (D1, D2)
    """
    A = np.stack(A_list, axis=0)
    B = np.stack(B_list, axis=0)
    A_mean = A.mean(axis=0, keepdims=True)
    B_mean = B.mean(axis=0, keepdims=True)
    A_centered = A - A_mean
    B_centered = B - B_mean
    cov = (A_centered.T @ B_centered) / (A.shape[0] - 1)
    return cov

def plot_heatmap(matrix, title, xlabel, ylabel, save_path=None):
    plt.figure(figsize=(8,6))
    sns.heatmap(matrix, cmap='coolwarm', center=0)
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# ---- USER INPUT ----
parent_folder = '/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/sigma_y_set2/sigma_y_10e_3/results'  # Set this
save_heatmaps = '/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/sigma_y_set2/sigma_y_10e_3/heatmaps'       # Set this
os.makedirs(save_heatmaps, exist_ok=True)
# --------------------

# Load data
with open(os.path.join(parent_folder, 'post_train_data.pkl'), 'rb') as f:
    post_train_data = pickle.load(f)

A_mn_iter = post_train_data['A_mn_iter']
theta_iter = post_train_data['theta_iter']
X_vfl_iter = post_train_data['X_vfl_iter']

no_runs = post_train_data['number_of_runs']
epoch = post_train_data['epoch']
training_time = post_train_data['training_time']
p_vec = post_train_data['p_vec']
M = p_vec.shape[0]

# Use last epoch and last training_time index
last_epoch = f'epoch_{epoch}'
last_time = f'{training_time}'

# Loop over all off-diagonal A_{mn}
for m in range(M):
    for n in range(M):
        if m == n:
            continue  # skip diagonal

        A_mn_key = f'{m+1}{n+1}'
        theta_key = f'{m+1}'

        A_mn_samples = []
        theta_samples = []
        xvfl_samples = []

        for r in range(no_runs):
            run_key = f'run_{r+1}'

            # A_{mn} at last epoch and time
            A_mn = A_mn_iter[run_key][last_epoch][last_time][A_mn_key].flatten()
            # theta_m at last epoch and time
            theta = theta_iter[run_key][last_epoch][theta_key][:,:,training_time-1].flatten()
            # x_vfl_m at last epoch and time
            x_vfl = X_vfl_iter[run_key][last_epoch][theta_key][:,training_time-1].flatten()

            A_mn_samples.append(A_mn)
            theta_samples.append(theta)
            xvfl_samples.append(x_vfl)

        # Covariances
        cov_A_theta = empirical_cross_cov(A_mn_samples, theta_samples)
        cov_A_xvfl = empirical_cross_cov(A_mn_samples, xvfl_samples)

        # Plot and save with LaTeX labels (removed \bigl and \bigr)
        plot_heatmap(
            cov_A_theta,
            r'$\,\mathrm{Cov}(a_{%d%d}^t,\,v_{%d})$' % (m+1, n+1, m+1),
            r'$v_{%d}$' % (m+1),
            r'$a_{%d%d}$' % (m+1, n+1),
            os.path.join(save_heatmaps, f'cov_A{m+1}{n+1}_theta{m+1}.png')
        )
        plot_heatmap(
            cov_A_xvfl,
            r'$\Gamma_{%d%d}^t \;:=\; \mathrm{Cov}(a_{%d%d}^t,\hat h_{%d,a}^t)$' % (m+1, n+1, m+1, n+1, m+1),
            r'$\hat h_{%d,a}$' % (m+1),
            r'$a_{%d%d}$' % (m+1, n+1),
            os.path.join(save_heatmaps, f'cov_A{m+1}{n+1}_xvfl{m+1}.png')
        )

# Cov(theta_m, x_vfl_m) for all components
for m in range(M):
    theta_key = f'{m+1}'
    theta_samples = []
    xvfl_samples = []
    for r in range(no_runs):
        run_key = f'run_{r+1}'
        theta = theta_iter[run_key][last_epoch][theta_key][:,:,training_time-1].flatten()
        x_vfl = X_vfl_iter[run_key][last_epoch][theta_key][:,training_time-1].flatten()
        theta_samples.append(theta)
        xvfl_samples.append(x_vfl)
    cov_theta_xvfl = empirical_cross_cov(theta_samples, xvfl_samples)
    plot_heatmap(
        cov_theta_xvfl,
        r'$\Lambda_{%d} \;=\;\mathrm{Cov}(v_{%d},\;\hat h_{%d,a})$' % (m+1, m+1, m+1),
        r'$\hat h_{%d,a}$' % (m+1),
        r'$v_{%d}$' % (m+1),
        os.path.join(save_heatmaps, f'cov_theta{m+1}_xvfl{m+1}.png')
    )