import os
import matplotlib.pyplot as plt
import pickle

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Colorblind-friendly palette (8 colors)
colorblind_colors = [
    "#f781bf",  # pink
    "#377eb8",  # blue
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#4daf4a",  # green
    "#a65628",  # brown
    "#ffff33",  # yellow
    "#e41a1c",  # red
]

# parent_folder = r'/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/sigma_y_set2'
# parent_folder = r'/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/sigma_Amn_set1'
# parent_folder = r'/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/sigma_theta_set1'

# parent_folder = r'/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/scalability/obs_dimension/dim_128/sigma_Amn_set1'
parent_folder = r'/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/scalability/num_components/comp16/sigma_Amn_set1'

# Load the results data
results_path = os.path.join(parent_folder, 'summary_plots', 'summary_results.pkl')
with open(results_path, 'rb') as f:
    summary_data = pickle.load(f)

# Create folder for saving the plots
summary_plots_folder = os.path.join(parent_folder, 'summary_plots_6420')
os.makedirs(summary_plots_folder, exist_ok=True)

# Define subfolders to plot and their corresponding labels
# selected_subfolders = ['sigma_y_10e_6', 'sigma_y_10e_5', 'sigma_y_10e_4', 'sigma_y_10e_3', 'sigma_y_10e_2', 'sigma_y_10e_1', 'sigma_y_10e0', 'sigma_y_10e1']
# selected_subfolders = ['sigma_y_10e_6', 'sigma_y_10e_4', 'sigma_y_10e_2', 'sigma_y_10e0']

# selected_subfolders = ['Amn_10e_6', 'Amn_10e_5', 'Amn_10e_4', 'Amn_10e_3', 'Amn_10e_2', 'Amn_10e_1', 'Amn_10e0', 'Amn_10e1']
# selected_subfolders = ['Amn_10e_6', 'Amn_10e_5', 'Amn_10e_4', 'Amn_10e_3', 'Amn_10e_2', 'Amn_10e_1', 'Amn_10e0']
# selected_subfolders = ['Amn_10e_6', 'Amn_10e_5', 'Amn_10e_4', 'Amn_10e_3', 'Amn_10e_2', 'Amn_10e_1']
# selected_subfolders = ['Amn_10e_6', 'Amn_10e_5', 'Amn_10e_4', 'Amn_10e_3', 'Amn_10e_2']
# selected_subfolders = ['Amn_10e_6', 'Amn_10e_5', 'Amn_10e_4', 'Amn_10e_3']
# selected_subfolders = ['Amn_10e_6', 'Amn_10e_5', 'Amn_10e_4']
selected_subfolders = ['Amn_10e_6', 'Amn_10e_4', 'Amn_10e_2', 'Amn_10e0']

# selected_subfolders = ['theta_m_10e_6', 'theta_m_10e_5', 'theta_m_10e_4', 'theta_m_10e_3', 'theta_m_10e_2', 'theta_m_10e_1', 'theta_m_10e0', 'theta_m_10e1']
# selected_subfolders = ['theta_m_10e_6', 'theta_m_10e_4', 'theta_m_10e_2', 'theta_m_10e0']

# subfolder_labels = [
#     r'$\Sigma_{y_m}^t \sim 10^{-6}$',
#     r'$\Sigma_{y_m}^t \sim 10^{-5}$',
#     r'$\Sigma_{y_m}^t \sim 10^{-4}$',
#     r'$\Sigma_{y_m}^t \sim 10^{-3}$',
#     r'$\Sigma_{y_m}^t \sim 10^{-2}$',
#     r'$\Sigma_{y_m}^t \sim 10^{-1}$',
#     r'$\Sigma_{y_m}^t \sim 10^{0}$',
#     r'$\Sigma_{y_m}^t \sim 10^{1}$',
# ]
# subfolder_labels = [
#     r'$\Sigma_{y_m}^t \sim 10^{-6}$',
#     r'$\Sigma_{y_m}^t \sim 10^{-4}$',
#     r'$\Sigma_{y_m}^t \sim 10^{-2}$',
#     r'$\Sigma_{y_m}^t \sim 10^{0}$'
# ]

# subfolder_labels = [
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-6}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-5}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-4}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-3}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-2}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-1}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{0}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{1}$'
# ]
# subfolder_labels = [
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-6}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-5}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-4}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-3}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-2}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-1}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{0}$'
# ]
# subfolder_labels = [
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-6}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-5}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-4}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-3}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-2}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-1}$'
# ]
# subfolder_labels = [
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-6}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-5}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-4}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-3}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-2}$'
# ]
# subfolder_labels = [
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-6}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-5}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-4}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-3}$'
# ]
# subfolder_labels = [
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-6}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-5}$',
#     r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-4}$'
# ]

subfolder_labels = [
    r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-6}$',
    r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-4}$',
    r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{-2}$',
    r'$\Sigma_{\hat{A}_{mn}}^0 \sim 10^{0}$'
]

# subfolder_labels = [
#     r'$\Sigma_{\theta_m}^0 \sim 10^{-6}$',
#     r'$\Sigma_{\theta_m}^0 \sim 10^{-4}$',
#     r'$\Sigma_{\theta_m}^0 \sim 10^{-2}$',
#     r'$\Sigma_{\theta_m}^0 \sim 10^{0}$'
# ]

# subfolder_labels = [
#     r'$\Sigma_{\theta_m}^0 \sim 10^{-6}$',
#     r'$\Sigma_{\theta_m}^0 \sim 10^{-5}$',
#     r'$\Sigma_{\theta_m}^0 \sim 10^{-4}$',
#     r'$\Sigma_{\theta_m}^0 \sim 10^{-3}$',
#     r'$\Sigma_{\theta_m}^0 \sim 10^{-2}$',
#     r'$\Sigma_{\theta_m}^0 \sim 10^{-1}$',
#     r'$\Sigma_{\theta_m}^0 \sim 10^{0}$',
#     r'$\Sigma_{\theta_m}^0 \sim 10^{1}$'
# ]

# 1. Plot Trace(Cov(Vec(A_mn))) vs Iteration for all off-diagonal blocks
for key in next(iter(summary_data['trace_cov_Amn_data'].values())).keys():
    plt.figure(figsize=(8, 6))
    for idx, (subfolder, label) in enumerate(zip(selected_subfolders, subfolder_labels)):
        if subfolder in summary_data['trace_cov_Amn_data']:
            plt.plot(
                summary_data['trace_cov_Amn_data'][subfolder][key],
                label=label,
                color=colorblind_colors[idx % len(colorblind_colors)]
            )

            # printing the final value of the trace covariance
            print(f"Final value of Trace Covariance for {key} in {subfolder}: {summary_data['trace_cov_Amn_data'][subfolder][key][-1]}")
    plt.xlabel('Number of iterations', fontsize=25)
    plt.ylabel(r'$\mathrm{Tr}(\Sigma_{\hat{A}_{' + key + '}^t})$', fontsize=25)
    plt.legend(fontsize=20)
    plt.grid(True)

    plot_path = os.path.join(summary_plots_folder, f'Trace_Cov_A_mn_{key}.pdf')
    plt.savefig(plot_path, format='pdf', dpi=800, bbox_inches='tight')
    plt.close()

# 2. Plot Largest Eigenvalues vs Iteration
for key in next(iter(summary_data['largest_eigenvalues_data'].values())).keys():
    plt.figure(figsize=(8, 6))
    for idx, (subfolder, label) in enumerate(zip(selected_subfolders, subfolder_labels)):
        if subfolder in summary_data['largest_eigenvalues_data']:
            plt.plot(
                summary_data['largest_eigenvalues_data'][subfolder][key],
                label=label,
                color=colorblind_colors[idx % len(colorblind_colors)]
            )
    plt.xlabel('Number of iterations', fontsize=25)
    plt.ylabel(r'$\lambda_{\max}(\Sigma_{\hat{A}_{' + key + '}^t})$', fontsize=25)
    plt.legend(fontsize=20)
    plt.grid(True)

    plot_path = os.path.join(summary_plots_folder, f'Largest_Eigenvalue_Cov_A_mn_{key}.pdf')
    plt.savefig(plot_path, format='pdf', dpi=800, bbox_inches='tight')
    plt.close()

# 3. Plot Trace(Cov(Vec(Error))) vs Iteration
for key in next(iter(summary_data['trace_cov_error_data'].values())).keys():
    plt.figure(figsize=(8, 6))
    for idx, (subfolder, label) in enumerate(zip(selected_subfolders, subfolder_labels)):
        if subfolder in summary_data['trace_cov_error_data']:
            plt.plot(
                summary_data['trace_cov_error_data'][subfolder][key],
                label=label,
                color=colorblind_colors[idx % len(colorblind_colors)]
            )
    plt.xlabel('Number of iterations', fontsize=25)
    plt.ylabel(r'$\mathrm{Tr}(\Sigma_{\epsilon_{' + key + '}^t})$', fontsize=25)
    plt.legend(fontsize=20)
    plt.grid(True)

    plot_path = os.path.join(summary_plots_folder, f'Trace_Cov_Error_{key}.pdf')
    plt.savefig(plot_path, format='pdf', dpi=800, bbox_inches='tight')
    plt.close()

# 4. Plot L2 Norm of Averaged Error vs Iteration
for key in next(iter(summary_data['avg_norm_error_data'].values())).keys():
    plt.figure(figsize=(8, 6))
    for idx, (subfolder, label) in enumerate(zip(selected_subfolders, subfolder_labels)):
        if subfolder in summary_data['avg_norm_error_data']:
            plt.plot(
                summary_data['avg_norm_error_data'][subfolder][key],
                label=label,
                color=colorblind_colors[idx % len(colorblind_colors)]
            )
    plt.xlabel('Number of iterations', fontsize=25)
    plt.ylabel(r'$\|\mu_{\hat{A}_{' + key + '}^t} - A_{' + key + '}\|_2$', fontsize=25)
    plt.legend(fontsize=20)
    plt.grid(True)

    plot_path = os.path.join(summary_plots_folder, f'L2_Norm_Avg_Error_{key}.pdf')
    plt.savefig(plot_path, format='pdf', dpi=800, bbox_inches='tight')
    plt.close()

# 5. Plot Trace(Cov(Ls_gradx_iter)) vs Iteration
for m in range(len(next(iter(summary_data['trace_cov_gradx_data'].values())))):
    plt.figure(figsize=(8, 6))
    for idx, (subfolder, label) in enumerate(zip(selected_subfolders, subfolder_labels)):
        if subfolder in summary_data['trace_cov_gradx_data']:
            plt.plot(
                summary_data['trace_cov_gradx_data'][subfolder][f'{m+1}'],
                label=label,
                color=colorblind_colors[idx % len(colorblind_colors)]
            )
    plt.xlabel('Number of iterations', fontsize=25)
    plt.ylabel(r'$\mathrm{Var}(g_{' + str(m+1) + ',s}^t)$', fontsize=25)
    plt.legend(fontsize=20)
    plt.grid(True)

    plot_path = os.path.join(summary_plots_folder, f'Trace_Cov_Ls_gradx_iter_{m+1}.pdf')
    plt.savefig(plot_path, format='pdf', dpi=800, bbox_inches='tight')
    plt.close()

# 6. Plot Trace(Cov(theta)) vs Iteration
for m in range(len(next(iter(summary_data['trace_cov_theta_data'].values())))):
    plt.figure(figsize=(8, 6))
    for idx, (subfolder, label) in enumerate(zip(selected_subfolders, subfolder_labels)):
        if subfolder in summary_data['trace_cov_theta_data']:
            plt.plot(
                summary_data['trace_cov_theta_data'][subfolder][f'{m+1}'],
                label=label,
                color=colorblind_colors[idx % len(colorblind_colors)]
            )
    plt.xlabel('Number of iterations', fontsize=25)
    plt.ylabel(r'$\mathrm{Tr}(\Sigma_{\theta_{' + str(m+1) + '}}^t)$', fontsize=25)
    plt.legend(fontsize=20)
    plt.grid(True)

    plot_path = os.path.join(summary_plots_folder, f'Trace_Cov_theta_{m+1}.pdf')
    plt.savefig(plot_path, format='pdf', dpi=800, bbox_inches='tight')
    plt.close()

# 7. Plot Trace(Cov(X_vfl_pred_iter)) vs Iteration
for m in range(len(next(iter(summary_data['trace_cov_X_vfl_pred_data'].values())))):
    plt.figure(figsize=(8, 6))
    for idx, (subfolder, label) in enumerate(zip(selected_subfolders, subfolder_labels)):
        if subfolder in summary_data['trace_cov_X_vfl_pred_data']:
            plt.plot(
                summary_data['trace_cov_X_vfl_pred_data'][subfolder][f'{m+1}'],
                label=label,
                color=colorblind_colors[idx % len(colorblind_colors)]
            )
    plt.xlabel('Number of iterations', fontsize=25)
    plt.ylabel(r'$\mathrm{Tr}(\Sigma_{h_' + str(m+1) + '}^t)$', fontsize=25)
    plt.legend(fontsize=20)
    plt.grid(True)

    plot_path = os.path.join(summary_plots_folder, f'Trace_Cov_X_vfl_pred_{m+1}.pdf')
    plt.savefig(plot_path, format='pdf', dpi=800, bbox_inches='tight')
    plt.close()

# 8. Plot Trace(CrossCov(A_mn_iter, theta_iter)) vs Iteration (off-diagonal only)
for key in next(iter(summary_data['cross_cov_A_theta'].values())).keys():
    plt.figure(figsize=(8, 6))
    for idx, (subfolder, label) in enumerate(zip(selected_subfolders, subfolder_labels)):
        if subfolder in summary_data['cross_cov_A_theta']:
            plt.plot(
                summary_data['cross_cov_A_theta'][subfolder][key],
                label=label,
                color=colorblind_colors[idx % len(colorblind_colors)]
            )
    plt.xlabel('Number of iterations', fontsize=25)
    plt.ylabel(r'$\Psi_{' + key + '}^t$', fontsize=25)
    plt.legend(fontsize=16)
    plt.grid(True)

    plot_path = os.path.join(summary_plots_folder, f'Trace_CrossCov_A_mn_theta_{key}.pdf')
    plt.savefig(plot_path, format='pdf', dpi=800, bbox_inches='tight')
    plt.close()

# 9. Plot Trace(CrossCov(A_mn_iter, X_vfl_pred_iter)) vs Iteration (off-diagonal only)
for key in next(iter(summary_data['cross_cov_A_Xvfl'].values())).keys():
    plt.figure(figsize=(8, 6))
    for idx, (subfolder, label) in enumerate(zip(selected_subfolders, subfolder_labels)):
        if subfolder in summary_data['cross_cov_A_Xvfl']:
            plt.plot(
                summary_data['cross_cov_A_Xvfl'][subfolder][key],
                label=label,
                color=colorblind_colors[idx % len(colorblind_colors)]
            )
    plt.xlabel('Number of iterations', fontsize=25)
    plt.ylabel(r'$\Gamma_{' + key + '}^t$', fontsize=25)
    plt.legend(fontsize=16)
    plt.grid(True)

    plot_path = os.path.join(summary_plots_folder, f'Trace_CrossCov_A_mn_Xvfl_{key}.pdf')
    plt.savefig(plot_path, format='pdf', dpi=800, bbox_inches='tight')
    plt.close()

# 10. Plot Trace(CrossCov(theta_iter, X_vfl_pred_iter)) vs Iteration
for m in range(len(next(iter(summary_data['cross_cov_theta_Xvfl'].values())))):
    plt.figure(figsize=(8, 6))
    for idx, (subfolder, label) in enumerate(zip(selected_subfolders, subfolder_labels)):
        if subfolder in summary_data['cross_cov_theta_Xvfl']:
            plt.plot(
                summary_data['cross_cov_theta_Xvfl'][subfolder][f'{m+1}'],
                label=label,
                color=colorblind_colors[idx % len(colorblind_colors)]
            )
    plt.xlabel('Number of iterations', fontsize=25)
    plt.ylabel(r'$\Lambda_{' + str(m+1) + '}^t$', fontsize=25)
    plt.legend(fontsize=16)
    plt.grid(True)

    plot_path = os.path.join(summary_plots_folder, f'Trace_CrossCov_theta_Xvfl_{m+1}.pdf')
    plt.savefig(plot_path, format='pdf', dpi=800, bbox_inches='tight')
    plt.close()

print("All plots have been generated and saved in the summary_plots folder.")