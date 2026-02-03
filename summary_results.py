import os
import numpy as np
import pickle

# Parent folder containing the subfolders
# parent_folder = r'/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/sigma_y_set2'
# parent_folder = r'/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/sigma_Amn_set1'
# parent_folder = r'/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/sigma_theta_set1'

# parent_folder = r'/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/scalability/obs_dimension/dim_128/sigma_Amn_set1'
parent_folder = r'/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/scalability/num_components/comp16/sigma_Amn_set1'

# Subfolder names
# subfolders = ['sigma_y_10e_6', 'sigma_y_10e_5', 'sigma_y_10e_4', 'sigma_y_10e_3', 'sigma_y_10e_2', 'sigma_y_10e_1', 'sigma_y_10e0', 'sigma_y_10e1']
# subfolders = ['Amn_10e_6', 'Amn_10e_5', 'Amn_10e_4', 'Amn_10e_3', 'Amn_10e_2', 'Amn_10e_1', 'Amn_10e0', 'Amn_10e1']
# subfolders = ['theta_m_10e_6', 'theta_m_10e_5', 'theta_m_10e_4', 'theta_m_10e_3', 'theta_m_10e_2', 'theta_m_10e_1', 'theta_m_10e0', 'theta_m_10e1']

subfolders = ['Amn_10e_6', 'Amn_10e_4', 'Amn_10e_2', 'Amn_10e0']

# Initialize dictionary to store all results
summary_data = {
    'parent_folder': parent_folder,
    'trace_cov_Amn_data': {},
    'element_data': {},
    'largest_eigenvalues_data': {},
    'trace_cov_error_data': {},
    'avg_norm_error_data': {},
    'trace_cov_gradx_data': {},
    'trace_cov_theta_data': {},
    'trace_cov_X_vfl_pred_data': {},
    'cross_cov_A_theta': {},
    'cross_cov_A_Xvfl': {},
    'cross_cov_theta_Xvfl': {},
}

# Process each subfolder once
for subfolder in subfolders:
    pkl_path = os.path.join(parent_folder, subfolder, 'results', 'post_train_data.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            post_train_data = pickle.load(f)

        # Extract common data
        p_vec = post_train_data['p_vec']
        K = post_train_data['total_iterations']
        no_runs = post_train_data['number_of_runs']
        training_time = post_train_data['training_time']
        epoch = post_train_data['epoch']
        M = len(p_vec)

        # Initialize data structures for this subfolder
        summary_data['trace_cov_Amn_data'][subfolder] = {}
        summary_data['largest_eigenvalues_data'][subfolder] = {}
        
        # 1. Process Cov(A_mn) data for all off-diagonal blocks
        cov_A_mn = post_train_data['cov_A_mn']
        for m in range(M):
            for n in range(M):
                if m != n:
                    key = f'{m+1}{n+1}'
                    if key in cov_A_mn:
                        trace_cov = np.zeros(K)
                        largest_eig = np.zeros(K)
                        for k in range(K):
                            cov_matrix = cov_A_mn[key][:, :, k]
                            trace_cov[k] = np.trace(cov_matrix)
                            largest_eig[k] = np.max(np.linalg.eigvalsh(cov_matrix))


                        summary_data['trace_cov_Amn_data'][subfolder][key] = trace_cov
                        summary_data['largest_eigenvalues_data'][subfolder][key] = largest_eig

        # 2. Process error matrices
        A_complete_true = post_train_data['A_complete_true']
        p_start_idx = post_train_data['p_start_idx']
        
        trace_cov_error = {f'{m+1}{n+1}': np.zeros(K) 
                          for m in range(M) 
                          for n in range(M) if m != n}
        avg_norm_error = {f'{m+1}{n+1}': np.zeros(K) 
                         for m in range(M) 
                         for n in range(M) if m != n}

        for m in range(M):
            for n in range(M):
                if m == n:
                    continue

                # Extract true A_mn block
                row_start = p_start_idx[m, 0]
                row_end = row_start + p_vec[m, 0]
                col_start = p_start_idx[n, 0]
                col_end = col_start + p_vec[n, 0]
                A_mn_true = A_complete_true[row_start:row_end, col_start:col_end]

                error_matrices = np.zeros((p_vec[m, 0] * p_vec[n, 0], K, no_runs))
                for r in range(no_runs):
                    for e in range(epoch):
                        for t in range(training_time):
                            iteration_idx = e * training_time + t
                            A_mn_est = post_train_data['A_mn_iter'][f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}'][f'{m+1}{n+1}']
                            error_matrix = A_mn_true - A_mn_est
                            error_matrices[:, iteration_idx, r] = error_matrix.flatten()

                avg_error = np.mean(error_matrices, axis=2)
                for k in range(K):
                    avg_norm_error[f'{m+1}{n+1}'][k] = np.linalg.norm(avg_error[:, k], ord=2)
                    cov_matrix = np.cov(error_matrices[:, k, :], rowvar=False)
                    trace_cov_error[f'{m+1}{n+1}'][k] = np.trace(cov_matrix)

        summary_data['trace_cov_error_data'][subfolder] = trace_cov_error
        summary_data['avg_norm_error_data'][subfolder] = avg_norm_error

        # 3. Process gradient data
        trace_cov_gradx = {f'{m+1}': np.zeros(K) for m in range(M)}
        for m in range(M):
            gradx_matrices = np.zeros((p_vec[m, 0], K, no_runs))
            for r in range(no_runs):
                for e in range(epoch):
                    for t in range(training_time):
                        iteration_idx = e * training_time + t
                        gradx_matrices[:, iteration_idx, r] = post_train_data['Ls_gradx_iter'][f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}'][f'{m+1}'][:, 0]

            for k in range(K):
                cov_matrix = np.cov(gradx_matrices[:, k, :], rowvar=False)
                trace_cov_gradx[f'{m+1}'][k] = np.trace(cov_matrix)

        summary_data['trace_cov_gradx_data'][subfolder] = trace_cov_gradx

        # 4. Process theta data
        trace_cov_theta = {f'{m+1}': np.zeros(K) for m in range(M)}
        for m in range(M):
            theta_matrices = np.zeros((p_vec[m, 0] * post_train_data['d_vec'][m, 0], K, no_runs))
            for r in range(no_runs):
                for e in range(epoch):
                    for t in range(training_time):
                        iteration_idx = e * training_time + t
                        theta_matrices[:, iteration_idx, r] = post_train_data['theta_iter'][f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:,:,t].flatten()

            for k in range(K):
                cov_matrix = np.cov(theta_matrices[:, k, :], rowvar=False)
                trace_cov_theta[f'{m+1}'][k] = np.trace(cov_matrix)

        summary_data['trace_cov_theta_data'][subfolder] = trace_cov_theta

        # 5. Process X_vfl_pred_iter data (trace of empirical covariance over runs)
        trace_cov_X_vfl_pred = {f'{m+1}': np.zeros(K) for m in range(M)}
        for m in range(M):
            # Each X_vfl_pred is (p_vec[m, 0], training_time)
            x_pred_matrices = np.zeros((p_vec[m, 0], K, no_runs))
            for r in range(no_runs):
                for e in range(epoch):
                    for t in range(training_time):
                        iteration_idx = e * training_time + t
                        # X_vfl_pred_iter: [run][epoch][component][:, t]
                        x_pred_matrices[:, iteration_idx, r] = post_train_data['X_vfl_pred_iter'][f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:, t]
            for k in range(K):
                cov_matrix = np.cov(x_pred_matrices[:, k, :], rowvar=False)
                trace_cov_X_vfl_pred[f'{m+1}'][k] = np.trace(cov_matrix)
        summary_data['trace_cov_X_vfl_pred_data'][subfolder] = trace_cov_X_vfl_pred

        # --- Cross-covariances ---

        # 1. Cov(A_mn_iter, theta_iter) for all off-diagonal (m, n), with theta_m
        cross_cov_A_theta = {f'{m+1}{n+1}': np.zeros(K) for m in range(M) for n in range(M) if m != n}
        for m in range(M):
            for n in range(M):
                if m == n:
                    continue
                key = f'{m+1}{n+1}'
                A_dim = p_vec[m, 0] * p_vec[n, 0]
                theta_dim = p_vec[m, 0] * post_train_data['d_vec'][m, 0]
                A_mn_matrix = np.zeros((A_dim, K, no_runs))
                theta_matrix = np.zeros((theta_dim, K, no_runs))
                for r in range(no_runs):
                    for e in range(epoch):
                        for t in range(training_time):
                            iteration_idx = e * training_time + t
                            A_mn_matrix[:, iteration_idx, r] = post_train_data['A_mn_iter'][f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}'][key].flatten()
                            theta_matrix[:, iteration_idx, r] = post_train_data['theta_iter'][f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:,:,t].flatten()
                for k in range(K):
                    if no_runs > 1:
                        cross_cov = np.cov(A_mn_matrix[:, k, :], theta_matrix[:, k, :], rowvar=True)
                        cross_block = cross_cov[:A_dim, A_dim:]
                        cross_cov_A_theta[key][k] = np.trace(cross_block)
                    else:
                        cross_cov_A_theta[key][k] = np.nan
        summary_data['cross_cov_A_theta'][subfolder] = cross_cov_A_theta

        # 2. Cov(A_mn_iter, X_vfl_pred_iter) for all off-diagonal (m, n), with X_vfl_pred_m
        cross_cov_A_Xvfl = {f'{m+1}{n+1}': np.zeros(K) for m in range(M) for n in range(M) if m != n}
        for m in range(M):
            for n in range(M):
                if m == n:
                    continue
                key = f'{m+1}{n+1}'
                A_dim = p_vec[m, 0] * p_vec[n, 0]
                x_dim = p_vec[m, 0]
                A_mn_matrix = np.zeros((A_dim, K, no_runs))
                x_pred_matrix = np.zeros((x_dim, K, no_runs))
                for r in range(no_runs):
                    for e in range(epoch):
                        for t in range(training_time):
                            iteration_idx = e * training_time + t
                            A_mn_matrix[:, iteration_idx, r] = post_train_data['A_mn_iter'][f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}'][key].flatten()
                            x_pred_matrix[:, iteration_idx, r] = post_train_data['X_vfl_pred_iter'][f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:, t]
                for k in range(K):
                    if no_runs > 1:
                        cross_cov = np.cov(A_mn_matrix[:, k, :], x_pred_matrix[:, k, :], rowvar=True)
                        cross_block = cross_cov[:A_dim, A_dim:]
                        cross_cov_A_Xvfl[key][k] = np.trace(cross_block)
                    else:
                        cross_cov_A_Xvfl[key][k] = np.nan
        summary_data['cross_cov_A_Xvfl'][subfolder] = cross_cov_A_Xvfl

        # 3. Cov(theta_iter, X_vfl_pred_iter) for all m
        cross_cov_theta_Xvfl = {f'{m+1}': np.zeros(K) for m in range(M)}
        for m in range(M):
            theta_dim = p_vec[m, 0] * post_train_data['d_vec'][m, 0]
            x_dim = p_vec[m, 0]
            theta_matrix = np.zeros((theta_dim, K, no_runs))
            x_pred_matrix = np.zeros((x_dim, K, no_runs))
            for r in range(no_runs):
                for e in range(epoch):
                    for t in range(training_time):
                        iteration_idx = e * training_time + t
                        theta_matrix[:, iteration_idx, r] = post_train_data['theta_iter'][f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:,:,t].flatten()
                        x_pred_matrix[:, iteration_idx, r] = post_train_data['X_vfl_pred_iter'][f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:, t]
            for k in range(K):
                if no_runs > 1:
                    cross_cov = np.cov(theta_matrix[:, k, :], x_pred_matrix[:, k, :], rowvar=True)
                    cross_block = cross_cov[:theta_dim, theta_dim:]
                    cross_cov_theta_Xvfl[f'{m+1}'][k] = np.trace(cross_block)
                else:
                    cross_cov_theta_Xvfl[f'{m+1}'][k] = np.nan
        summary_data['cross_cov_theta_Xvfl'][subfolder] = cross_cov_theta_Xvfl

    else:
        print(f"File not found: {pkl_path}")

# Create summary_plots directory if it doesn't exist
summary_plots_dir = os.path.join(parent_folder, 'summary_plots')
os.makedirs(summary_plots_dir, exist_ok=True)

# Save all computed results
results_path = os.path.join(summary_plots_dir, 'summary_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(summary_data, f)