import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Parent folder containing the subfolders
parent_folder = r'/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/sigma_y_set2'  # Replace with the actual path

# Subfolder names
# subfolders = ['proc_noise_10e_6', 'proc_noise_10e_5', 'proc_noise_10e_4', 'proc_noise_10e_3', 'proc_noise_10e_2', 'proc_noise_10e_1', 'proc_noise_10e0', 'proc_noise_10e1']

subfolders = ['sigma_y_10e_6', 'sigma_y_10e_5', 'sigma_y_10e_4', 'sigma_y_10e_3', 'sigma_y_10e_2', 'sigma_y_10e_1', 'sigma_y_10e0', 'sigma_y_10e1']
# Create a folder for saving summary plots
summary_plots_folder = os.path.join(parent_folder, 'summary_plots')
os.makedirs(summary_plots_folder, exist_ok=True)

# 1. Plot Trace(Cov(Vec(A_mn_21))) vs Iteration
trace_data = {}

for subfolder in subfolders:
    csv_path = os.path.join(parent_folder, subfolder, 'results', 'Trace_Cov_A_mn_21.csv')
    if os.path.exists(csv_path):
        trace_data[subfolder] = np.loadtxt(csv_path, delimiter=',')
    else:
        print(f"File not found: {csv_path}")

# Plot the data
plt.figure(figsize=(10, 6))
for subfolder, data in trace_data.items():
    plt.plot(data, label=subfolder)

# Add labels, title, and legend
plt.xlabel('Iteration')
plt.ylabel('Trace(Cov(Vec(A_mn_21)))')
plt.title('Trace(Cov(Vec(A_mn_21))) vs Iteration for Different Observation Noises')
plt.legend()
plt.grid(True)

# Save the plot
plot_path = os.path.join(summary_plots_folder, 'Trace_Cov_A_mn_21.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# 2. Plot Cov(A_mn_21)[0, 0] vs Iteration
element_data = {}

for subfolder in subfolders:
    csv_path = os.path.join(parent_folder, subfolder, 'results', 'Cov_A_mn_21_element_0_0.csv')
    if os.path.exists(csv_path):
        element_data[subfolder] = np.loadtxt(csv_path, delimiter=',')
    else:
        print(f"File not found: {csv_path}")

# Plot the data
plt.figure(figsize=(10, 6))
for subfolder, data in element_data.items():
    plt.plot(data, label=subfolder)

# Add labels, title, and legend
plt.xlabel('Iteration')
plt.ylabel('Cov(A_mn_21)[0, 0]')
plt.title('Evolution of Cov(A_mn_21)[0, 0] for Different Observation Noises')
plt.legend()
plt.grid(True)

# Save the plot
plot_path = os.path.join(summary_plots_folder, 'Cov_A_mn_21_element_0_0.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# 3. Plot Largest Eigenvalue of Cov(A_mn) vs Iteration
largest_eigenvalues_data = {}

for subfolder in subfolders:
    pkl_path = os.path.join(parent_folder, subfolder, 'results', 'post_train_data.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            post_train_data = pickle.load(f)
        
        cov_A_mn = post_train_data['cov_A_mn']
        largest_eigenvalues_data[subfolder] = {}

        # Compute the largest eigenvalue for each off-diagonal matrix
        for key, cov_matrices in cov_A_mn.items():
            if cov_matrices.shape[0] == cov_matrices.shape[1]:
                largest_eigenvalues_data[subfolder][key] = [
                    np.max(np.linalg.eigvalsh(cov_matrices[:, :, i])) for i in range(cov_matrices.shape[2])
                ]
            else:
                print(f"Skipping {key} in {subfolder}: Covariance matrices are not square.")
    else:
        print(f"File not found: {pkl_path}")

# Plot the largest eigenvalue for each off-diagonal covariance matrix in the same figure
for key in cov_A_mn.keys():
    plt.figure(figsize=(10, 6))
    for subfolder, eigenvalues_dict in largest_eigenvalues_data.items():
        if key in eigenvalues_dict:
            plt.plot(eigenvalues_dict[key], label=f'{subfolder}')
    plt.xlabel('Iteration')
    plt.ylabel(f'Largest Eigenvalue of Cov(A_mn_{key})')
    plt.title(f'Largest Eigenvalue of Cov(A_mn_{key}) vs Iteration for Different Observation Noises')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(summary_plots_folder, f'Largest_Eigenvalue_Cov_A_mn_{key}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

# # 4. Plot Trace(Cov(Vec(Error))) vs Iteration
# trace_cov_error_data = {}

# for subfolder in subfolders:
#     pkl_path = os.path.join(parent_folder, subfolder, 'results', 'post_train_data.pkl')
#     if os.path.exists(pkl_path):
#         with open(pkl_path, 'rb') as f:
#             post_train_data = pickle.load(f)

#         A_complete_true = post_train_data['A_complete_true']
#         A_mn_iter = post_train_data['A_mn_iter']
#         p_vec = post_train_data['p_vec']
#         p_start_idx = post_train_data['p_start_idx']
#         K = post_train_data['total_iterations']
#         no_runs = post_train_data['number_of_runs']
#         training_time = post_train_data['training_time']
#         epoch = post_train_data['epoch']

#         trace_cov_error = {f'{m+1}{n+1}': np.zeros(K) for m in range(len(p_vec)) for n in range(len(p_vec)) if m != n}

#         for m in range(len(p_vec)):
#             for n in range(len(p_vec)):
#                 if m == n:
#                     continue

#                 row_start = p_start_idx[m, 0]
#                 row_end = row_start + p_vec[m, 0]
#                 col_start = p_start_idx[n, 0]
#                 col_end = col_start + p_vec[n, 0]
#                 A_mn_true = A_complete_true[row_start:row_end, col_start:col_end]

#                 error_matrices = np.zeros((p_vec[m, 0] * p_vec[n, 0], K, no_runs))
#                 for r in range(no_runs):
#                     for e in range(epoch):
#                         for t in range(training_time):
#                             iteration_idx = e * training_time + t
#                             A_mn_est = A_mn_iter[f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}'][f'{m+1}{n+1}']
#                             error_matrix = A_mn_true - A_mn_est
#                             error_matrices[:, iteration_idx, r] = error_matrix.flatten()

#                 for k in range(K):
#                     cov_matrix = np.cov(error_matrices[:, k, :], rowvar=False)
#                     trace_cov_error[f'{m+1}{n+1}'][k] = np.trace(cov_matrix)

#         trace_cov_error_data[subfolder] = trace_cov_error
#     else:
#         print(f"File not found: {pkl_path}")

# for key in trace_cov_error_data[subfolders[0]].keys():
#     plt.figure(figsize=(10, 6))
#     for subfolder, trace_cov_error in trace_cov_error_data.items():
#         plt.plot(trace_cov_error[key], label=f'{subfolder}')
#     plt.xlabel('Iteration')
#     plt.ylabel(f'Trace(Cov(Vec(Error_{key})))')
#     plt.title(f'Trace(Cov(Vec(Error_{key}))) vs Iteration for Different Observation Noises')
#     plt.legend()
#     plt.grid(True)

#     # Save the plot
#     plot_path = os.path.join(summary_plots_folder, f'Trace_Cov_Error_{key}.png')
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.close()


# Trace of covariance of error matrices and Averaged error norm

trace_cov_error_data = {}
avg_norm_error_data = {}


for subfolder in subfolders:
    pkl_path = os.path.join(parent_folder, subfolder, 'results', 'post_train_data.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            post_train_data = pickle.load(f)

        A_complete_true = post_train_data['A_complete_true']
        A_mn_iter = post_train_data['A_mn_iter']
        p_vec = post_train_data['p_vec']
        p_start_idx = post_train_data['p_start_idx']
        K = post_train_data['total_iterations']
        no_runs = post_train_data['number_of_runs']
        training_time = post_train_data['training_time']
        epoch = post_train_data['epoch']

        # Initialize dictionaries to store trace of covariance and norm of averaged error
        trace_cov_error = {f'{m+1}{n+1}': np.zeros(K) for m in range(len(p_vec)) for n in range(len(p_vec)) if m != n}
        avg_norm_error = {f'{m+1}{n+1}': np.zeros(K) for m in range(len(p_vec)) for n in range(len(p_vec)) if m != n}

        for m in range(len(p_vec)):
            for n in range(len(p_vec)):
                if m == n:
                    continue

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
                            A_mn_est = A_mn_iter[f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}'][f'{m+1}{n+1}']
                            error_matrix = A_mn_true - A_mn_est
                            error_matrices[:, iteration_idx, r] = error_matrix.flatten()

                # Compute the average error matrix over runs for each iteration
                avg_error_matrices = np.mean(error_matrices, axis=2)  # Shape: (num_elements, K)

                # Compute the L2 norm of the averaged error matrix for each iteration
                for k in range(K):
                    avg_norm_error[f'{m+1}{n+1}'][k] = np.linalg.norm(avg_error_matrices[:, k], ord=2)  # L2 norm

                # Compute the trace of covariance of the error vector at each iteration
                for k in range(K):
                    cov_matrix = np.cov(error_matrices[:, k, :], rowvar=False)
                    trace_cov_error[f'{m+1}{n+1}'][k] = np.trace(cov_matrix)

        trace_cov_error_data[subfolder] = trace_cov_error
        avg_norm_error_data[subfolder] = avg_norm_error
    else:
        print(f"File not found: {pkl_path}")

# Plot Trace(Cov(Vec(Error))) vs Iteration
for key in trace_cov_error_data[subfolders[0]].keys():
    plt.figure(figsize=(10, 6))
    for subfolder, trace_cov_error in trace_cov_error_data.items():
        plt.plot(trace_cov_error[key], label=f'{subfolder}')
    plt.xlabel('Iteration')
    plt.ylabel(f'Trace(Cov(Vec(Error_{key})))')
    plt.title(f'Trace(Cov(Vec(Error_{key}))) vs Iteration for Different Observation Noises')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(summary_plots_folder, f'Trace_Cov_Error_{key}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

# Plot Norm of Averaged Error vs Iteration
for key in avg_norm_error_data[subfolders[0]].keys():
    plt.figure(figsize=(10, 6))
    for subfolder, avg_norm_error in avg_norm_error_data.items():
        plt.plot(avg_norm_error[key], label=f'{subfolder}')
    plt.xlabel('Iteration')
    plt.ylabel(f'L2 Norm of Averaged Error_{key}')
    plt.title(f'L2 Norm of Averaged Error_{key} vs Iteration for Different Observation Noises')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(summary_plots_folder, f'L2_Norm_Avg_Error_{key}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


# 5. Plot Trace(Cov(Ls_gradx_iter)) vs Iteration
# Initialize a dictionary to store trace of covariance of Ls_gradx_iter for each subfolder
trace_cov_gradx_data = {}

# Loop through each subfolder
for subfolder in subfolders:
    pkl_path = os.path.join(parent_folder, subfolder, 'results', 'post_train_data.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            post_train_data = pickle.load(f)

        # Extract necessary data
        Ls_gradx_iter = post_train_data['Ls_gradx_iter']  # Gradient data
        p_vec = post_train_data['p_vec']  # Size of each component
        K = post_train_data['total_iterations']  # Total number of iterations
        no_runs = post_train_data['number_of_runs']  # Number of runs
        M = len(p_vec)  # Number of components

        # Initialize a dictionary to store trace of covariance for this subfolder
        trace_cov_gradx = {f'{m+1}': np.zeros(K) for m in range(M)}

        # Loop through each component
        for m in range(M):
            gradx_matrices = np.zeros((p_vec[m, 0], K, no_runs))  # Shape: (num_elements, iterations, runs)

            # Collect gradient data for all runs
            for r in range(no_runs):
                for e in range(post_train_data['epoch']):
                    for t in range(post_train_data['training_time']):
                        iteration_idx = e * post_train_data['training_time'] + t
                        gradx_matrices[:, iteration_idx, r] = Ls_gradx_iter[f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}'][f'{m+1}'][:, 0]

            # Compute the trace of covariance of the gradient vector at each iteration
            for k in range(K):
                cov_matrix = np.cov(gradx_matrices[:, k, :], rowvar=False)
                trace_cov_gradx[f'{m+1}'][k] = np.trace(cov_matrix)

        # Store the trace of covariance data for this subfolder
        trace_cov_gradx_data[subfolder] = trace_cov_gradx
    else:
        print(f"File not found: {pkl_path}")

# Plot Trace(Cov(Ls_gradx_iter)) vs Iteration for each component
for m in range(len(p_vec)):  # Iterate over all components
    plt.figure(figsize=(10, 6))
    for subfolder, trace_cov_gradx in trace_cov_gradx_data.items():
        plt.plot(trace_cov_gradx[f'{m+1}'], label=f'{subfolder}')
    plt.xlabel('Iteration')
    plt.ylabel(f'Trace(Cov(Ls_gradx_iter_{m+1}))')
    plt.title(f'Trace(Cov(Ls_gradx_iter_{m+1})) vs Iteration for Different Observation Noises')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(summary_plots_folder, f'Trace_Cov_Ls_gradx_iter_{m+1}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


# 6. Plot Trace(Cov(theta)) vs Iteration
# Initialize dictionary to store trace of covariance data
trace_cov_theta_data = {}

for subfolder in subfolders:
    pkl_path = os.path.join(parent_folder, subfolder, 'results', 'post_train_data.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            post_train_data = pickle.load(f)

        # Extract necessary data
        theta_iter = post_train_data['theta_iter']
        p_vec = post_train_data['p_vec']
        # d_vec = post_train_data['d_vec']
        d_vec = np.array([[8], [8]])
        K = post_train_data['total_iterations']
        no_runs = post_train_data['number_of_runs']
        training_time = post_train_data['training_time']
        epoch = post_train_data['epoch']
        M = len(p_vec)  # Number of components

        # Initialize trace of covariance for this subfolder
        trace_cov_theta = {f'{m+1}': np.zeros(K) for m in range(M)}

        # For each component
        for m in range(M):
            # Initialize array to store theta matrices for all runs and iterations
            # Shape: (flattened_theta_size, iterations, runs)
            theta_matrices = np.zeros((p_vec[m, 0] * d_vec[m, 0], K, no_runs))

            # Collect theta data for all runs
            for r in range(no_runs):
                for e in range(epoch):
                    for t in range(training_time):
                        iteration_idx = e * training_time + t
                        # Get theta matrix and flatten it
                        theta_matrices[:, iteration_idx, r] = theta_iter[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:,:,t].flatten()

            # Compute trace of covariance at each iteration
            for k in range(K):
                # Compute covariance across runs for this iteration
                cov_matrix = np.cov(theta_matrices[:, k, :], rowvar=False)
                trace_cov_theta[f'{m+1}'][k] = np.trace(cov_matrix)

        trace_cov_theta_data[subfolder] = trace_cov_theta
    else:
        print(f"File not found: {pkl_path}")

# Plot trace of covariance for each component
for m in range(M):
    plt.figure(figsize=(10, 6))
    for subfolder in subfolders:
        if subfolder in trace_cov_theta_data:
            plt.plot(trace_cov_theta_data[subfolder][f'{m+1}'], 
                    label=f'{subfolder}')
    
    plt.xlabel('Iteration')
    plt.ylabel(f'Trace(Cov(theta_{m+1}))')
    plt.title(f'Trace of Covariance of theta_{m+1} vs Iteration')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(summary_plots_folder, f'Trace_Cov_theta_{m+1}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()