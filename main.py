'''Only for off-diagonal blocks of the system matrix A'''
import data_retriever
import numpy as np
import global_learner
import local_learner
import spectral_radius
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
import pandas as pd
import training_plots
import validation_plots
import sys
import time
import os
import pickle

all_data    = data_retriever.RetrieveData('config.ini')

# Retreiving all data from the file
M                                           = all_data.num_components
training_time                               = all_data.training_time
total_time                                  = all_data.total_time
# validation_time                             = all_data.validation_time
GL_data                                     = all_data.global_learners_pack
LL_data                                     = all_data.local_learners_pack
p_vec                                       = all_data.comp_size_vec
p                                           = all_data.global_state_size
d_vec                                       = all_data.output_size_vec
d                                           = all_data.global_output_size
hmatrix_data                                = all_data.HMatrix_pack
results_location                            = all_data.results_location
ckf_data                                    = all_data.CKF_pack
# validation_data                             = all_data.validation_pack
d_start_vec                                 = all_data.output_start_vec


os.makedirs(results_location, exist_ok=True)

A_complete      = ckf_data['A_complete']
C_complete      = ckf_data['C_complete']

# run 
no_runs         = 10

# Initializing the epoch
epoch                           = 1

# Total number of iterates
K                               = epoch * training_time

# print(type(p_vec[0,0]))
p_start_idx                     = np.zeros((M, 1), dtype = int)

for m in range(M):
    if m > 0:
        p_start_idx[m, 0]   = p_start_idx[m-1, 0] + p_vec[m-1, 0]

# Stopping condition for local loss and local loss values for each component
global_loss_tol                  = 10**(-11)

# Averaging time intervals
avg_t_start         = 1669
avg_t_end           = 1680

# Snapshot time
snapshot_time       = 1690


run_SR_check        = False
run_training        = True
plot_training       = True
run_validation      = False

start_time              = time.time()

# Initializing the local model vector to be sent to the global model
X_dkf                   = {}
X_dkf_pred              = {}
X_dkf_resd              = {}
spectral_radius_list    = {}

if run_SR_check:

    LL1     = {}
    for m in range(M):
        LL1[f'{m+1}']                            = local_learner.LocalModel(training_time, LL_data[f'comp_{m+1}'])

        # Running the local model
        LL1[f'{m+1}'].run_DKF()

        # Combining the local model data to be used in the global model
        X_dkf[f'{m+1}']         = LL1[f'{m+1}'].X_dkf
        X_dkf_pred[f'{m+1}']    = LL1[f'{m+1}'].X_dkf_pred
        X_dkf_resd[f'{m+1}']    = LL1[f'{m+1}'].X_dkf_resd
    

    # Initializing the Hmatrix class and checking the spectral radius condition
    SR                      = spectral_radius.SpectralRadius(hmatrix_data, X_dkf)

    # Collecting the spectral radius of each subsystem
    for m in range(M):
        spectral_radius_list[f'{m+1}']      = SR.SR_List(m+1)
        print(f'for comp_{m+1}: ', 'max = ', np.max(spectral_radius_list[f'{m+1}']), 'argmax = ', np.argmax(spectral_radius_list[f'{m+1}']))

    print(SR.check_SR_condn_all())

    os.makedirs(results_location, exist_ok=True)      
    time_vector = np.linspace(1, training_time, training_time)

    for m in range(M):
        # save_dir        = os.path.join(self.results_location, f'comp_{m+1}')
        # os.makedirs(save_dir, exist_ok=True)
        
        plt.figure()
        plt.rcParams.update({'font.size': 15})
        plt.plot(time_vector, spectral_radius_list[f'{m+1}'][:,0])
        plt.xlabel('Time')
        plt.ylabel('Spectral Radius')
        plt.title(f'Spectral Radius of system of component {m+1} vs time')
        # Save the plot
        # plot_filename = os.path.join(save_dir, f'SR_{m+1}_vs_time.png')
        plot_filename = os.path.join(results_location, f'SR_{m+1}_vs_time.png')
        plt.savefig(plot_filename)
        # plt.show()
        plt.close()

    

    


# Training runs if training condition is true
if run_training:
    # Initializing the off-diagonal elements
    A_mn_est        = {}
    A_mn_iter       = {}
    A_mn_vec        = {}
    A_mn_error      = {}

    Ls_gradx_iter   = {}
    global_loss_vec     = {}
    local_loss_vec      = {}
    global_loss_comp_vec    = {}

    # Initializing the augmented local model vector to be shared with the global model
    X_vfl                   = {}
    X_vfl_pred              = {}
    X_vfl_resd              = {}

    X_ckf_resd              = {}
    X_dep                   = {}

    # local_loss_vec              = {}
    theta_iter                  = {}
    theta_vec                   = {}
    theta_est                   = {}

    # Local Learner Dictionary
    LL                          = {}


    for r in range(no_runs):
        A_mn_est[f'run_{r+1}']              = {}
        A_mn_iter[f'run_{r+1}']             = {}
        A_mn_vec[f'run_{r+1}']              = {}
        A_mn_error[f'run_{r+1}']            = {}
        global_loss_vec[f'run_{r+1}']       = {}
        local_loss_vec[f'run_{r+1}']        = {}
        global_loss_comp_vec[f'run_{r+1}']  = {}

        Ls_gradx_iter[f'run_{r+1}']         = {}

        theta_est[f'run_{r+1}']             = {}
        theta_iter[f'run_{r+1}']            = {}
        theta_vec[f'run_{r+1}']             = {}
        X_vfl[f'run_{r+1}']                 = {}
        X_vfl_pred[f'run_{r+1}']            = {}
        X_dep[f'run_{r+1}']                 = {}
        X_vfl_resd[f'run_{r+1}']            = {}
        LL[f'run_{r+1}']                    = {}

        for m in range(M):

            # Initializing theta estimates
            theta_est[f'run_{r+1}'][f'{m+1}']     = np.random.normal(0.5, 0.01, (p_vec[m, 0], d_vec[m, 0]))
            # theta_est[f'run_{r+1}'][f'{m+1}']     = np.random.normal(0.5, 10, (p_vec[m, 0], d_vec[m, 0]))
            # theta_est[f'run_{r+1}'][f'{m+1}']     = np.random.normal(0, 0.01, (p_vec[m, 0], d_vec[m, 0]))
            # if m == 1:
            #     theta_est[f'run_{r+1}'][f'{m+1}']     = np.random.normal(0, 0.01, (p_vec[m, 0], d_vec[m, 0]))

            for n in range(M):
                if m == n:
                    continue
                else:
                    # A_mn_est[f'run_{r+1}'][f'{m+1}{n+1}'] = 0 * np.ones((p_vec[m, 0], p_vec[n, 0])) 
                    # A_mn_est[f'run_{r+1}'][f'{m+1}{n+1}'] = np.random.normal(0.8, 0.075, (p_vec[m, 0], p_vec[n, 0]))
                    A_mn_est[f'run_{r+1}'][f'{m+1}{n+1}'] = np.random.normal(0.8, 1e-6, (p_vec[m, 0], p_vec[n, 0]))
                    # A_mn_est[f'run_{r+1}'][f'{m+1}{n+1}'] = np.random.normal(0, 0.075, (p_vec[m, 0], p_vec[n, 0]))
                    # if m == 0 and n == 1:
                    #     A_mn_est[f'run_{r+1}'][f'{m+1}{n+1}'] = np.random.normal(0, 0.075, (p_vec[m, 0], p_vec[n, 0])) 



        for e in range(epoch):
            A_mn_iter[f'run_{r+1}'][f'epoch_{e+1}']                 = {}
            A_mn_vec[f'run_{r+1}'][f'epoch_{e+1}']                  = {}
            global_loss_vec[f'run_{r+1}'][f'epoch_{e+1}']           = np.zeros((1, training_time))
            local_loss_vec[f'run_{r+1}'][f'epoch_{e+1}']            = {}
            global_loss_comp_vec[f'run_{r+1}'][f'epoch_{e+1}']      = {}
            A_mn_error[f'run_{r+1}'][f'epoch_{e+1}']                = {}

            Ls_gradx_iter[f'run_{r+1}'][f'epoch_{e+1}']             = {}

            theta_iter[f'run_{r+1}'][f'epoch_{e+1}']          = {}
            theta_vec[f'run_{r+1}'][f'epoch_{e+1}']           = {}
            X_vfl[f'run_{r+1}'][f'epoch_{e+1}']               = {}
            X_vfl_pred[f'run_{r+1}'][f'epoch_{e+1}']          = {}
            X_dep[f'run_{r+1}'][f'epoch_{e+1}']               = {}
            X_vfl_resd[f'run_{r+1}'][f'epoch_{e+1}']          = {}

            for t in range(training_time):
                A_mn_iter[f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}']         = {}
                A_mn_vec[f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}']          = {}
                # A_mn_error[f'epoch_{e+1}'][f'{t+1}']        = {}
                Ls_gradx_iter[f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}']     = {}

                

            for m in range(M):
                local_loss_vec[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}']    = np.zeros((1, training_time))
                global_loss_comp_vec[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}']    = np.zeros((1, training_time))

                Ls_gradx_iter[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}']     = np.zeros((p_vec[m, 0], training_time))
                theta_iter[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}']     = np.zeros((p_vec[m, 0], d_vec[m, 0], training_time))
                theta_vec[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}']     = np.zeros((p_vec[m, 0]*d_vec[m, 0], training_time))
                X_vfl[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}']         = np.zeros((p_vec[m,0], training_time))
                X_vfl_pred[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}']    = np.zeros((p_vec[m,0], training_time))
                X_dep[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}']         = np.zeros((p_vec[m, 0], training_time))
                X_vfl_resd[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}']    = np.zeros((1, training_time))

                for n in range(M):
                    if m == n:
                        continue
                    else:
                        A_mn_error[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}{n+1}']   = np.zeros((1, training_time))


    for m in range(M):
        X_ckf_resd[f'{m+1}']    = np.zeros((1, training_time))

        

    A_cap               = {}
    GL                  = {}


    # Starting the training
    for r in range(no_runs):

        
        print('run = ', r+1)

        skip_condition_global   = False
        # Initializing global learner
        GL[f'run_{r+1}']                  = global_learner.GlobalModel(M, training_time, GL_data, A_mn_est[f'run_{r+1}'])

        for m in range(M):
            LL[f'run_{r+1}'][f'{m+1}']                            = local_learner.LocalModel(training_time, LL_data[f'comp_{m+1}'], theta_est[f'run_{r+1}'][f'{m+1}'])

            # Running the local model
            LL[f'run_{r+1}'][f'{m+1}'].run_DKF()

            # Combining the local model data to be used in the global model
            X_dkf[f'{m+1}']         = LL[f'run_{r+1}'][f'{m+1}'].X_dkf
            X_dkf_pred[f'{m+1}']    = LL[f'run_{r+1}'][f'{m+1}'].X_dkf_pred
            X_dkf_resd[f'{m+1}']    = LL[f'run_{r+1}'][f'{m+1}'].X_dkf_resd
            # print(X_dkf_resd[f'{m+1}'])


        global_loss_curr        = float(8)
        global_loss_prev        = float(0)

        
        # Initializing the entry to be used by the global model
        x_dkf_global                = {}
        x_vfl_global                = {}
        grad_x                      = {}

        for e in range(epoch):
            print('epoch = ', e+1)
            for t in range(training_time):
                # Estimating the augmented local model with the current value of theta

                for m in range(M):
                    # X_vfl_pred[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:,t:t+1]                   = LL[f'run_{r+1}'][f'{m+1}'].VFL_prediction(t)
                    X_vfl[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:,t:t+1]                        = LL[f'run_{r+1}'][f'{m+1}'].VFL_estimate(t)
                    X_vfl_pred[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:,t:t+1]                   = LL[f'run_{r+1}'][f'{m+1}'].VFL_prediction(t)
                    X_vfl_resd[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][0,t:t+1]                   = np.linalg.norm(LL[f'run_{r+1}'][f'{m+1}'].VFL_residual(t))
                    local_loss_vec[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][0, t]                    = LL[f'run_{r+1}'][f'{m+1}'].VFL_local_loss(t)
                    x_vfl_global[f'{m+1}']                                                          = X_vfl[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:,t:t+1]
                    x_dkf_global[f'{m+1}']                                                          = X_dkf[f'{m+1}'][:,t:t+1]

                    # print(f"Keys in LL[f'run_{r+1}']: {LL[f'run_{r+1}'].keys()}")
                    # print(f"Expected key: {f'{m+1}'}")      
                    theta_iter[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:,:,t]             = LL[f'run_{r+1}'][f'{m+1}'].theta.copy()
                    theta_vec[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:,t]            = LL[f'run_{r+1}'][f'{m+1}'].theta.flatten()
                    # if avg_t_start <= t < avg_t_end:
                    #     print(f'for {m+1} and {t+1} = ', theta_iter[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:,:,t] @ LL_data[f'comp_{m+1}']['Y'][:,t:t+1], X_vfl[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:,t:t+1] - X_dkf[f'{m+1}'][:,t:t+1])

                # Gradient descent of the off-diagonal elements
                
                grad_x          = GL[f'run_{r+1}'].Gradx(x_dkf_global, x_vfl_global, t)
                Ls_gradx_iter[f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}']     = grad_x.copy()

                # for m in range(M):
                    # if not skip_condition[f'{m+1}']:
                if not skip_condition_global:
                    if np.abs(global_loss_curr - global_loss_prev) > global_loss_tol:
                        A_mn_est[f'run_{r+1}']        = GL[f'run_{r+1}'].GradDescent(x_dkf_global, x_vfl_global, t).copy()
                    else:
                        skip_condition_global       = True
                        print('time of GD stoppage = ', t)

                A_mn_iter[f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}']         = GL[f'run_{r+1}'].A_mn.copy()
                A_mn_vec[f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}']          = GL[f'run_{r+1}'].A_mn_vec.copy()

                for m in range(M):
                    row_start_idx   = p_start_idx[m, 0]
                    row_end_idx     = p_start_idx[m, 0] + p_vec[m, 0]
                    for n in range(M):
                        if m == n:
                            X_dep[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:,t:t+1]    += LL_data[f'comp_{m+1}']['A'] @ X_dkf[f'{m+1}'][:,t:t+1]
                        else:
                            X_dep[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:,t:t+1]    += A_mn_iter[f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}'][f'{m+1}{n+1}'] @ X_dkf[f'{n+1}'][:,t:t+1]
                            col_start_idx   = p_start_idx[n, 0]
                            col_end_idx     = p_start_idx[n, 0] + p_vec[n, 0]

                            A_mn_error[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}{n+1}'][0, t]  = np.linalg.norm(A_mn_iter[f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}'][f'{m+1}{n+1}'] - A_complete[row_start_idx: row_end_idx, col_start_idx:col_end_idx], ord = 'fro')

                global_loss_vec[f'run_{r+1}'][f'epoch_{e+1}'][0, t]    = GL[f'run_{r+1}'].GlobalLoss(x_dkf_global, x_vfl_global, t)

                # if 0 <= t < 3500:
                #     print(global_loss_vec[f'run_{r+1}'][f'epoch_{e+1}'][0, t])

                # print(global_loss_vec[f'run_{r+1}'][f'epoch_{e+1}'][0, t])

                global_loss_prev       = global_loss_curr
                global_loss_curr       = global_loss_vec[f'run_{r+1}'][f'epoch_{e+1}'][0, t]

                for m in range(M):
                    global_loss_comp_vec[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][0, t]   = GL[f'run_{r+1}'].GlobalLoss_comp(m, x_dkf_global, x_vfl_global, t)
                    if not skip_condition_global:
                        LL[f'run_{r+1}'][f'{m+1}'].GradDescent(grad_x[f'{m+1}'],t)


                    # theta_iter[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][:,:,t]             = LL[f'run_{r+1}'][f'{m+1}'].theta.copy()
                    if e == epoch - 1 and t == training_time - 1:
                        theta_est[f'run_{r+1}'][f'{m+1}']         = LL[f'run_{r+1}'][f'{m+1}'].theta.copy()

        # Reconstructing A for each run
        A_cap[f'run_{r+1}']             = np.zeros((p,p))
        row_end_idx     = 0
        for i in range(M):
            col_end_idx         = 0
            row_start_idx       = row_end_idx
            row_end_idx         = row_start_idx + p_vec[i, 0]
            for j in range(M):
                col_start_idx   = col_end_idx
                col_end_idx     = col_start_idx + p_vec[j, 0]
                if i == j:
                    A_cap[f'run_{r+1}'][row_start_idx: row_end_idx, col_start_idx: col_end_idx]   = LL_data[f'comp_{i+1}']['A']
                else:
                    A_cap[f'run_{r+1}'][row_start_idx: row_end_idx, col_start_idx: col_end_idx]   = A_mn_est[f'run_{r+1}'][f'{i+1}{j+1}']

    ####### TRAINING-DONE ##########
    
    print('TRAINING HAS ENDED')

    # for t in range(avg_t_start, avg_t_end):
    #     for m in range(M):
    #         print(f'for {m+1} and {t+1} = ', theta_iter[f'run_{no_runs}'][f'epoch_{epoch}'][f'{m+1}'][:,:,t] @ LL_data[f'comp_{m+1}']['Y'][:,t:t + 1], X_vfl[f'run_{no_runs}'][f'epoch_{epoch}'][f'{m+1}'][:,t:t+1] - X_dkf[f'{m+1}'][:,t:t+1])



# ########################### Covariance Computation ###########################
# trace_cov_A_mn = {f'{m+1}{n+1}': np.zeros((K,1)) for m in range(M) for n in range(M) if m != n}
# trace_cov_theta = {f'{m+1}': np.zeros((K,1)) for m in range(M)}
# trace_cov_x_vfl_pred = {f'{m+1}': np.zeros((K, 1)) for m in range(M)}


# for e in range(epoch):
#     for t in range(training_time):
    
#         for key in trace_cov_A_mn.keys():
#             values_A_mn = []
#             for r in range(no_runs):
#                 values_A_mn.append(A_mn_vec[f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}'][key])
            
#             value_array_A_mn = np.array(values_A_mn)
#             trace_cov_A_mn[key][e*training_time + t, 0] = np.trace(np.cov(value_array_A_mn, rowvar=False))

#         for key in trace_cov_theta.keys():
#             values_theta = []
#             values_x_vfl_pred = []
#             for r in range(no_runs):
#                 values_theta.append(theta_vec[f'run_{r+1}'][f'epoch_{e+1}'][key][:,t])
#                 values_x_vfl_pred.append(X_vfl_pred[f'run_{r+1}'][f'epoch_{e+1}'][key][:, t])
            
#             value_array_theta = np.array(values_theta)
#             trace_cov_theta[key][e*training_time + t, 0] = np.trace(np.cov(value_array_theta, rowvar=False))

#             value_array_x_vfl_pred = np.array(values_x_vfl_pred)
#             trace_cov_x_vfl_pred[key][e * training_time + t, 0] = np.trace(np.cov(value_array_x_vfl_pred, rowvar=False))
        
# print('Final trace_cov_A_mn = ', trace_cov_A_mn['21'][training_time - 1, 0])



# # Plot Trace(Cov(Vec(A_mn))) vs iteration
# for key, trace_values in trace_cov_A_mn.items():

#     trace_csv_path = os.path.join(results_location, f"Trace_Cov_A_mn_{key}.csv")
#     np.savetxt(trace_csv_path, trace_values, delimiter=",")
#     print(f"Saved Trace(Cov(Vec(A_mn_{key}))) to {trace_csv_path}")

#     plt.figure()
#     plt.plot(range(training_time), trace_values)
#     plt.xlabel('Iteration')
#     plt.ylabel(f'Trace(Cov(Vec(A_mn_{key})))')
#     plt.title(f'Trace(Cov(Vec(A_mn_{key}))) vs Iteration')
#     plt.savefig(os.path.join(results_location, f'Trace_Cov_A_mn_{key}.png'))
#     plt.close()

# # Plot Trace(Cov(Vec(theta))) vs iteration
# for key, trace_values in trace_cov_theta.items():
    
#     trace_csv_path = os.path.join(results_location, f"Trace_Cov_theta_{key}.csv")
#     np.savetxt(trace_csv_path, trace_values, delimiter=",")
#     print(f"Saved Trace(Cov(Vec(theta_{key}))) to {trace_csv_path}")

#     plt.figure()
#     plt.plot(range(training_time), trace_values)
#     plt.xlabel('Iteration')
#     plt.ylabel(f'Trace(Cov(Vec(theta_{key})))')
#     plt.title(f'Trace(Cov(Vec(theta_{key}))) vs Iteration')
#     plt.savefig(os.path.join(results_location, f'Trace_Cov_theta_{key}.png'))
#     plt.close() 

# # Save Trace(Cov(Vec(x_vfl_pred))) to CSV
# for key, trace_values in trace_cov_x_vfl_pred.items():
#     trace_csv_path = os.path.join(results_location, f"Trace_Cov_x_vfl_pred_{key}.csv")
#     np.savetxt(trace_csv_path, trace_values, delimiter=",")
#     print(f"Saved Trace(Cov(Vec(x_vfl_pred_{key}))) to {trace_csv_path}")
####################### training plot code - starts here #######################
Y               = ckf_data['Y'][:,:training_time]

B      = ckf_data['B']
Q      = ckf_data['Q']
R      = ckf_data['R']
P0     = ckf_data['P0']
x0     = ckf_data['x0']
# # Getting the CKF values
CKF          = KalmanFilter(A_complete, B, C_complete, Q, R, P0, x0)

X_ckf           = np.zeros((A_complete.shape[0], training_time))
X_ckf_pred      = np.zeros((A_complete.shape[0], training_time))
# X_ckf_resd      = np.zeros((1, training_time))
for t in range(training_time):
    CKF.predict()
    X_ckf_pred[:,t:t+1]         = CKF.get_state()
    for m in range(M):
        # print(type(d_start_vec[m, 0]))
        # print(type(d_start_vec[m, 0] + d_vec[m, 0]))
        # if 0 <= t < 20:
            # print(f'd_start_idx of comp_{m+1} = ', d_start_vec[m, 0])
            # print(f'ckf_resd from comp_{m+1} = ', np.linalg.norm(CKF.residual(Y[:,t:t+1])[d_start_vec[m, 0]:d_start_vec[m, 0] + d_vec[m, 0],:]))
        X_ckf_resd[f'{m+1}'][0,t:t+1]         = np.linalg.norm(CKF.residual(Y[:,t:t+1])[d_start_vec[m, 0]:d_start_vec[m, 0] + d_vec[m, 0],:])
    CKF.update(Y[:,t:t+1])
    X_ckf[:,t:t+1]              = CKF.get_state()
    # print('X_ckf shape = ', X_ckf[:,t:t+1].shape, ' CKF shape = ', CKF.get_state().shape)


# Recontructing A
# A_cap           = np.zeros((p, p))
# row_end_idx     = 0
# for i in range(M):
#     col_end_idx         = 0
#     row_start_idx       = row_end_idx
#     row_end_idx         = row_start_idx + p_vec[i, 0]
#     for j in range(M):
#         col_start_idx   = col_end_idx
#         col_end_idx     = col_start_idx + p_vec[j, 0]
#         if i == j:
#             A_cap[row_start_idx: row_end_idx, col_start_idx: col_end_idx]   = LL_data[f'comp_{i+1}']['A']
#         else:
#             A_cap[row_start_idx: row_end_idx, col_start_idx: col_end_idx]   = A_mn_est[f'{i+1}{j+1}']
            
# # for t in range(training_time):

# #     for m in range(M):
# #         for j in range(M):
# #             if m == j:
# #                 X_dep[f'{m+1}'][:,t:t+1]    += LL_data[f'comp_{m+1}']['A'] @ X_dkf[f'{m+1}'][:,t:t+1]
# #             else:
# #                 X_dep[f'{m+1}'][:,t:t+1]    += A_mn_iter[f'epoch_{epoch}'][f'{t+1}'][f'{m+1}{j+1}'] @ X_dkf[f'{j+1}'][:,t:t+1]

# # for m in range(M):
# #     print(f'theta of {m+1} = ', theta_est_avg[f'{m+1}'])



# # for e in range(epoch):
# #     avg_val         = 0
# #     for t in range(avg_t_start, avg_t_end):
# #         avg_val += 1/(avg_t_end - avg_t_start) * global_loss_vec[f'run_{no_runs}'][f'epoch_{e+1}'][t, 0]
# #     print(f'L_g at epoch_{e+1} = ', avg_val)
# # Taking average of values across all the runs

X_vfl_avg                   = {}
X_vfl_pred_avg              = {}
X_vfl_resd_avg              = {}
X_dep_avg                   = {}
local_loss_avg              = {}
global_loss_comp_avg        = {}
global_loss_avg             = {}
theta_est_avg               = {}
A_mn_error_avg              = {}
A_mn_est_avg                = {}

for e in range(epoch):
    X_vfl_avg[f'epoch_{e+1}']               = {}
    X_vfl_pred_avg[f'epoch_{e+1}']          = {}
    X_vfl_resd_avg[f'epoch_{e+1}']          = {}
    X_dep_avg[f'epoch_{e+1}']               = {}
    local_loss_avg[f'epoch_{e+1}']          = {}
    global_loss_comp_avg[f'epoch_{e+1}']    = {}
    global_loss_avg[f'epoch_{e+1}']         = np.zeros((1, training_time))
    A_mn_error_avg[f'epoch_{e+1}']           = {}

    for m in range(M):
        X_vfl_avg[f'epoch_{e+1}'][f'{m+1}']         = np.zeros((p_vec[m, 0], training_time))
        X_vfl_pred_avg[f'epoch_{e+1}'][f'{m+1}']              = np.zeros((p_vec[m, 0], training_time))
        X_vfl_resd_avg[f'epoch_{e+1}'][f'{m+1}']             = np.zeros((1, training_time))
        X_dep_avg[f'epoch_{e+1}'][f'{m+1}']         = np.zeros((p_vec[m, 0], training_time))
        local_loss_avg[f'epoch_{e+1}'][f'{m+1}']              = np.zeros((1, training_time))
        global_loss_comp_avg[f'epoch_{e+1}'][f'{m+1}']        = np.zeros((1, training_time))


A_cap_avg           = np.mean([A_cap[f'run_{r+1}'] for r in range(no_runs)], axis = 0)

for m in range(M):
    theta_est_avg[f'{m+1}']         = np.mean([theta_est[f'run_{r+1}'][f'{m+1}'] for r in range(no_runs)], axis = 0)

    for n in range(M):
        if m != n:
            A_mn_est_avg[f'{m+1}{n+1}']        = np.mean([A_mn_est[f'run_{r+1}'][f'{m+1}{n+1}'] for r in range(no_runs)], axis = 0)



for e in range(epoch):
    for r in range(no_runs):
        # print('global_loss_vec shape = ', global_loss_vec[f'run_{r+1}'][f'epoch_{e+1}'].shape)
        global_loss_avg[f'epoch_{e+1}']         += (1/no_runs) * global_loss_vec[f'run_{r+1}'][f'epoch_{e+1}']

    for m in range(M):
        for r in range(no_runs):
            X_vfl_avg[f'epoch_{e+1}'][f'{m+1}'] += (1/no_runs) * X_vfl[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}']
            X_vfl_pred_avg[f'epoch_{e+1}'][f'{m+1}'] += (1/no_runs) * X_vfl_pred[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}']
            X_vfl_resd_avg[f'epoch_{e+1}'][f'{m+1}'] += (1/no_runs) * X_vfl_resd[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}']
            X_dep_avg[f'epoch_{e+1}'][f'{m+1}'] += (1/no_runs) * X_dep[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}']
            local_loss_avg[f'epoch_{e+1}'][f'{m+1}'] += (1/no_runs) * local_loss_vec[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}']
            global_loss_comp_avg[f'epoch_{e+1}'][f'{m+1}'] += (1/no_runs) * global_loss_comp_vec[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}']                


########################### Covariance Computation ###########################

# Pre-allocate memory for covariance matrices
cov_A_mn = {f'{m+1}{n+1}': np.zeros((p_vec[m, 0] * p_vec[n, 0], p_vec[m, 0] * p_vec[n, 0], K))
            for m in range(M) for n in range(M) if m != n}
cov_theta = {f'{m+1}': np.zeros((p_vec[m, 0] * d_vec[m, 0], p_vec[m, 0] * d_vec[m, 0], K))
             for m in range(M)}

# Loop through epochs and training time to compute covariance
for e in range(epoch):
    for t in range(training_time):
        iteration_idx = e * training_time + t  # Flatten epoch and time into a single index

        # Compute covariance for A_mn matrices
        for key in cov_A_mn.keys():
            values_A_mn = []
            for r in range(no_runs):
                values_A_mn.append(A_mn_vec[f'run_{r+1}'][f'epoch_{e+1}'][f'{t+1}'][key].flatten())
            value_array_A_mn = np.array(values_A_mn)
            cov_A_mn[key][:, :, iteration_idx] = np.cov(value_array_A_mn, rowvar=False)  # Store the full covariance matrix

        # Compute covariance for theta
        for key in cov_theta.keys():
            values_theta = []
            for r in range(no_runs):
                values_theta.append(theta_vec[f'run_{r+1}'][f'epoch_{e+1}'][key][:, t].flatten())
            value_array_theta = np.array(values_theta)
            cov_theta[key][:, :, iteration_idx] = np.cov(value_array_theta, rowvar=False)  # Store the full covariance matrix


post_train_data = {}

post_train_data['epoch']                = epoch
post_train_data['training_time']        = training_time
post_train_data['number_of_runs']       = no_runs
post_train_data['total_iterations']     = K
post_train_data['A_mn_iter']           = A_mn_iter
post_train_data['p_vec']               = p_vec
post_train_data['p_start_idx']         = p_start_idx
post_train_data['d_vec']               = d_vec
post_train_data['d_start_vec']         = d_start_vec

post_train_data['X_dkf']                = X_dkf
post_train_data['X_dkf_pred']           = X_dkf_pred
post_train_data['X_dkf_residual']       = X_dkf_resd
post_train_data['X_vfl']                = X_vfl_avg
post_train_data['X_vfl_pred']           = X_vfl_pred_avg

post_train_data['X_vfl_pred_iter']       = X_vfl_pred
post_train_data['X_vfl_iter']               = X_vfl

post_train_data['X_vfl_residual']       = X_vfl_resd_avg
post_train_data['local_loss']           = local_loss_vec
post_train_data['global_loss']          = global_loss_vec
post_train_data['global_loss_comp']     = global_loss_comp_vec
# post_train_data['spectral_radius']      = spectral_radius_list
post_train_data['A_complete_true']      = A_complete
post_train_data['C_complete_true']      = C_complete
post_train_data['X_ckf']                = X_ckf
post_train_data['X_ckf_pred']           = X_ckf_pred
post_train_data['X_ckf_residual']       = X_ckf_resd
# post_train_data['X_ckf_pred_cap']     = X_ckf_pred_cap
post_train_data['X_dep']                = X_dep_avg
post_train_data['A_cap']                = A_cap_avg
post_train_data['A_mn_est']             = A_mn_est_avg
post_train_data['theta_cap']            = theta_est_avg
post_train_data['A_mn_error']           = A_mn_error
post_train_data['cov_A_mn']             = cov_A_mn
post_train_data['cov_theta']            = cov_theta
post_train_data['Ls_gradx_iter']         = Ls_gradx_iter
post_train_data['theta_iter']           = theta_iter
post_train_data['theta_vec']            = theta_vec

# Saving the trained variables in the dictionary
save_train_data            = os.path.join(all_data.results_location, 'post_train_data.pkl')
with open(save_train_data, 'wb') as f:
    pickle.dump(post_train_data, f)



# # save_train_plots.save_residuals(run = no_runs, epoch = 1)
# # save_train_plots.save_residuals(run = no_runs, epoch = 3)
# # save_train_plots.save_residuals(run = no_runs, epoch = 6)
# # save_train_plots.save_residuals(run = no_runs, epoch=9)
# # save_train_plots.save_residuals(run = no_runs, epoch = 11)

# # save_train_plots.save_loss_vs_epoch(time = snapshot_time)
# # save_train_plots.save_A_mn_vs_epoch(run = no_runs, time = snapshot_time)
# # save_train_plots.save_A_mn_vs_epoch_CI(time=snapshot_time)


# end_time            = time.time()
# print('time = ', end_time - start_time)


# train_data_load_path        = os.path.join(all_data.results_location, 'post_train_data.pkl')
# with open(train_data_load_path, 'rb') as f:
#     post_train_data         = pickle.load(f)

# # for t in range(avg_t_start, avg_t_end):
# #     print(post_train_data['X_ckf_pred'][2, t])

# print('A_true = ', A_complete)
# print('A_cap = ', post_train_data['A_cap'])
# for r in range(no_runs):
#     print(f'A_cap for run_{r+1} = ', A_cap[f'run_{r+1}'])
    # print(f'run_{r+1} = ', A_mn_est[f'run_{r+1}'])
# print(post_train_data['A_mn_est'])

###################### training plot code - ends here #######################
if plot_training:
    save_train_plots       = training_plots.DataSaver(post_train_data, M, no_runs, epoch, training_time, total_time,  results_location, p_vec, d_vec, p_start_idx)
    save_train_plots.save_all(time = snapshot_time, epoch=epoch)

############################################# VALIDATION ####################################################################

# if validation_time == 0:
#     print(f'Length of time series allotted for validation is zero')
#     sys.exit()

# Retreiving all data from the file

# validation_time         = training_time
if run_validation:
    # Loading the training data
    # train_data_load_path        = os.path.join(all_data.results_location, 'post_train_data.pkl')
    # with open(train_data_load_path, 'rb') as f:
    #     post_train_data         = pickle.load(f)

    # Extracting variables from the training data
    A_mn_est_avg        = post_train_data['A_mn_est']
    theta_est_avg       = post_train_data['theta_cap']
    X_dkf               = post_train_data['X_dkf']
    global_loss_vec     = post_train_data['global_loss']
    X_vfl_resd          = post_train_data['X_vfl_residual']
    X_dkf_resd          = post_train_data['X_dkf_residual']

    

    # print(X_vfl_resd.keys())
    # print('global loss keys = ', global_loss_vec['run_1'].keys())



    global_loss_valid     = np.zeros((1, validation_time))
    local_loss_valid      = {}

    # print(validation_time)

    # Initializing global learner
    GL_valid                = global_learner.GlobalModel(M, validation_time, GL_data, A_mn_est_avg)

    # Initializing global learner
    LL_valid                = {}

    # Initializing the local model vector to be sent to the global model
    X_dkf_valid                 = {}
    X_dkf_pred_valid            = {}
    X_dkf_resd_valid            = {}
    # Initializing the augmented local model vector to be shared with the global model
    X_vfl_valid                 = {}
    X_vfl_pred_valid            = {}
    X_vfl_resd_valid        = {}
    X_dep_valid             = {}

    X_ckf_resd_valid        = {}

    # Initializing loss vectors and theta

    for m in range(M):

        # Creating the Local model based on the data
        LL_valid[f'{m+1}']          = local_learner.LocalModel(validation_time, validation_data[f'comp_{m+1}'], theta_est_avg[f'{m+1}'])
        # print(validation_data[f'comp_{m+1}'])
        # print(f'reached here safely for m = {m+1}')
        # Running the local model
        LL_valid[f'{m+1}'].run_DKF()
        # print(LL_valid[f'{m+1}'].X_dkf_pred)
        X_dkf_resd_valid[f'{m+1}']      = LL_valid[f'{m+1}'].X_dkf_resd
        # print(X_dkf_resd_valid[f'{m+1}'])
        # Combining the local model data to be used in the global model
        X_dkf_valid[f'{m+1}']           = LL_valid[f'{m+1}'].X_dkf
        X_dkf_pred_valid[f'{m+1}']      = LL_valid[f'{m+1}'].X_dkf_pred
        X_vfl_valid[f'{m+1}']           = np.zeros((p_vec[m,0], validation_time))
        X_vfl_pred_valid[f'{m+1}']      = np.zeros((p_vec[m,0], validation_time))
        X_vfl_resd_valid[f'{m+1}']    = np.zeros((1, validation_time))
        X_ckf_resd_valid[f'{m+1}']      = np.zeros((1, validation_time))
        global_loss_curr        = float(8)
        global_loss_prev        = float(0)

        # print(f'reached here safely for m = {m+1}')

        X_dep_valid[f'{m+1}']         = np.zeros((p_vec[m, 0], validation_time))

        local_loss_valid[f'{m+1}']      = np.zeros((1, validation_time))
        # local_loss_vec[f'{m+1}']        = np.zeros((training_time, 1))

        # theta_iter[f'{m+1}']            = {}
        # theta_est[f'{m+1}']             = np.zeros((p_vec[m, 0], d_vec[m, 0]))

        # for e in range(epoch):
        #     theta_iter[f'{m+1}'][f'epoch_{e+1}']        = np.zeros((p_vec[m, 0], d_vec[m, 0], training_time))


        
    # Initializing the entry to be used by the global model
    x_dkf_global_valid                = {}
    x_vfl_global_valid                = {}
    # grad_x                      = {}
    # spectral_radius_list        = {}

    # # Initializing the Hmatrix class and checking the spectral radius condition
    # SR                      = spectral_radius.SpectralRadius(hmatrix_data, X_dkf)

    # Collecting the spectral radius of each subsystem
    # for m in range(M):
    #     spectral_radius_list[f'{m+1}']      = SR.SR_List(m+1)


    # print(SR.check_SR_condn_all())

    # avg_t_start = 1669
    # avg_t_end   = 1680

    # global_loss_train_use_vec = np.zeros((avg_t_end - avg_t_start, 1))

    # for t in range(avg_t_start, avg_t_end):
    #     avg = 0
    #     for r in range(no_runs):
    #         avg += 1 / (avg_t_end - avg_t_start) * global_loss_vec[f'run_{r+1}'][f'epoch_{epoch}'][t, 0]

    #     global_loss_train_use_vec[t - avg_t_start, 0]       = avg
        

    global_loss_comp_valid = {}

    for m in range(M):
        global_loss_comp_valid[f'{m+1}'] = 0

    for t in range(validation_time):
        # Estimating the augmented local model with the current value of theta

        for m in range(M):
            
            X_vfl_valid[f'{m+1}'][:,t:t+1]                        = LL_valid[f'{m+1}'].VFL_estimate(t)
            X_vfl_pred_valid[f'{m+1}'][:,t:t+1]                   = LL_valid[f'{m+1}'].VFL_prediction(t)
            X_vfl_resd_valid[f'{m+1}'][0,t]                   = np.linalg.norm(LL_valid[f'{m+1}'].VFL_residual(t))
            local_loss_valid[f'{m+1}'][0, t]                         = LL_valid[f'{m+1}'].VFL_local_loss(t)
            x_vfl_global_valid[f'{m+1}']                          = X_vfl_valid[f'{m+1}'][:,t:t+1]
            x_dkf_global_valid[f'{m+1}']                          = X_dkf_valid[f'{m+1}'][:,t:t+1]
            
            for n in range(M):
                if m == n:
                    X_dep_valid[f'{m+1}'][:,t:t+1]    += validation_data[f'comp_{m+1}']['A'] @ X_dkf_valid[f'{m+1}'][:,t:t+1]
                else:
                    X_dep_valid[f'{m+1}'][:,t:t+1]    += A_mn_est_avg[f'{m+1}{n+1}'] @ X_dkf_valid[f'{n+1}'][:,t:t+1]
        # Gradient descent of the off-diagonal elements
        
        # grad_x          = GL.Gradx(x_dkf_global, x_vfl_global, t)
        # for m in range(M):
            # if not skip_condition[f'{m+1}']:
        # if not skip_condition_global:
        #     if np.abs(global_loss_curr - global_loss_prev) > global_loss_tol:
        #         A_mn_est        = GL.GradDescent(x_dkf_global, x_vfl_global, t)
        #     else:
        #         skip_condition_global       = True
        #         print('time of GD stoppage = ', t)

        # A_mn_iter[f'epoch_{e+1}'][f'{t+1}']         = GL.A_mn.copy()
        # global_loss_valid[t,0]      = GL_valid.GlobalLoss_valid(x_dkf_global_valid, x_vfl_global_valid, t)
        global_loss_valid[0, t]      = GL_valid.GlobalLoss(x_dkf_global_valid, x_vfl_global_valid, t)

        # if avg_t_start <= t < avg_t_end:
        #     print('at t = ', t, X_dkf_valid[f'{1+1}'][0,t])
            # print(global_loss_valid[0, t])

        # if avg_t_start <= t < avg_t_end:
        #     for m in range(M):
        #         print(GL_valid.GlobalLoss_comp(m,x_dkf_global_valid, x_vfl_global_valid, t))
        #         global_loss_comp_valid[f'{m+1}']        += 1/(avg_t_end - avg_t_start) * GL_valid.GlobalLoss_comp(m,x_dkf_global_valid, x_vfl_global_valid, t)
        # global_loss_prev       = global_loss_curr
        # global_loss_curr       = global_loss_vec[f'epoch_{e+1}'][t,0]

        # for m in range(M):
        #     if not skip_condition_global:
        #         LL[f'{m+1}'].GradDescent(grad_x[f'{m+1}'],t)


        #     theta_iter[f'{m+1}'][f'epoch_{e+1}'][:,:,t]             = LL[f'{m+1}'].theta
        #     if e == epoch - 1 and t == training_time - 1:
        #         theta_est[f'{m+1}']         = LL[f'{m+1}'].theta


    X_dkf_concat        = np.zeros((p, avg_t_end - avg_t_start))
    X_dkf_concat_train  = np.zeros((p, avg_t_end - avg_t_start))
    for m in range(M):
        # print(f'global loss of comp_{m+1} = ', global_loss_comp_valid[f'{m+1}'])
        # print(f'del_R_l of comp_{m+1} = ', np.abs(np.mean(X_dkf_resd_valid[f'{m+1}'][0, avg_t_start:avg_t_end]) - np.mean(X_dkf_resd[f'{m+1}'][0, avg_t_start: avg_t_end])) / np.mean(X_dkf_resd[f'{m+1}'][0, avg_t_start: avg_t_end]) * 100)
        # print(f'del_R_a of comp_{m+1} = ', np.abs(np.mean(X_vfl_resd_valid[f'{m+1}'][0, avg_t_start:avg_t_end]) - np.mean(X_vfl_resd[f'run_{no_runs}'][f'epoch_{epoch}'][f'{m+1}'][0, avg_t_start:avg_t_end])) / np.mean(X_vfl_resd[f'run_{no_runs}'][f'epoch_{epoch}'][f'{m+1}'][0, avg_t_start:avg_t_end]) * 100)
        print(f'R_l validation of comp_{m+1} = ', np.mean(X_dkf_resd_valid[f'{m+1}'][0, avg_t_start:avg_t_end])**2)
        print(f'R_a validation of comp_{m+1} = ', np.mean(X_vfl_resd_valid[f'{m+1}'][0, avg_t_start:avg_t_end])**2)
        print(f'R_l training of comp_{m+1} = ', np.mean(X_dkf_resd[f'{m+1}'][0, avg_t_start: avg_t_end])**2)
        print(f'R_a training of comp_{m+1} = ', np.mean(X_vfl_resd[f'epoch_{epoch}'][f'{m+1}'][0, avg_t_start:avg_t_end])**2)
        # print('L_g = ', np.abs(np.mean(global_loss_valid[avg_t_start: avg_t_end, 0]) - np.mean(global_loss_vec[f'run_{no_runs}'][f'epoch_{epoch}'][avg_t_start: avg_t_end, 0])) / np.mean(global_loss_vec[f'run_{no_runs}'][f'epoch_{epoch}'][avg_t_start: avg_t_end, 0]) * 100)

        # print(f'Local loss at comp_{m+1} = ', np.mean(local_loss_valid[f'{m+1}'][0, avg_t_start: avg_t_end]))

    print('L_g = ', np.mean(global_loss_valid[0, avg_t_start: avg_t_end]))
        # print(f'L_g at t = {avg_t_start} = ', global_loss_valid[avg_t_start, 0])
        # X_dkf_concat[p_start_idx[m, 0]: p_start_idx[m, 0] + p_vec[m, 0], ]

    # for t in range(avg_t_start, avg_t_end):
    #     for m in range(M):
    #         X_dkf_concat[p_start_idx[m, 0]: p_start_idx[m, 0] + p_vec[m, 0], t - avg_t_start:t - avg_t_start+1]     = X_dkf_valid[f'{m+1}'][:,t:t+1]
    #         X_dkf_concat_train[p_start_idx[m, 0]: p_start_idx[m, 0] + p_vec[m, 0], t - avg_t_start:t - avg_t_start+1]     = X_dkf[f'{m+1}'][:,t:t+1]
    # # print('L_g = ', np.mean(np.abs(global_loss_valid[avg_t_start:avg_t_end, 0] - global_loss_train_use_vec[:,0]) / global_loss_train_use_vec))

    # global_loss_valid_normalized = np.mean(global_loss_valid[avg_t_start:avg_t_end, 0]) /  np.mean(np.linalg.norm(X_dkf_concat, axis = 0))
    # global_loss_train_normalized = np.mean(global_loss_vec[f'run_{no_runs}'][f'epoch_{epoch}'][avg_t_start:avg_t_end, 0]) / np.mean(np.linalg.norm(X_dkf_concat_train, axis = 0))
    # print('L_s_valid = ', np.mean(global_loss_valid[avg_t_start:avg_t_end, 0] /  np.linalg.norm(X_dkf_concat, axis = 0)))
    # print('L_s_train = ', np.mean(global_loss_vec[f'run_{no_runs}'][f'epoch_{epoch}'][avg_t_start:avg_t_end, 0] / np.linalg.norm(X_dkf_concat_train, axis = 0)))
    # print('L_g_relative = ', np.mean(np.abs(global_loss_valid_normalized - global_loss_train_normalized) / global_loss_train_normalized * 100))
    # print('L_g_relative = ', np.abs(global_loss_valid_normalized - global_loss_train_normalized) / global_loss_train_normalized * 100)
    # print_t = 1690
    # print('size of A_22 = ', LL_data['comp_2']['A'].shape)
    # print('size of theta = ', theta_est_avg['2'].shape)
    # print('size of Y_valid_2 = ', validation_data['comp_2']['Y'][:,t:t+1].shape)
    # print('size of A_21_est = ', A_mn_est_avg['21'].shape)
    # print('size of x_2_dkf_valid = ', X_dkf_pred_valid['2'][:,t:t+1].shape)

    # print('error_after_training')
    # print(np.linalg.norm(LL_data['comp_2']['A'] @ theta_est_avg['2'] @ validation_data['comp_2']['Y'][:,avg_t_start:avg_t_start+1] - A_mn_est_avg['21'] @ X_dkf_pred_valid['2'][:,avg_t_start:avg_t_start+1])**2)
    # print(np.linalg.norm(LL_data['comp_2']['A'] @ theta_est_avg['2'] @ LL_data['comp_2']['Y'][:,avg_t_start:avg_t_start+1] - A_mn_est_avg['21'] @ X_dkf_pred['2'][:,avg_t_start:avg_t_start+1])**2)

    # print(theta_iter[f'run_{r+1}'][f'epoch_{e+1}']['2'][:,:,avg_t_start].shape)

    # print('error_with_last_A_and_theta')
    # print(np.linalg.norm(LL_data['comp_2']['A'] @ theta_iter[f'run_{no_runs}'][f'epoch_{epoch}']['2'][:,:,avg_t_start] @ LL_data['comp_2']['Y'][:,avg_t_start:avg_t_start+1] - A_mn_iter[f'run_{no_runs}'][f'epoch_{epoch}'][f'{avg_t_start+1}']['21'] @ X_dkf_pred['2'][:,avg_t_start:avg_t_start+1])**2)
    # print(np.linalg.norm(LL_data['comp_2']['A'] @ theta_est[f'run_{no_runs}']['2'] @ validation_data['comp_2']['Y'][:,avg_t_start:avg_t_start+1] - A_mn_est_avg['21'] @ X_dkf_pred_valid['2'][:,avg_t_start:avg_t_start+1])**2)

    # print('error_computed_from_global_loss_function')
    # print(global_loss_vec[f'run_{no_runs}'][f'epoch_{epoch}'][avg_t_start, 0])
    # print(global_loss_valid[avg_t_start, 0])

    # print('theta_y and difference between x_vfl and x_dfk')
    # print(theta_iter[f'run_{no_runs}'][f'epoch_{epoch}']['2'][:,:,avg_t_start] @ LL_data['comp_2']['Y'][:,avg_t_start:avg_t_start+1], X_vfl[f'run_{no_runs}'][f'epoch_{epoch}']['2'][:, avg_t_start:avg_t_start+1] - X_dkf['2'][:, avg_t_start:avg_t_start+1])

    # frob_norm = GL_data['lambda_g'] * (np.linalg.norm(A_mn_est_avg['12'], 'fro')**2 + np.linalg.norm(A_mn_est_avg['21'], 'fro')**2)

    # print('frob_norm = ', frob_norm)


    # Y_valid             = ckf_data['Y'][:,training_time:]
    Y_valid             = ckf_data['Y_valid'][:,:validation_time]
    A_complete_valid    = ckf_data['A_complete_valid']
    # # print('shape of Y_valid = ', Y_valid.shape)

    B       = np.zeros((p, p))
    Q       = 0.0005 * np.eye(p)
    R       = 0.0005 * np.eye(d)
    P0      = Q
    x0      = np.zeros((p, 1))
    # # Getting the CKF values



    CKF_valid          = KalmanFilter(A_complete_valid, B, C_complete, Q, R, P0, x0)

    X_ckf_valid           = np.zeros((A_complete_valid.shape[0], validation_time))
    X_ckf_pred_valid      = np.zeros((A_complete_valid.shape[0], validation_time))
    # X_ckf_resd_valid      = np.zeros((C_complete.shape[0], validation_time))
    for t in range(validation_time):
        CKF_valid.predict()
        X_ckf_pred_valid[:,t:t+1]           = CKF_valid.get_state()
        # X_ckf_resd_valid[:,t:t+1]                 = CKF_valid.residual(Y_valid[:,t:t+1])

        for m in range(M):
            X_ckf_resd_valid[f'{m+1}'][0, t] = np.linalg.norm(CKF_valid.residual(Y_valid[:,t:t+1])[d_start_vec[m, 0]:d_start_vec[m, 0] + d_vec[m, 0],:])
        CKF_valid.update(Y_valid[:,t:t+1])
        # print('t = ', t)
        # print('X_ckf shape = ', X_ckf_valid[:,t:t+1].shape, ' CKF shape = ', CKF_valid.get_state().shape)
        X_ckf_valid[:,t:t+1]              = CKF_valid.get_state()
        


    # # # Recontructing A
    # # A_cap           = np.zeros((p, p))
    # # row_end_idx     = 0
    # # for i in range(M):
    # #     col_end_idx         = 0
    # #     row_start_idx       = row_end_idx
    # #     row_end_idx         = row_start_idx + p_vec[i, 0]
    # #     for j in range(M):
    # #         col_start_idx   = col_end_idx
    # #         col_end_idx     = col_start_idx + p_vec[j, 0]
    # #         if i == j:
    # #             A_cap[row_start_idx: row_end_idx, col_start_idx: col_end_idx]   = LL_data[f'comp_{i+1}']['A']
    # #         else:
    # #             A_cap[row_start_idx: row_end_idx, col_start_idx: col_end_idx]   = A_mn_est[f'{i+1}{j+1}']
                
    # for t in range(validation_time):

    #     for m in range(M):
    #         for j in range(M):
    #             if m == j:
    #                 X_dep_valid[f'{m+1}'][:,t:t+1]    += LL_data[f'comp_{m+1}']['A'] @ X_dkf_valid[f'{m+1}'][:,t:t+1]
    #             else:
    #                 X_dep_valid[f'{m+1}'][:,t:t+1]    += A_mn_est_avg[f'{m+1}{j+1}'] @ X_dkf_valid[f'{j+1}'][:,t:t+1]
                

    post_valid_data = {}
    post_valid_data['X_dkf']              = X_dkf_valid
    post_valid_data['X_dkf_pred']         = X_dkf_pred_valid
    post_valid_data['X_dkf_residual']     = X_dkf_resd_valid
    post_valid_data['X_vfl']              = X_vfl_valid
    post_valid_data['X_vfl_pred']         = X_vfl_pred_valid
    post_valid_data['X_vfl_residual']     = X_vfl_resd_valid
    post_valid_data['local_loss']         = local_loss_valid
    post_valid_data['global_loss']        = global_loss_valid
    # post_valid_data['spectral_radius']    = spectral_radius_list
    # post_valid_data['A_complete_true']    = A_complete
    # post_valid_data['C_complete_true']    = C_complete
    post_valid_data['X_ckf']              = X_ckf_valid
    post_valid_data['X_ckf_pred']         = X_ckf_pred_valid
    post_valid_data['X_ckf_residual']     = X_ckf_resd_valid
    # post_valid_data['X_ckf_pred_cap']     = X_ckf_pred_cap
    post_valid_data['X_dep']              = X_dep_valid
    # post_valid_data['A_cap']              = A_cap
    post_valid_data['A_mn_error']          = None

    save_valid_data       = validation_plots.DataSaver(post_valid_data, M, validation_time, total_time,  all_data.valid_location, p_vec, d_vec, p_start_idx)
    # save_data.save_X_pred()
    # save_data.save_local_loss()
    save_valid_data.save_all()



# # Code to remind me that the simulation has ended
# try:
#     # Use macOS 'say' command to speak through speakers
#     os.system('say "Your simulation has finished running"')
# except Exception as e:
#     print(f"Could not play audio alert: {e}")

