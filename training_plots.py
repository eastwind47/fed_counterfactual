import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


class DataSaver:
    def __init__(self, post_run_data, M, total_runs, total_epoch, training_time, total_time, results_location, p_vec, d_vec, p_start_idx):
        self.M                  = M
        self.total_runs         = total_runs
        self.training_time      = training_time
        self.total_time         = total_time
        self.results_location   = results_location
        self.total_epoch        = total_epoch
        self.X_dkf              = post_run_data['X_dkf']
        self.X_dkf_pred         = post_run_data['X_dkf_pred']
        self.X_dkf_residual     = post_run_data['X_dkf_residual']
        self.X_vfl              = post_run_data['X_vfl']
        self.X_vfl_pred         = post_run_data['X_vfl_pred']
        self.X_vfl_residual     = post_run_data['X_vfl_residual']
        self.local_loss         = post_run_data['local_loss']
        self.global_loss        = post_run_data['global_loss']
        self.global_loss_comp   = post_run_data['global_loss_comp']
        # self.spectral_radius    = post_run_data['spectral_radius']
        self.A_complete_true    = post_run_data['A_complete_true']
        self.C_complete_true    = post_run_data['C_complete_true']
        self.X_ckf_pred         = post_run_data['X_ckf_pred']
        self.X_ckf              = post_run_data['X_ckf']
        self.X_ckf_residual     = post_run_data['X_ckf_residual']
        # self.X_ckf_pred_cap     = post_run_data['X_ckf_pred_cap']
        self.X_dep              = post_run_data['X_dep']
        self.A_est              = post_run_data['A_cap']
        self.theta_est          = post_run_data['theta_cap']
        self.A_mn_error         = post_run_data['A_mn_error']

        self.p_vec              = p_vec
        self.d_vec              = d_vec
        self.p_start_idx        = p_start_idx

        self.cov_A_mn          = post_run_data['cov_A_mn']
        self.cov_theta         = post_run_data['cov_theta']

    def create_comp_folders(self):
        for m in range(self.M):
            comp_save_dir           = os.path.join(self.results_location, f'comp_{m+1}')
            os.makedirs(comp_save_dir, exist_ok = True)

    def save_X_pred(self, epoch):
        # Define the time vector
        time_vector = np.linspace(1, self.training_time, self.training_time)

        # ckf Index
        for m in range(self.M):
            # Iterate through each row in X_dkf and plot
            for i in range(self.p_vec[m, 0]):
                ckf_idx = self.p_start_idx[m, 0] + i
                plt.figure()
                plt.rcParams.update({'font.size': 15})
                plt.plot(time_vector, self.X_dkf_pred[f'{m+1}'][i,:], color='blue', label='X_clnt')
                plt.plot(time_vector, self.X_vfl_pred[f'epoch_{epoch}'][f'{m+1}'][i,:], color='orange', label='X_aug')
                plt.plot(time_vector, self.X_dep[f'epoch_{epoch}'][f'{m+1}'][i,:], color='magenta', label='X_ser')
                plt.plot(time_vector, self.X_ckf_pred[ckf_idx,:], color='green', label='X_cent')
                # if ckf_idx == 2:
                #     for t in range(1669, 1680):
                #         print(self.X_ckf_pred[ckf_idx, t])
                plt.xlabel('Time')
                plt.ylabel(f'X')
                plt.title(f'X of comp_{m+1} and dimension {i+1} vs Time')
                plt.legend()

                # Save the plot
                plot_filename = os.path.join(self.results_location, f'C{m+1}_X{i+1}_vs_t.png')
                plt.savefig(plot_filename)
                plt.close()

    # def save_local_loss(self, run, epoch):

    #     # Define the time vector
    #     time_vector = np.linspace(1, self.training_time, self.training_time)
        
    #     for m in range(self.M):
    #         # save_dir        = os.path.join(self.results_location, f'comp_{m+1}')
    #         # os.makedirs(save_dir, exist_ok=True)

    #         plt.figure()
    #         plt.rcParams.update({'font.size': 18})
    #         plt.plot(time_vector, self.local_loss[f'run_{run}'][f'epoch_{epoch}'][f'{m+1}'][:,0])
    #         plt.xlabel('Time')
    #         plt.ylabel(f'Local loss')
    #         plt.title(f'Local loss of component {m+1} vs time')
    #         # Save the plot
    #         # plot_filename = os.path.join(save_dir, f'LocalLoss_{m+1}.png')
    #         plot_filename = os.path.join(self.results_location, f'LocalLoss_{m+1}.png')
    #         plt.savefig(plot_filename)
    #         # plt.show()
    #         plt.close()



    # def save_local_loss(self, epoch):
    # # Define the time vector
    #     time_vector = np.linspace(1, self.training_time, self.training_time)
        
    #     for m in range(self.M):
    #         plt.figure()
    #         plt.rcParams.update({'font.size': 18})
    #         plt.plot(time_vector, self.local_loss[f'epoch_{epoch}'][f'{m+1}'][:, 0])
    #         plt.xlabel('Time')
    #         plt.ylabel(f'Local loss')
    #         plt.title(f'Local loss of component {m+1} vs time')

    #         # Save the plot
    #         plot_filename = os.path.join(self.results_location, f'LocalLoss_{m+1}.png')
    #         plt.savefig(plot_filename)
    #         plt.close()



    # def save_SR(self):
    #     if self.spectral_radius:
    #         time_vector = np.linspace(1, self.total_time, self.total_time)
            
    #         for m in range(self.M):
    #             # save_dir        = os.path.join(self.results_location, f'comp_{m+1}')
    #             # os.makedirs(save_dir, exist_ok=True)
                
    #             plt.figure()
    #             plt.rcParams.update({'font.size': 15})
    #             plt.plot(time_vector, self.spectral_radius[f'{m+1}'][:,0])
    #             plt.xlabel('Time')
    #             plt.ylabel('Spectral Radius')
    #             plt.title(f'Spectral Radius of system of component {m+1} vs time')
    #             # Save the plot
    #             # plot_filename = os.path.join(save_dir, f'SR_{m+1}_vs_time.png')
    #             plot_filename = os.path.join(self.results_location, f'SR_{m+1}_vs_time.png')
    #             plt.savefig(plot_filename)
    #             # plt.show()
    #             plt.close()



    def save_ckf_vs_vfl(self, epoch):
        time_vector = np.linspace(1, self.training_time, self.training_time)

        for m in range(self.M):
            # save_dir        = os.path.join(self.results_location, f'comp_{m+1}')
            # os.makedirs(save_dir, exist_ok=True)

            row_start_idx       = self.p_start_idx[m, 0]
            row_end_idx         = self.p_start_idx[m, 0] + self.p_vec[m, 0]
            plt.figure()
            plt.rcParams.update({'font.size': 14})
            plt.plot(time_vector, np.linalg.norm(self.X_ckf_pred[row_start_idx:row_end_idx,:] - self.X_vfl_pred[f'epoch_{epoch}'][f'{m+1}'], axis = 0), label = r'$||h_o-h_a||$')
            # plt.plot(time_vector, np.linalg.norm(self.X_dep[f'epoch_{epoch}'][f'{m+1}'] - self.X_vfl_pred[f'epoch_{epoch}'][f'{m+1}'], axis = 0), label = '||X_ser-X_aug||')
            plt.plot(time_vector, np.linalg.norm(self.X_ckf_pred[row_start_idx:row_end_idx,:] - self.X_dep[f'epoch_{epoch}'][f'{m+1}'], axis = 0), label = r'$||h_o-h_s||$')
            plt.plot(time_vector, np.linalg.norm(self.X_vfl_pred[f'epoch_{epoch}'][f'{m+1}'] - self.X_dkf_pred[f'{m+1}'], axis = 0), label = r'$||h_a-h_c||$')
            # plt.plot(time_vector, np.linalg.norm(self.X_dep[f'epoch_{epoch}'][f'{m+1}'] - self.X_dkf_pred[f'{m+1}'], axis = 0), label = '||X_clnt-X_ser||')
            # plt.plot(time_vector, np.linalg.norm(self.X_ckf_pred[row_start_idx:row_end_idx,:] - self.X_dkf_pred[f'{m+1}'], axis = 0), label = '||X_clnt-X_cent||')
            plt.xlabel('Time')
            plt.ylabel(r'$L_2 norm$')
            # plt.ylim((0, 0.02))
            # plt.title(f'||X_cent vs X_aug|| vs time for component{m+1}')
            plt.legend()
            # Save the plot
            # plot_filename = os.path.join(save_dir, f'SR_{m+1}_vs_time.png')
            plot_filename = os.path.join(self.results_location, f'C{m+1}_L2_norm_X_cent_vs_X_aug.png')
            plt.savefig(plot_filename)
            # plt.show()
            plt.close()

    # def save_A_mn_error(self, run, epoch):
    #     time_vector = np.linspace(1, self.training_time, self.training_time)

    #     save_dir    = os.path.join(self.results_location, 'A_mn_error')
    #     os.makedirs(save_dir, exist_ok=True)
    #     for m in range(self.M):
    #         # save_dir        = os.path.join(self.results_location, f'comp_{m+1}')
    #         # os.makedirs(save_dir, exist_ok=True)
    #         for n in range(self.M):
    #             if m != n:
    #                 plt.figure()
    #                 plt.rcParams.update({'font.size': 18})
    #                 plt.plot(time_vector, self.A_mn_error[f'run_{run}'][f'epoch_{epoch}'][f'{m+1}{n+1}'], label = f'||(A_{m+1}{n+1})_est - (A_{m+1}{n+1})_true||')
    #                 plt.xlabel('Time')
    #                 plt.ylabel('Frobenius Norm')
    #                 plt.title(f'||(A_{m+1}{n+1})_est - (A_{m+1}{n+1})_true|| vs time')
    #         # Save the plot
    #         # plot_filename = os.path.join(save_dir, f'SR_{m+1}_vs_time.png')
    #                 plot_filename = os.path.join(save_dir, f'A_{m+1}{n+1}_error.png')
    #                 plt.savefig(plot_filename)
    #         # plt.show()
    #                 plt.close()

    # def save_A_mn_error(self, epoch):
    #     time_vector = np.linspace(1, self.training_time, self.training_time)

    #     save_dir = os.path.join(self.results_location, 'A_mn_error')
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     for m in range(self.M):
    #         for n in range(self.M):
    #             if m != n:
    #                 plt.figure()
    #                 plt.rcParams.update({'font.size': 18})
    #                 plt.plot(time_vector, self.A_mn_error[f'epoch_{epoch}'][f'{m+1}{n+1}'], 
    #                         label=f'||(A_{m+1}{n+1})_est - (A_{m+1}{n+1})_true||')
    #                 plt.xlabel('Time')
    #                 plt.ylabel('Frobenius Norm')
    #                 plt.title(f'||(A_{m+1}{n+1})_est - (A_{m+1}{n+1})_true|| vs time')

    #                 # Save the plot
    #                 plot_filename = os.path.join(save_dir, f'A_{m+1}{n+1}_error.png')
    #                 plt.savefig(plot_filename)
    #                 plt.close()


    def save_residuals(self, epoch):
        time_vector = np.linspace(1, self.training_time, self.training_time)

        for m in range(self.M):
            plt.figure()
            plt.rcParams.update({'font.size': 15})
            plt.plot(time_vector, self.X_dkf_residual[f'{m+1}'].T, label='R_l')
            plt.plot(time_vector, self.X_vfl_residual[f'epoch_{epoch}'][f'{m+1}'].T, label='R_a')
            plt.plot(time_vector, self.X_ckf_residual[f'{m+1}'].T, label='R_cent')
            plt.xlabel('Time')
            plt.ylabel('Residuals')
            plt.title(f'Residuals for comp_{m+1} at epoch_{epoch}')
            plt.legend()

            plot_filename = os.path.join(self.results_location, f'C{m+1}_epoch{epoch}_resd')
            plt.savefig(plot_filename)
            plt.close()


    # def vfl_resd_heatmap(self, epoch):

    #     data_matrix         = np.zeros((epoch, self.training_time))

    #     for e in range(epoch):
    #         for t in range(self.training_time):

    #             data_matrix[e, t] 

    def save_loss_vs_epoch(self, time):

        epoch_vec           = np.linspace(1, self.total_epoch, self.total_epoch)

        # for m in range(self.M):
        #     error_mat       = np.zeros((self.total_epoch, self.total_runs))

        #     for r in range(self.total_runs):
        #         for e in range(self.total_epoch):
        #             error_mat[e, r]         = self.local_loss[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}'][0, time]

        #     print(f"Error matrix for global loss at time {time}:\n{error_mat}")
        #     df              = pd.DataFrame(error_mat, columns = [f'Run_{r+1}' for r in range(self.total_runs)])
        #     df_melted       = df.melt(var_name = "Run", value_name = "LocalLoss", ignore_index = False)
        #     print(f"Melted DataFrame for global loss at time {time}:\n{df_melted}")
        #     df_melted['Epoch']      = np.tile(np.arange(1, error_mat.shape[0]+1), error_mat.shape[1])

        #     plt.figure()
        #     plt.rcParams.update({'font.size': 15})
        #     sns.lineplot(data=df_melted, x = "Epoch", y = "LocalLoss", errorbar = "ci")
        #     # plt.ylim((0,0.8))
        #     plt.xlabel('epoch')
        #     plt.ylabel(rf'$(L_{m+1})_a$')
        #     # plt.title(f'Local Loss of comp_{m+1} vs epoch at time t = {time}')
        #     plt.grid(True, linestyle = ':', linewidth = 0.5)
        #     plot_filename       = os.path.join(self.results_location, f'C{m+1}_LocalLoss_vs_epoch_t_{time}.png')
        #     plt.savefig(plot_filename, dpi = 800, bbox_inches = 'tight')
        #     plt.close()

        # for r in range(self.total_runs):
        #     for e in range(self.total_epoch):
        #         error_mat[e, r]         = self.global_loss[f'run_{r+1}'][f'epoch_{e+1}'][0, time]


        # df              = pd.DataFrame(error_mat, columns = [f'Run_{r+1}' for r in range(self.total_runs)])
        # df_melted       = df.melt(var_name = "Run", value_name = "GlobalLoss", ignore_index = False)
        # df_melted['Epoch']      = np.tile(np.arange(1, error_mat.shape[0]+1), error_mat.shape[1])

        # plt.figure()
        # plt.rcParams.update({'font.size': 15})
        # sns.lineplot(data=df_melted, x = "Epoch", y = "GlobalLoss", errorbar = "ci")
        # # plt.ylim((0,0.8))
        # plt.xlabel('epoch')
        # plt.ylabel(r'$L_s$')
        # # plt.ylim((0, 0.1))
        # plt.grid(True, linestyle = ':', linewidth = 0.5)
        # # plt.title(f'Global Loss vs epoch at time t = {time}')
        # plot_filename       = os.path.join(self.results_location, f'GlobalLoss_vs_epoch_t_{time}.png')
        # plt.savefig(plot_filename, dpi = 800, bbox_inches = 'tight')
        # plt.close()



        global_loss_vec         = np.zeros((self.total_epoch, 1))

        for e in range(self.total_epoch):
            global_loss_vec[e, 0]        = self.global_loss[f'run_{self.total_runs}'][f'epoch_{e+1}'][0, time]

        plt.figure()
        plt.plot(epoch_vec, global_loss_vec)
        plt.xlabel('Epoch')
        plt.ylabel('Global Loss')
        plt.title(f'Global loss vs epoch at time t = {time}')
        plot_filename       = os.path.join(self.results_location, f'GlobalLoss_vs_epoch_t_{time}')
        plt.savefig(plot_filename)
        

    # def save_A_mn_vs_epoch(self, run, time):

    #     epoch_vec           = np.linspace(1, self.total_epoch, self.total_epoch)

    #     save_dir    = os.path.join(self.results_location, 'A_mn_error')
    #     os.makedirs(save_dir, exist_ok=True)
    #     for m in range(self.M):
    #         for n in range(self.M):
    #             A_mn_epoch_vec      = np.zeros((self.total_epoch, 1))
    #             if m != n:
                    

    #                 for e in range(self.total_epoch):
    #                     A_mn_epoch_vec[e, 0]        = self.A_mn_error[f'run_{run}'][f'epoch_{e+1}'][f'{m+1}{n+1}'][time, 0]

    #             plt.figure()
    #             plt.rcParams.update({'font.size': 18})
    #             plt.plot(epoch_vec, A_mn_epoch_vec, label = f'Amn_{m+1}{n+1}')
    #             plt.xlabel('epoch')
    #             plt.ylabel(rf'$||A_{m+1,n+1}_{{est}} - A_{m+1,n+1}_{{true}}||_{{F}}$')
    #             # plt.title(rf'$A_{m+1}{n+1}_error vs epoch at time t = {time}$')
    #             plt.grid(True, linestyle = ':', linewidth = 0.5)
    #             plot_filename       = os.path.join(save_dir, f'A_{m+1}{n+1}_error_vs_epoch.png')
    #             plt.savefig(plot_filename, dpi = 800, bbox_inches = 'tight')
    #             plt.close()

    def save_A_mn_vs_epoch_CI(self, time):

        save_dir    = os.path.join(self.results_location, 'A_mn_error')
        os.makedirs(save_dir, exist_ok=True)

        for m in range(self.M):
            for n in range(self.M):

                error_mat           = np.zeros((self.total_epoch, self.total_runs))

                if m != n:

                    for r in range(self.total_runs):
                        for e in range(self.total_epoch):
                            error_mat[e, r]         = self.A_mn_error[f'run_{r+1}'][f'epoch_{e+1}'][f'{m+1}{n+1}'][0, time]

                
                    df              = pd.DataFrame(error_mat, columns = [f'Run_{r+1}' for r in range(self.total_runs)])
                    df_melted       = df.melt(var_name = "Run", value_name = "Error", ignore_index = False)
                    df_melted['Epoch']      = np.tile(np.arange(1, error_mat.shape[0]+1), error_mat.shape[1])

                    plt.figure()
                    plt.rcParams.update({'font.size': 15})
                    sns.lineplot(data=df_melted, x = "Epoch", y = "Error", errorbar = "ci")
                    # plt.ylim((0,0.8))
                    plt.xlabel('epoch')
                    plt.ylabel(rf'$||{{\hat{{A}}_{{{m+1}{n+1}}}}} - {{A_{{{m+1}{n+1}}}}}||_{{F}}$')
                    # plt.title(rf'$A_{m+1}{n+1}_error vs epoch at time t = {time}$')
                    plt.grid(True, linestyle = ':', linewidth = 0.5)
                    plot_filename       = os.path.join(save_dir, f'A_{m+1}{n+1}_error_vs_epoch_CI.png')
                    plt.savefig(plot_filename, dpi = 800, bbox_inches = 'tight')
                    plt.close()
                    


    def save_final_Aest(self):
        save_path       = os.path.join(self.results_location, 'A_est.csv')
        df_A_est        = pd.DataFrame(self.A_est)
        df_A_est.to_csv(save_path, header=None, index=None)

    # def save_final_Aest(self):

    #     final_A_est         = np.zeros(self.A_est['run_1'].shape)
    #     for r in range(self.total_runs):
    #         final_A_est         += (1/self.total_runs) * self.A_est[f'run_{r+1}']


    #     save_path       = os.path.join(self.results_location, 'A_est.csv')
    #     df_A_est        = pd.DataFrame(final_A_est)
    #     df_A_est.to_csv(save_path, header=None, index=None)
    #     pd.read_csv(save_path)

    def save_theta_est(self):
        
        for m in range(self.M):
            # save_dir        = os.path.join(self.results_location, f'comp_{m+1}')
            # os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(self.results_location, f'C{m+1}_theta_est.csv')
            df_theta_est    = pd.DataFrame(self.theta_est[f'{m+1}'])
            df_theta_est.to_csv(save_path, header = None, index = None)

    
    def save_covariance_traces(self):
        """
        Compute the trace of covariance matrices and plot them against iterations.
        """

        # Plot Trace(Cov(A_mn)) vs Iteration
        for key, cov_matrices in self.cov_A_mn.items():
            # Compute the trace for each iteration
            trace_values = np.trace(cov_matrices, axis1=0, axis2=1)

            # Save the trace values to a CSV file
            trace_csv_path = os.path.join(self.results_location, f"Trace_Cov_A_mn_{key}.csv")
            np.savetxt(trace_csv_path, trace_values, delimiter=",")
            print(f"Saved Trace(Cov(A_mn_{key})) to {trace_csv_path}")

            # Plot the trace values
            plt.figure()
            plt.plot(range(len(trace_values)), trace_values)
            plt.xlabel('Iteration')
            plt.ylabel(f'Trace(Cov(A_mn_{key}))')
            plt.title(f'Trace(Cov(A_mn_{key})) vs Iteration')
            plt.grid(True, linestyle=':', linewidth=0.5)
            plot_filename = os.path.join(self.results_location, f'Trace_Cov_A_mn_{key}.png')
            plt.savefig(plot_filename, dpi=800, bbox_inches='tight')
            plt.close()

        # Plot Trace(Cov(theta)) vs Iteration
        for key, cov_matrices in self.cov_theta.items():
            # Compute the trace for each iteration
            trace_values = np.trace(cov_matrices, axis1=0, axis2=1)

            # Save the trace values to a CSV file
            trace_csv_path = os.path.join(self.results_location, f"Trace_Cov_theta_{key}.csv")
            np.savetxt(trace_csv_path, trace_values, delimiter=",")
            print(f"Saved Trace(Cov(theta_{key})) to {trace_csv_path}")

            # Plot the trace values
            plt.figure()
            plt.plot(range(len(trace_values)), trace_values)
            plt.xlabel('Iteration')
            plt.ylabel(f'Trace(Cov(theta_{key}))')
            plt.title(f'Trace(Cov(theta_{key})) vs Iteration')
            plt.grid(True, linestyle=':', linewidth=0.5)
            plot_filename = os.path.join(self.results_location, f'Trace_Cov_theta_{key}.png')
            plt.savefig(plot_filename, dpi=800, bbox_inches='tight')
            plt.close()


    # Tracking a covariance matrix element
    def track_covariance_element(self, matrix_type, key, row_idx, col_idx):
        """
        Track how a specific element in the covariance matrix evolves over iterations.

        Parameters:
        - matrix_type (str): Type of covariance matrix ('A_mn' or 'theta').
        - key (str): Key identifying the specific matrix (e.g., '12' for A_mn or '1' for theta).
        - row_idx (int): Row index of the element to track.
        - col_idx (int): Column index of the element to track.
        """
        if matrix_type == 'A_mn':
            cov_matrices = self.cov_A_mn[key]
        elif matrix_type == 'theta':
            cov_matrices = self.cov_theta[key]
        else:
            raise ValueError("Invalid matrix_type. Use 'A_mn' or 'theta'.")

        # Extract the specific element across all iterations
        element_values = cov_matrices[row_idx, col_idx, :]

        # Save the element values to a CSV file
        element_csv_path = os.path.join(self.results_location, f"Cov_{matrix_type}_{key}_element_{row_idx}_{col_idx}.csv")
        np.savetxt(element_csv_path, element_values, delimiter=",")
        print(f"Saved evolution of element ({row_idx}, {col_idx}) in Cov({matrix_type}_{key}) to {element_csv_path}")

        # Plot the evolution of the element
        plt.figure()
        plt.plot(range(len(element_values)), element_values)
        plt.xlabel('Iteration')
        plt.ylabel(f'Cov({matrix_type}_{key})[{row_idx}, {col_idx}]')
        plt.title(f'Evolution of Cov({matrix_type}_{key})[{row_idx}, {col_idx}]')
        plt.grid(True, linestyle=':', linewidth=0.5)
        print("code reached here")
        plot_filename = os.path.join(self.results_location, f"Cov_{matrix_type}_{key}_element_{row_idx}_{col_idx}.png")
        plt.savefig(plot_filename, dpi=800, bbox_inches='tight')
        plt.close()


    # Plotting the largest eigenvalue of the covariance matrix
    def save_covariance_eigenvalues(self):
        """
        Compute the largest eigenvalue of covariance matrices and plot them against iterations.
        """

        # Plot Largest Eigenvalue of Cov(A_mn) vs Iteration
        for key, cov_matrices in self.cov_A_mn.items():
            # Compute the largest eigenvalue for each iteration
            largest_eigenvalues = np.linalg.eigvalsh(cov_matrices).max(axis=0)

            # Save the largest eigenvalues to a CSV file
            eigen_csv_path = os.path.join(self.results_location, f"Largest_Eigenvalue_Cov_A_mn_{key}.csv")
            np.savetxt(eigen_csv_path, largest_eigenvalues, delimiter=",")
            print(f"Saved Largest Eigenvalue(Cov(A_mn_{key})) to {eigen_csv_path}")

            # Plot the largest eigenvalues
            plt.figure()
            plt.plot(range(len(largest_eigenvalues)), largest_eigenvalues)
            plt.xlabel('Iteration')
            plt.ylabel(f'Largest Eigenvalue(Cov(A_mn_{key}))')
            plt.title(f'Largest Eigenvalue(Cov(A_mn_{key})) vs Iteration')
            plt.grid(True, linestyle=':', linewidth=0.5)
            plot_filename = os.path.join(self.results_location, f"Largest_Eigenvalue_Cov_A_mn_{key}.png")
            plt.savefig(plot_filename, dpi=800, bbox_inches='tight')
            plt.close()

        # Plot Largest Eigenvalue of Cov(theta) vs Iteration
        for key, cov_matrices in self.cov_theta.items():
            # Compute the largest eigenvalue for each iteration
            largest_eigenvalues = np.linalg.eigvalsh(cov_matrices).max(axis=0)

            # Save the largest eigenvalues to a CSV file
            eigen_csv_path = os.path.join(self.results_location, f"Largest_Eigenvalue_Cov_theta_{key}.csv")
            np.savetxt(eigen_csv_path, largest_eigenvalues, delimiter=",")
            print(f"Saved Largest Eigenvalue(Cov(theta_{key})) to {eigen_csv_path}")

            # Plot the largest eigenvalues
            plt.figure()
            plt.plot(range(len(largest_eigenvalues)), largest_eigenvalues)
            plt.xlabel('Iteration')
            plt.ylabel(f'Largest Eigenvalue(Cov(theta_{key}))')
            plt.title(f'Largest Eigenvalue(Cov(theta_{key})) vs Iteration')
            plt.grid(True, linestyle=':', linewidth=0.5)
            plot_filename = os.path.join(self.results_location, f"Largest_Eigenvalue_Cov_theta_{key}.png")
            plt.savefig(plot_filename, dpi=800, bbox_inches='tight')
            plt.close()



    def save_all(self, time, epoch):
        self.save_X_pred(epoch)
        # self.save_local_loss(run, epoch)
        # self.save_global_loss(run, epoch)
        # self.save_SR()
        self.save_ckf_vs_vfl(epoch)
        self.save_residuals(epoch)
        self.save_loss_vs_epoch(time)
        self.save_A_mn_vs_epoch_CI(time)
        self.save_final_Aest()
        self.save_theta_est()
        self.save_covariance_traces()

        self.track_covariance_element(matrix_type='A_mn', key='21', row_idx=0, col_idx=0)



