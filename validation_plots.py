import matplotlib.pyplot as plt
import numpy as np
import os
# import pandas as pd


class DataSaver:
    def __init__(self, post_valid_data, M, validation_time, total_time, results_location, p_vec, d_vec, p_start_idx):
        self.M                  = M
        self.validation_time      = validation_time
        self.total_time         = total_time
        self.results_location   = results_location
        self.X_dkf              = post_valid_data['X_dkf']
        self.X_dkf_pred         = post_valid_data['X_dkf_pred']
        self.X_dkf_residual     = post_valid_data['X_dkf_residual']
        self.X_vfl              = post_valid_data['X_vfl']
        self.X_vfl_pred         = post_valid_data['X_vfl_pred']
        self.X_vfl_residual     = post_valid_data['X_vfl_residual']
        self.local_loss         = post_valid_data['local_loss']
        self.global_loss        = post_valid_data['global_loss']
        # self.spectral_radius    = post_valid_data['spectral_radius']
        # self.A_complete_true    = post_valid_data['A_complete_true']
        # self.C_complete_true    = post_valid_data['C_complete_true']
        self.X_ckf_pred         = post_valid_data['X_ckf_pred']
        self.X_ckf              = post_valid_data['X_ckf']
        self.X_ckf_residual     = post_valid_data['X_ckf_residual']
        self.X_dep              = post_valid_data['X_dep']
        # self.A_est              = post_valid_data['A_cap']
        self.A_mn_error         = post_valid_data['A_mn_error']

        self.p_vec              = p_vec
        self.d_vec              = d_vec
        self.p_start_idx        = p_start_idx

        os.makedirs(self.results_location, exist_ok=True)

        # try:
        #     os.makedirs(self.results_location, exist_ok=True)
        #     print(f"Directory created at {self.results_location}")
        # except Exception as e:
        #     print(f"Error creating directory: {e}")



    def create_comp_folders(self):
        for m in range(self.M):
            comp_save_dir = os.path.join(self.results_location, f'comp_{m+1}')
            os.makedirs(comp_save_dir, exist_ok=True)

    def save_X_pred(self):
        # Define the time vector
        time_vector = np.linspace(1, self.validation_time, self.validation_time)

        for m in range(self.M):
            for i in range(self.p_vec[m, 0]):
                ckf_idx = self.p_start_idx[m, 0] + i
                plt.figure()
                plt.plot(time_vector, self.X_dkf_pred[f'{m+1}'][i, :], color='blue', label='X_clnt')
                plt.plot(time_vector, self.X_vfl_pred[f'{m+1}'][i, :], color='orange', label='X_aug')
                plt.plot(time_vector, self.X_dep[f'{m+1}'][i, :], color='magenta', label='X_ser')
                plt.plot(time_vector, self.X_ckf_pred[ckf_idx, :], color='green', label='X_cent')
                plt.xlabel('Time')
                plt.ylabel('X')
                plt.title(f'X of comp_{m+1} and dimension {i+1} vs Time')
                plt.legend()
                plot_filename = os.path.join(self.results_location, f'C{m+1}_X{i+1}_vs_t.png')
                plt.savefig(plot_filename)
                plt.close()

    def save_local_loss(self):
        time_vector = np.linspace(1, self.validation_time, self.validation_time)

        for m in range(self.M):
            plt.figure()
            plt.plot(time_vector, self.local_loss[f'{m+1}'][0, :])
            plt.xlabel('Time')
            plt.ylabel('Local loss')
            plt.title(f'Local loss of component {m+1} vs time')
            plot_filename = os.path.join(self.results_location, f'LocalLoss_{m+1}.png')
            plt.savefig(plot_filename)
            plt.close()

    def save_global_loss(self):
        time_vector = np.linspace(10, self.validation_time, self.validation_time - 10)

        plt.figure()
        plt.plot(time_vector, self.global_loss[0,10:])
        plt.xlabel('Time')
        plt.ylabel('Global Loss')
        plt.title('Global loss vs time')
        plot_filename = os.path.join(self.results_location, 'GlobalLoss_vs_t.png')
        plt.savefig(plot_filename)
        plt.close()

    def save_ckf_vs_vfl(self):
        time_vector = np.linspace(1, self.validation_time, self.validation_time)

        for m in range(self.M):
            # save_dir        = os.path.join(self.results_location, f'comp_{m+1}')
            # os.makedirs(save_dir, exist_ok=True)

            # X_vfl_pred_avg = np.mean([self.X_vfl_pred[f'run_{r+1}'][f'epoch_{epoch}'][f'{m+1}'] for r in range(self.total_runs)], axis = 0)
            # X_dep_avg = np.mean([self.X_dep[f'run_{r+1}'][f'epoch_{epoch}'][f'{m+1}'] for r in range(self.total_runs)], axis = 0)

            row_start_idx       = self.p_start_idx[m, 0]
            row_end_idx         = self.p_start_idx[m, 0] + self.p_vec[m, 0]
            plt.figure()
            plt.rcParams.update({'font.size': 22})
            plt.plot(time_vector, np.linalg.norm(self.X_ckf_pred[row_start_idx:row_end_idx,:] - self.X_vfl_pred[f'{m+1}'], axis = 0), label = r'$||h_o-h_a||$')
            # plt.plot(time_vector, np.linalg.norm(self.X_dep[f'epoch_{epoch}'][f'{m+1}'] - self.X_vfl_pred[f'epoch_{epoch}'][f'{m+1}'], axis = 0), label = '||X_ser-X_aug||')
            plt.plot(time_vector, np.linalg.norm(self.X_ckf_pred[row_start_idx:row_end_idx,:] - self.X_dep[f'{m+1}'], axis = 0), label = r'$||h_o-h_s||$')
            plt.plot(time_vector, np.linalg.norm(self.X_vfl_pred[f'{m+1}'] - self.X_dkf_pred[f'{m+1}'], axis = 0), label = r'$||h_a-h_c||$')
            # plt.plot(time_vector, np.linalg.norm(self.X_dep[f'epoch_{epoch}'][f'{m+1}'] - self.X_dkf_pred[f'{m+1}'], axis = 0), label = '||X_clnt-X_ser||')
            # plt.plot(time_vector, np.linalg.norm(self.X_ckf_pred[row_start_idx:row_end_idx,:] - self.X_dkf_pred[f'{m+1}'], axis = 0), label = '||X_clnt-X_cent||')

            # if m == 1:
                # print(np.linalg.norm(self.X_ckf_pred[row_start_idx:row_end_idx,:] - self.X_vfl_pred[f'{m+1}'], axis = 0)[1669:1680])
                # print(np.linalg.norm(self.X_ckf_pred[row_start_idx:row_end_idx,:] - self.X_dep[f'{m+1}'], axis = 0)[1669:1680])
                # print(np.linalg.norm(self.X_vfl_pred[f'{m+1}'] - self.X_dkf_pred[f'{m+1}'], axis = 0)[1669:1680])

            plt.xlabel('Time')
            plt.ylabel(r'$l_2$ norm')
            # plt.ylim((0, 0.02))
            # plt.title(f'||X_cent vs X_aug|| vs time for component{m+1}')
            plt.legend()
            plt.grid(True, linestyle = ':', linewidth = 0.5)
            # Save the plot
            # plot_filename = os.path.join(save_dir, f'SR_{m+1}_vs_time.png')
            plot_filename = os.path.join(self.results_location, f'C{m+1}_L2_norm_X_cent_vs_X_aug.png')
            plt.savefig(plot_filename, dpi = 800, bbox_inches = 'tight')
            # plt.show()
            plt.close()

    # def save_SR(self):
    #     time_vector = np.linspace(1, self.total_time, self.total_time)

    #     for m in range(self.M):
    #         plt.figure()
    #         plt.plot(time_vector, self.spectral_radius[f'{m+1}'][:, 0])
    #         plt.xlabel('Time')
    #         plt.ylabel('Spectral Radius')
    #         plt.title(f'Spectral Radius of system of component {m+1} vs time')
    #         plot_filename = os.path.join(self.results_location, f'SR_{m+1}_vs_time.png')
    #         plt.savefig(plot_filename)
    #         plt.close()

    # def save_ckf_vs_vfl(self):
    #     time_vector = np.linspace(1, self.validation_time, self.validation_time)

    #     for m in range(self.M):
    #         row_start_idx = self.p_start_idx[m, 0]
    #         row_end_idx = self.p_start_idx[m, 0] + self.p_vec[m, 0]
    #         plt.figure()
    #         plt.plot(time_vector, np.linalg.norm(self.X_ckf_pred[row_start_idx:row_end_idx, :] - self.X_vfl_pred[f'{m+1}'], axis=0), label='||X_cent-X_aug||')
    #         plt.plot(time_vector, np.linalg.norm(self.X_dep[f'{m+1}'] - self.X_vfl_pred[f'{m+1}'], axis=0), label='||X_ser-X_aug||')
    #         plt.plot(time_vector, np.linalg.norm(self.X_ckf_pred[row_start_idx:row_end_idx, :] - self.X_dep[f'{m+1}'], axis=0), label='||X_cent-X_ser||')
    #         plt.xlabel('Time')
    #         plt.ylabel('L2_norm')
    #         plt.title(f'||X_cent vs X_aug|| vs time for component {m+1}')
    #         plt.legend()
    #         plot_filename = os.path.join(self.results_location, f'C{m+1}_L2_norm_X_cent_vs_X_aug.png')
    #         plt.savefig(plot_filename)
    #         plt.close()

    # def save_A_mn_error(self):
    #     time_vector = np.linspace(1, self.validation_time, self.validation_time)

    #     save_dir = os.path.join(self.results_location, 'A_mn_error')
    #     os.makedirs(save_dir, exist_ok=True)

    #     for m in range(self.M):
    #         for n in range(self.M):
    #             if m != n:
    #                 plt.figure()
    #                 plt.plot(time_vector, self.A_mn_error[f'{m+1}{n+1}'], label=f'||(A_{m+1}{n+1})_est - (A_{m+1}{n+1})_true||')
    #                 plt.xlabel('Time')
    #                 plt.ylabel('Frobenius Norm')
    #                 plt.title(f'||(A_{m+1}{n+1})_est - (A_{m+1}{n+1})_true|| vs time')
    #                 plot_filename = os.path.join(save_dir, f'A_{m+1}{n+1}_error.png')
    #                 plt.savefig(plot_filename)
    #                 plt.close()

    # def save_final_Aest(self):
    #     save_path = os.path.join(self.results_location, 'A_est.csv')
    #     df_A_est = pd.DataFrame(self.A_est)
    #     df_A_est.to_csv(save_path, header=None, index=None)
    #     pd.read_csv(save_path)

    def save_residuals(self):
        time_vector = np.linspace(1, self.validation_time, self.validation_time)

        for m in range(self.M):
            plt.figure()
            plt.rcParams.update({'font.size': 15})
            plt.plot(time_vector, self.X_dkf_residual[f'{m+1}'].T, label='R_l')
            plt.plot(time_vector, self.X_vfl_residual[f'{m+1}'].T, label='R_a')
            plt.plot(time_vector, self.X_ckf_residual[f'{m+1}'].T, label='R_cent')
            plt.xlabel('Time')
            plt.ylabel('Residuals')
            plt.title(f'Residuals for comp_{m+1}')
            plt.legend()

            plot_filename = os.path.join(self.results_location, f'C{m+1}_resd')
            plt.savefig(plot_filename)
            plt.close()

    def save_all(self):
        self.save_X_pred()
        self.save_local_loss()
        self.save_global_loss()
        self.save_ckf_vs_vfl()
        self.save_residuals()
        # self.save_SR()
        # self.save_ckf_vs_vfl()
        # self.save_final_Aest()
