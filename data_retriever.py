# Importing all the necessary packages
import configparser
import os
import pandas as pd
import numpy as np

from kalman_filter import KalmanFilter


def _run_dkf_baseline(A: np.ndarray,
                      B: np.ndarray,
                      C: np.ndarray,
                      Q: np.ndarray,
                      R: np.ndarray,
                      P0: np.ndarray,
                      x0: np.ndarray,
                      Y: np.ndarray,
                      U: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a centralized Kalman filter to produce cooperative trajectories."""
    T = Y.shape[1]
    dkf = KalmanFilter(A, B, C, Q, R, P0.copy(), x0.copy())
    p_m = A.shape[0]
    s_m = B.shape[1]

    X_dkf = np.zeros((p_m, T))
    X_dkf_pred = np.zeros((p_m, T))
    X_dkf_resd = np.zeros((1, T))

    for t in range(T):
        u_prev = U[:, t - 1:t] if t > 0 else np.zeros((s_m, 1))
        dkf.predict(u_prev)
        X_dkf_pred[:, t:t + 1] = dkf.get_state()
        residual = dkf.residual(Y[:, t:t + 1])
        X_dkf_resd[0, t:t + 1] = np.linalg.norm(residual)
        dkf.update(Y[:, t:t + 1])
        X_dkf[:, t:t + 1] = dkf.get_state()

    return X_dkf, X_dkf_pred, X_dkf_resd


def _residual_norm_from_pred(C: np.ndarray,
                             X_dkf_pred: np.ndarray,
                             Y: np.ndarray) -> np.ndarray:
    """Compute per-time residual norms from predicted DKF states."""
    y_hat = C @ X_dkf_pred
    return np.linalg.norm(Y - y_hat, axis=0, keepdims=True)

class RetrieveData:
    def __init__(self, ini_file_name):
        """
        Initialize the class with configuration file name
        """
        # config.ini file name
        self.ini_file_name = ini_file_name

        # Loading the configuration file
        self.config = configparser.ConfigParser()
        self.config.read(self.ini_file_name)

        # Number of components
        self.num_components = int(self.config['COMPONENTS']['num_components'])

        # Total time for the time series
        self.total_time = int(self.config['COMPONENTS']['total_time'])

        # Global learning rates/weights
        self.alpha_a = float(self.config['GLOBAL_PARAMETERS']['alpha_a'])
        self.alpha_b = float(self.config['GLOBAL_PARAMETERS']['alpha_b'])
        self.xi = float(self.config['GLOBAL_PARAMETERS']['xi'])

        # Global regularization rate
        self.global_regularization_rate = float(
            self.config['GLOBAL_PARAMETERS']['lambda_g']
        )

        # Proportion of training data
        self.training_prop = float(eval(self.config['TRAINING']['train_set_prop']))

        # Training time
        self.training_time = int(np.ceil(self.training_prop * self.total_time))

        # Validation time
        # self.validation_time = 10000  # Commented out

        # Initializing data from each component
        self.local_learners_pack = None

        # Initializing diagonal matrix data for the global learner
        self.global_learners_pack = None

        # Initializing data for the HMatrix
        self.HMatrix_pack = None

        # Validation pack
        # self.validation_pack = None  # Commented out

        # Initializing vector that stores size of states in each component
        self.comp_size_vec = None

        # Size of the global state vector
        self.global_state_size = None

        # Output size vector 
        self.output_size_vec = None

        # Global output size
        self.global_output_size = None

        # Input size bookkeeping
        self.input_size_vec = None
        self.global_input_size = None

        # Input start indices (for reconstruction)
        self.input_start_vec = np.zeros((self.num_components, 1), dtype=int)

        # State start indices (per-component offsets within the global state)
        self.comp_start_vec = np.zeros((self.num_components, 1), dtype=int)

        # Output data location 
        self.results_location = self.config['LOCATION']['results_location']

        # Validation results location 
        # self.valid_location = self.config['LOCATION']['valid_location']  # Commented out

        # Centralized Kalman Filter pack
        self.CKF_pack = None

        # Global Y
        self.global_Y = None

        # Global Y valid   
        # self.global_Y_valid = None  # Commented out

        # Output start indices
        self.output_start_vec = np.zeros((self.num_components, 1), dtype=int)

        # Preparing the data
        self.PrepareData()

    def PrepareData(self):
        """
        Get the data for each component.
        
        Outputs:
        - M: Number of components
        - T: Length of time series data
        - components: A nested dictionary containing properties of all components and the corresponding observation time series
        """
        try:
            # Getting the data folder location from config file
            data_location = self.config['LOCATION']['data_location']
            # valid_data_location = self.config['LOCATION']['valid_data_location']  # Commented out

            # Client-side hyper-parameters
            lambda_l = [float(x.strip()) for x in self.config['LOCAL_PARAMETERS']['lambda_l'].split(',') if x.strip()]
            eta_1 = [float(x.strip()) for x in self.config['LOCAL_PARAMETERS']['eta_1'].split(',') if x.strip()]
            eta_2 = [float(x.strip()) for x in self.config['LOCAL_PARAMETERS']['eta_2'].split(',') if x.strip()]
            gamma_1 = [float(x.strip()) for x in self.config['LOCAL_PARAMETERS']['gamma_1'].split(',') if x.strip()]
            gamma_2 = [float(x.strip()) for x in self.config['LOCAL_PARAMETERS']['gamma_2'].split(',') if x.strip()]

            def expand(values, name):
                if len(values) == 1 and self.num_components > 1:
                    return values * self.num_components
                if len(values) != self.num_components:
                    raise ValueError(
                        f"Expected {self.num_components} entries for {name} but received {len(values)}"
                    )
                return values

            lambda_l = expand(lambda_l, 'lambda_l')
            eta_1 = expand(eta_1, 'eta_1')
            eta_2 = expand(eta_2, 'eta_2')
            gamma_1 = expand(gamma_1, 'gamma_1')
            gamma_2 = expand(gamma_2, 'gamma_2')

            # Dictionary to store all component data and global learner package
            self.local_learners_pack = {}
            self.global_learners_pack = {}
            self.HMatrix_pack = {}
            self.CKF_pack = {}
            # self.validation_pack = {}  # Commented out
            self.global_Y = {}
            self.global_U = {}
            # self.global_Y_valid = {}  # Commented out
            A_mm = {}
            B_mm = {}
            # Initializing total size of state vector
            self.comp_size_vec = np.zeros((self.num_components, 1), dtype=int)
            self.global_state_size = 0

            self.output_size_vec = np.zeros((self.num_components, 1), dtype=int)
            self.global_output_size = 0

            self.input_size_vec = np.zeros((self.num_components, 1), dtype=int)
            self.global_input_size = 0

            A_complete_path = os.path.join(data_location, 'A_complete.csv')
            C_complete_path = os.path.join(data_location, 'C_complete.csv')
            B_complete_path = os.path.join(data_location, 'B_complete.csv')
            Q_complete_path = os.path.join(data_location, 'Q_complete.csv')
            R_complete_path = os.path.join(data_location, 'R_complete.csv')
            x0_complete_path = os.path.join(data_location, 'x0_complete.csv')
            U_complete_path = os.path.join(data_location, 'U_complete.csv')


            df_A_complete = pd.read_csv(A_complete_path, header=None)
            df_C_complete = pd.read_csv(C_complete_path, header=None)
            df_B_complete = pd.read_csv(B_complete_path, header=None)
            df_Q_complete = pd.read_csv(Q_complete_path, header=None)
            df_R_complete = pd.read_csv(R_complete_path, header=None)
            df_x0_complete = pd.read_csv(x0_complete_path, header=None)
            df_U_complete = pd.read_csv(U_complete_path, header=None)


            A_complete = df_A_complete.to_numpy()
            C_complete = df_C_complete.to_numpy()
            B_complete = df_B_complete.to_numpy()
            Q_complete = df_Q_complete.to_numpy()
            R_complete = df_R_complete.to_numpy()
            x0_complete = df_x0_complete.to_numpy().T
            U_complete = df_U_complete.to_numpy().T
            
            # A_complete_valid_path = os.path.join(valid_data_location, 'A_complete.csv')  # Commented out
            # df_A_complete_valid = pd.read_csv(A_complete_valid_path, header=None)  # Commented out
            # A_complete_valid = df_A_complete_valid.to_numpy()  # Commented out

            for m in range(self.num_components):
                A_matrix_path = os.path.join(data_location, f'C{m+1}/A.csv')
                C_matrix_path = os.path.join(data_location, f'C{m+1}/C.csv')
                Y_path = os.path.join(data_location, f'C{m+1}/Y.csv')
                B_path = os.path.join(data_location, f'C{m+1}/B.csv')
                Q_path = os.path.join(data_location, f'C{m+1}/Q.csv')
                R_path = os.path.join(data_location, f'C{m+1}/R.csv')
                x0_path = os.path.join(data_location, f'C{m+1}/x0.csv')
                U_path = os.path.join(data_location, f'C{m+1}/U.csv')

                df_A = pd.read_csv(A_matrix_path, header=None)
                df_C = pd.read_csv(C_matrix_path, header=None)
                df_Y = pd.read_csv(Y_path, header=None)
                df_B = pd.read_csv(B_path, header=None)
                df_Q = pd.read_csv(Q_path, header=None)
                df_R = pd.read_csv(R_path, header=None)
                df_x0 = pd.read_csv(x0_path, header=None)
                df_U = pd.read_csv(U_path, header=None)

                A = df_A.to_numpy()
                C = df_C.to_numpy()
                Y = df_Y.to_numpy().T
                B = df_B.to_numpy()
                Q = df_Q.to_numpy()
                R = df_R.to_numpy()
                x0 = df_x0.to_numpy().T
                U = df_U.to_numpy().T
                P0 = Q

                if Y.shape[1] != self.total_time: 
                    raise ValueError(f"Length of time series from config file = {self.total_time} != Length of observation series {Y.shape[1]}")
                
                Y_train = Y[:, :self.training_time]
                U_train = U[:, :self.training_time]

                # Y_valid_path = os.path.join(valid_data_location, f'C{m+1}/Y.csv')  # Commented out
                # df_Y_valid = pd.read_csv(Y_valid_path, header=None)  # Commented out
                # Y_valid = df_Y_valid.to_numpy().T  # Commented out
                
                # if Y_valid.shape[1] != self.validation_time:  # Commented out
                #     raise ValueError(f"Length of time series from config file = {self.validation_time} != Length of observation series {Y_valid.shape[1]}")  # Commented out

                d_m, p_m = C.shape

                self.comp_size_vec[m, 0] = p_m
                self.global_state_size += p_m

                # Store output and input sizes
                self.output_size_vec[m, 0] = d_m
                self.global_output_size += d_m

                s_m = B.shape[1]
                self.input_size_vec[m, 0] = s_m
                self.global_input_size += s_m

                self.local_learners_pack[f'comp_{m+1}'] = {
                    'A': A,
                    'B': B,
                    'C': C,
                    'Q': Q,
                    'R': R,
                    'P0': P0,
                    'x0': x0,
                    'Y': Y_train,
                    'U': U_train,
                    'Y_full': Y,
                    'U_full': U,
                    'd_m': d_m,
                    'p_m': p_m,
                    's_m': s_m,
                    'lambda_l': lambda_l[m],
                    'eta_1': eta_1[m],
                    'eta_2': eta_2[m],
                    'gamma_1': gamma_1[m],
                    'gamma_2': gamma_2[m],
                    'phi0': np.zeros((p_m, 1)),
                }

                x_dkf_pred_csv = os.path.join(data_location, f'C{m+1}/X_dkf_pred.csv')
                x_dkf_est_csv = os.path.join(data_location, f'C{m+1}/X_dkf_est.csv')

                has_pred_csv = os.path.exists(x_dkf_pred_csv)
                has_est_csv = os.path.exists(x_dkf_est_csv)
                if has_pred_csv != has_est_csv:
                    raise FileNotFoundError(
                        f"Expected both DKF files or none for component {m+1}: "
                        f"{x_dkf_pred_csv}, {x_dkf_est_csv}"
                    )

                if has_pred_csv and has_est_csv:
                    df_x_dkf_pred = pd.read_csv(x_dkf_pred_csv, header=None)
                    df_x_dkf_est = pd.read_csv(x_dkf_est_csv, header=None)
                    x_dkf_pred_full = df_x_dkf_pred.to_numpy().T
                    x_dkf_full = df_x_dkf_est.to_numpy().T

                    x_dkf = x_dkf_full[:, :self.training_time]
                    x_dkf_pred = x_dkf_pred_full[:, :self.training_time]
                    x_dkf_resd = _residual_norm_from_pred(C, x_dkf_pred, Y_train)
                else:
                    x_dkf, x_dkf_pred, x_dkf_resd = _run_dkf_baseline(
                        A, B, C, Q, R, P0, x0, Y_train, U_train
                    )

                pack = self.local_learners_pack[f'comp_{m+1}']
                pack['Hc_tm1'] = x_dkf
                pack['X_dkf'] = x_dkf
                pack['X_dkf_pred'] = x_dkf_pred
                pack['X_dkf_resd'] = x_dkf_resd
                if has_pred_csv and has_est_csv:
                    pack['X_dkf_full'] = x_dkf_full
                    pack['X_dkf_pred_full'] = x_dkf_pred_full

                # self.validation_pack[f'comp_{m+1}'] = {  # Commented out
                #     'B': B,
                #     'Q': Q,
                #     'R': R,
                #     'P0': P0,
                #     'x0': x0,
                #     'A': A,
                #     'C': C,
                #     'Y': Y_valid,
                #     'd_m': d_m,
                #     'p_m': p_m,
                #     'eta_l': eta_l[m],
                #     'eta_g': eta_g[m],
                #     'lambda_l': lambda_l[m],
                # }
                self.global_Y[f'{m+1}'] = Y
                self.global_U[f'{m+1}'] = U
                # self.global_Y_valid[f'{m+1}'] = Y_valid  # Commented out
                A_mm[f'{m+1}{m+1}'] = A
                B_mm[f'{m+1}{m+1}'] = B

            global_Y = np.zeros((self.global_output_size, self.total_time))
            global_U = np.zeros((self.global_input_size, self.total_time))
            # global_Y_valid = np.zeros((self.global_output_size, self.total_time))  # Commented out
            d_start_idx = 0
            u_start_idx = 0
            x_start_idx = 0
            for m in range(self.num_components):
                global_Y[d_start_idx:d_start_idx + self.output_size_vec[m, 0], :] = self.global_Y[f'{m+1}']
                global_U[u_start_idx:u_start_idx + self.input_size_vec[m, 0], :] = self.global_U[f'{m+1}']
                # global_Y_valid[d_start_idx:d_start_idx + self.output_size_vec[m, 0], :] = self.global_Y_valid[f'{m+1}']  # Commented out
                if m > 0:
                    self.output_start_vec[m, 0] = d_start_idx
                    self.input_start_vec[m, 0] = u_start_idx
                    self.comp_start_vec[m, 0] = x_start_idx

                d_start_idx = d_start_idx + self.output_size_vec[m, 0]
                u_start_idx = u_start_idx + self.input_size_vec[m, 0]
                x_start_idx = x_start_idx + self.comp_size_vec[m, 0]

            A_mn_init = {}
            for i in range(self.num_components):
                p_i = self.comp_size_vec[i, 0]
                for j in range(self.num_components):
                    if i == j:
                        continue
                    p_j = self.comp_size_vec[j, 0]
                    row_start = sum(self.comp_size_vec[k, 0] for k in range(i))
                    col_start = sum(self.comp_size_vec[k, 0] for k in range(j))
                    A_mn_init[f'{i+1}{j+1}'] = A_complete[
                        row_start:row_start + p_i,
                        col_start:col_start + p_j,
                    ]

            B_mn_init = {}
            for i in range(self.num_components):
                p_i = self.comp_size_vec[i, 0]
                for j in range(self.num_components):
                    if i == j:
                        continue
                    s_j = self.input_size_vec[j, 0]
                    row_start = sum(self.comp_size_vec[k, 0] for k in range(i))
                    col_start = sum(self.input_size_vec[k, 0] for k in range(j))
                    B_mn_init[f'{i+1}{j+1}'] = B_complete[
                        row_start:row_start + p_i,
                        col_start:col_start + s_j,
                    ]

            self.global_learners_pack['A_mm'] = A_mm
            self.global_learners_pack['B_mm'] = B_mm
            # self.global_learners_pack['A_init_offdiag'] = A_mn_init
            # self.global_learners_pack['B_init_offdiag'] = B_mn_init
            self.global_learners_pack['alpha_a'] = self.alpha_a
            self.global_learners_pack['alpha_b'] = self.alpha_b
            self.global_learners_pack['lambda_g'] = self.global_regularization_rate
            self.global_learners_pack['xi'] = self.xi
            self.global_learners_pack['p_vec'] = self.comp_size_vec
            self.global_learners_pack['s_vec'] = self.input_size_vec

            self.HMatrix_pack['comp_data'] = self.local_learners_pack
            self.HMatrix_pack['lambda_g'] = self.global_regularization_rate
            self.HMatrix_pack['alpha_a'] = self.alpha_a
            self.HMatrix_pack['alpha_b'] = self.alpha_b
            self.HMatrix_pack['xi'] = self.xi
            self.HMatrix_pack['p'] = self.global_state_size
            self.HMatrix_pack['p_vec'] = self.comp_size_vec
            self.HMatrix_pack['d'] = self.global_output_size
            self.HMatrix_pack['d_vec'] = self.output_size_vec     
            self.HMatrix_pack['M'] = self.num_components
            self.HMatrix_pack['total_time'] = self.total_time

            self.CKF_pack['A_complete'] = A_complete
            self.CKF_pack['C_complete'] = C_complete
            self.CKF_pack['Y'] = global_Y
            self.CKF_pack['B_complete'] = B_complete
            self.CKF_pack['Q'] = Q_complete
            self.CKF_pack['R'] = R_complete
            self.CKF_pack['P0'] = Q_complete
            self.CKF_pack['x0'] = x0_complete
            self.CKF_pack['U'] = global_U
            self.CKF_pack['U_complete'] = U_complete
            self.CKF_pack['p_vec'] = self.comp_size_vec
            self.CKF_pack['s_vec'] = self.input_size_vec
            self.CKF_pack['output_start_vec'] = self.output_start_vec
            self.CKF_pack['input_start_vec'] = self.input_start_vec
            self.CKF_pack['comp_start_vec'] = self.comp_start_vec
            # self.CKF_pack['Y_valid'] = global_Y_valid  # Commented out
            # self.CKF_pack['A_complete_valid'] = A_complete_valid  # Commented out

        except FileNotFoundError as e:
            print(f"An error occurred: {e}")

        except ValueError as e:
            print(f"An error occurred: {e}")
