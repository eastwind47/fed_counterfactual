"""
Contains routines to carry out required computations at each local model
"""

# Importing the necessary libraries
import numpy as np
import kalman_filter

class LocalModel:
    def __init__(self, T, component_data, theta = None):
        
        # Total length of time series
        self.T              = T

        # Extracting the data received
        self.A              = component_data['A']
        self.C              = component_data['C']
        self.p_m            = component_data['p_m']
        self.eta_l          = component_data['eta_l']
        self.eta_g          = component_data['eta_g']
        self.Y              = component_data['Y']
        self.d_m            = component_data['d_m']
        self.lambda_l       = component_data['lambda_l']
        self.X_dkf          = np.zeros((self.p_m, self.T))
        self.X_dkf_resd     = np.zeros((1, self.T))
        self.X_dkf_pred     = np.zeros((self.p_m, self.T))

        self.x0             = component_data['x0']
        self.B              = component_data['B']
        self.Q              = component_data['Q']
        self.R              = component_data['R']
        self.P0             = component_data['P0']

        self.x0_vfl         = self.x0
        self.y0             = self.C @ self.x0_vfl

        # Initializing the augmented model
        self.X_vfl          = np.zeros((self.p_m, self.T))
        self.X_vfl_pred     = np.zeros((self.p_m, self.T))

        # Initializing the theta value
        self.theta          = theta 

    def run_DKF(self):
        DKF            = kalman_filter.KalmanFilter(self.A, self.B, self.C, self.Q, self.R, self.P0, self.x0)

        for t in range(self.T):
            DKF.predict(np.zeros((self.p_m, 1)))
            self.X_dkf_pred[:,t:t+1]    = DKF.get_state()
            self.X_dkf_resd[0,t:t+1]    = np.linalg.norm(DKF.residual(self.Y[:,t:t+1]))
            # print(self.X_dkf_resd[0,t:t+1])
            # self.X_dkf_resd[0,t:t+1]    = np.linalg.norm(DKF.residual(self.Y[:,t:t+1]))**2
            DKF.update(self.Y[:,t:t+1])
            self.X_dkf[:,t:t+1]         = DKF.get_state()

    # Residual from the prediction of augmented local model
    
    def VFL_prediction(self, t):
        if t == 0:
            self.X_vfl_pred[:,t:t+1]        = self.A @ self.x0
        else:
            self.X_vfl_pred[:,t:t+1]        = self.A @ self.X_vfl[:,t-1:t]
        return self.X_vfl_pred[:,t:t+1]
    
    def VFL_residual(self, t):
        # return np.linalg.norm(self.Y[:,t:t+1] - self.C @ self.X_vfl_pred[:,t:t+1]) / np.linalg.norm(self.Y[:,t:t+1])
        #  return np.linalg.norm(self.Y[:,t:t+1] - self.C @ self.X_vfl_pred[:,t:t+1])
        return self.Y[:,t:t+1] - self.C @ self.X_vfl_pred[:,t:t+1]
    
    # def DKF_residual(self, t):
    #     return np.linalg.norm(self.Y[:,t:t+1] - self.C @ self.X_dkf_pred[:,t:t+1]) / np.linalg.norm(self.Y[:,t:t+1])
    
    def VFL_local_loss(self, t):
        r_t         = self.VFL_residual(t)
        return np.linalg.norm(r_t)**2 + self.lambda_l * np.linalg.norm(self.theta, 'fro')**2
        # return np.linalg.norm(r_t)**2
    
    def VFL_estimate(self, t):
        self.X_vfl[:,t:t+1]         = self.X_dkf[:,t:t+1] + self.theta @ self.Y[:,t:t+1]
        return self.X_vfl[:,t:t+1]


    def Grad_theta_L(self, t):
        y_t         = self.Y[:,t:t+1]
        if t == 0:
            y_t_1       = self.y0
            x_t_1       = self.x0
        else:
            y_t_1       = self.Y[:,t-1:t]
            x_t_1       = self.X_dkf[:,t-1:t]         
            
        grad_theta_L    = -2 * self.A.T @ self.C.T @ y_t @ y_t_1.T \
                + 2 * self.A.T @ self.C.T @ self.C @ self.A @ (x_t_1 @ y_t_1.T + self.theta @ y_t_1 @ y_t_1.T) + 2 * self.lambda_l * self.theta
            
        return grad_theta_L

    
    def GradDescent(self, grad_x, t):
            """
            t == 0 issue has been partially resolved. But make sure grad_x obtained from global model is reasonable.
            """
            if t == 0:
                y_t_1           = self.y0
            else:
                y_t_1           = self.Y[:,t-1:t]
            # if 28000 <= t < 28010:
            #     print ('t = ', t)
            #     print('global influence = ', self.eta_g * grad_x @ y_t_1.T)
            #     print('local influence = ', self.eta_l * self.Grad_theta_L(t))
            #     print('local gradient = ', self.Grad_theta_L(t))
            #     print('theta = ', self.theta)
            self.theta          = self.theta - self.eta_g * grad_x @ y_t_1.T - self.eta_l * self.Grad_theta_L(t)
        