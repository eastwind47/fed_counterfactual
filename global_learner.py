# Global model

# Importing required libraries
import numpy as np

class GlobalModel:
    
    def __init__(self, M, T, global_learner_data, A_mn):

        # Number of components
        self.M                  = M

        # Length of time series
        self.T                  = T
        
        # Global learning rate
        self.gamma_g            = global_learner_data['gamma_g']

        # Global regularization rate
        self.lambda_g           = global_learner_data['lambda_g']

        # Set of diagonal matrices
        self.A_mm               = global_learner_data['A_mm']


        # Initializing the off-diagonal matrices
        self.A_mn               = A_mn

        # Initializing the vector form of off-diagonal matrices
        self.A_mn_vec           = {}
        # Initialize the vectorized form of off-diagonal matrices
        self.A_mn_vec = {}
        for m in range(self.M):
            for n in range(self.M):
                if m != n:
                    # Vectorize the off-diagonal matrix A_mn[f'{m+1}{n+1}']
                    self.A_mn_vec[f'{m+1}{n+1}'] = self.A_mn[f'{m+1}{n+1}'].flatten()


    # Function for global loss
    def GlobalLoss(self, x_dkf, x_vfl, t):

        """
        Note that it has not taken care of the initial conditions
        """

        if t == 0:
            return 5
        else:
            global_loss         = 0

            for m in range(self.M):
                gt_m            = self.A_mm[f'{m+1}{m+1}'] @ (x_vfl[f'{m+1}'] - x_dkf[f'{m+1}'])
                sum_term        = np.zeros(gt_m.shape)
                frob_norm_term  = 0
                for j in range(self.M):
                    if j == m:
                        continue
                    else:
                        sum_term += self.A_mn[f'{m+1}{j+1}'] @ x_dkf[f'{j+1}']
                        frob_norm_term += np.linalg.norm(self.A_mn[f'{m+1}{j+1}'], 'fro')**2

                gt_m            = gt_m - sum_term

                global_loss     += np.linalg.norm(gt_m)**2 
                
            global_loss += self.lambda_g * frob_norm_term

        return global_loss
    
    def GlobalLoss_valid(self, x_dkf, x_vfl, t):

        """
        Note that it has not taken care of the initial conditions
        """

        if t == 0:
            return 5
        else:
            global_loss         = 0

            for m in range(self.M):
                gt_m            = self.A_mm[f'{m+1}{m+1}'] @ (x_vfl[f'{m+1}'] - x_dkf[f'{m+1}'])
                sum_term        = np.zeros(gt_m.shape)
                frob_norm_term  = 0
                for j in range(self.M):
                    if j == m:
                        continue
                    else:
                        sum_term += self.A_mn[f'{m+1}{j+1}'] @ x_dkf[f'{m+1}']
                        frob_norm_term += np.linalg.norm(self.A_mn[f'{m+1}{j+1}'], 'fro')**2

                gt_m            = gt_m - sum_term

                global_loss     += np.linalg.norm(gt_m)**2 
                
            global_loss += self.lambda_g * frob_norm_term 

        return global_loss



    def GradA(self, x_dkf, x_vfl, t):
        gradA_mn        = {}
        """
        We are taking the value at t=0 based on our assumptions about initial data. Check again if it
        needs to be changed. Explicitly substituting the initial value of x
        """
    
        for m in range(self.M):
            for n in range(self.M):
                if m == n:
                    continue
                else:
                    delta_x     = x_vfl[f'{m+1}'] - x_dkf[f'{m+1}']

                    sum_term    = np.zeros(self.A_mn[f'{m+1}{n+1}'].shape)

                    for l in range(self.M):
                        if l == m:
                            continue
                        else:
                            sum_term += self.A_mn[f'{m+1}{l+1}'] @ x_dkf[f'{l+1}'] @ x_dkf[f'{n+1}'].T

                    gradA_mn[f'{m+1}{n+1}']     = -2 * self.A_mm[f'{m+1}{m+1}'] @ delta_x @ x_dkf[f'{n+1}'].T\
                        + 2 * sum_term + 2 * self.lambda_g * self.A_mn[f'{m+1}{n+1}']

        return gradA_mn
    
    def Gradx(self, x_dkf, x_vfl, t):
        gradx           = {}
        """
        We are taking the value at t=0 based on our assumptions about initial data. Check again if it
        needs to be changed. Explicitly substituting the initial value of x
        """
        for m in range(self.M):
            delta_x     = x_vfl[f'{m+1}'] - x_dkf[f'{m+1}']
            sum_term        = np.zeros(x_dkf[f'{m+1}'].shape)

            for j in range(self.M):
                if j == m:
                    continue
                else:
                    sum_term += self.A_mn[f'{m+1}{j+1}'] @ x_dkf[f'{j+1}']

            gradx[f'{m+1}']     = 2 * self.A_mm[f'{m+1}{m+1}'].T @ self.A_mm[f'{m+1}{m+1}'] @ delta_x \
                - 2 * self.A_mm[f'{m+1}{m+1}'].T @ sum_term

        return gradx


    def GradDescent(self, x_dkf, x_vfl, t):
        
        gradA_mn        = self.GradA(x_dkf, x_vfl, t)
        # print(gradA_mn)
        # print('gradA_mn = ', gradA_mn)
        # print('gamma_g * gradA_mn = ', self.gamma_g * gradA_mn)
        for m in range(self.M):
            for n in range(self.M):
                if m == n:
                    continue
                else:
                    self.A_mn[f'{m+1}{n+1}'] = self.A_mn[f'{m+1}{n+1}'] - self.gamma_g * gradA_mn[f'{m+1}{n+1}']
                    self.A_mn_vec[f'{m+1}{n+1}'] = self.A_mn[f'{m+1}{n+1}'].flatten()

        return self.A_mn
    
    def A_mn(self):
        return self.A_mn

    def GlobalLoss_comp(self, comp, x_dkf, x_vfl, t):

        if t == 0:
            return 5
        else:
            global_loss         = 0

            gt_m            = self.A_mm[f'{comp+1}{comp+1}'] @ (x_vfl[f'{comp+1}'] - x_dkf[f'{comp+1}'])
            sum_term        = np.zeros(gt_m.shape)
            frob_norm_term  = 0
            for j in range(self.M):
                if j == comp:
                    continue
                else:
                    sum_term += self.A_mn[f'{comp+1}{j+1}'] @ x_dkf[f'{j+1}']
                    frob_norm_term += np.linalg.norm(self.A_mn[f'{comp+1}{j+1}'], 'fro')**2

            gt_m            = gt_m - sum_term

            global_loss     += np.linalg.norm(gt_m)**2 + self.lambda_g * frob_norm_term


        return global_loss
