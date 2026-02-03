

"""
Class that computes the state transition matrix for the linear system of the off-diagonal terms and the
related functions
"""
# Importing required libraries
import numpy as np
import scipy.linalg

class SpectralRadius:
    def __init__(self, hmatrix_package, dkf_data):
        self.component_data     = hmatrix_package['comp_data']
        self.gamma_g            = hmatrix_package['gamma_g']
        self.lambda_g           = hmatrix_package['lambda_g']
        self.X_dkf              = dkf_data
        self.p                  = hmatrix_package['p']
        self.p_vec              = hmatrix_package['p_vec']
        self.d                  = hmatrix_package['d']
        self.d_vec              = hmatrix_package['d_vec']
        self.M                  = hmatrix_package['M']
        self.T                  = hmatrix_package['total_time']


    # P_nn function
    def P_nn(self, m, n, t):

        if t == 0:
            x_n_t_1         = self.component_data[f'comp_{n}']['x0']
        else:
            x_n_t_1         = self.X_dkf[f'{n}'][:,t-1:t]

        P1              = (1 - 2 * self.gamma_g * self.lambda_g) * np.eye(self.component_data[f'comp_{n}']['p_m'])\
              - 2 * self.gamma_g * x_n_t_1 @ x_n_t_1.T
        
        # if t == 6000:
        #     print('2nd half of P = ', x_n_t_1 @ x_n_t_1.T)
        #     print('eigenvalues of P = ', scipy.linalg.eigvals(P1))
        #     print('shape of P = ', P1.shape)
        
        return np.kron(P1, np.eye(self.component_data[f'comp_{m}']['p_m']))
    
    # Q_mn function
    def Q_mn(self, m, n, t):
        
        if t == 0:
            x_n_t_1         = self.component_data[f'comp_{n}']['x0']
            y_m_t_1         = self.component_data[f'comp_{m}']['C'] @ self.component_data[f'comp_{m}']['x0']
        else:
            x_n_t_1         = self.X_dkf[f'{n}'][:,t-1:t]
            y_m_t_1         = self.component_data[f'comp_{m}']['Y'][:,t-1:t]
        
        return y_m_t_1 @ x_n_t_1.T


    def V_nl(self, n, l, t):

        if t == 0:
            x_n_t_1         = self.component_data[f'comp_{n}']['x0']
            x_l_t_1         = self.component_data[f'comp_{l}']['x0']

        else:
            x_n_t_1         = self.X_dkf[f'{n}'][:,t-1:t]
            x_l_t_1         = self.X_dkf[f'{l}'][:,t-1:t]

        return x_n_t_1 @ x_l_t_1.T
    
    def N(self, m, t):

        if t == 0:
            y_m_t_1         = self.component_data[f'comp_{m}']['C'] @ self.component_data[f'comp_{m}']['x0']
        else:
            y_m_t_1         = self.component_data[f'comp_{m}']['Y'][:,t-1:t]

        return y_m_t_1 @ y_m_t_1.T

    def HMatrix(self, m, t):
        # print('m = ', m)
        """
        There is no zeroth component, m can only start from 1
        """
        hmatrix         = np.zeros(((self.p - self.p_vec[m-1, 0] + self.d_vec[m-1, 0]) * self.p_vec[m-1, 0], (self.p - self.p_vec[m-1, 0] + self.d_vec[m-1, 0]) * self.p_vec[m-1, 0]))
        # print(hmatrix.shape)
        row_end_idx     = 0
        # col_end_idx     = 0

        # size of component and size of outputs
        p_m             = self.p_vec[m-1, 0]
        d_m             = self.d_vec[m-1, 0]

        # Extracting the local parameter values
        eta_g           = self.component_data[f'comp_{m}']['eta_g']
        eta_l           = self.component_data[f'comp_{m}']['eta_l']
        lambda_l        = self.component_data[f'comp_{m}']['lambda_l']

        # check
        R               = 2 * self.component_data[f'comp_{m}']['A']
        M1              = 2 * self.component_data[f'comp_{m}']['A'].T @ self.component_data[f'comp_{m}']['A']
        M2              = 2 * self.component_data[f'comp_{m}']['A'].T @ self.component_data[f'comp_{m}']['C'].T @ self.component_data[f'comp_{m}']['C'] @ self.component_data[f'comp_{m}']['A']
        M               = eta_g * M1 + eta_l * M2

        for i in range(self.M):
            col_start_idx           = row_end_idx
            if i == m - 1:
                continue
            else:
                row_start_idx       = row_end_idx
                row_end_idx         = row_start_idx + self.p_vec[i, 0] * p_m
            
            for j in range(i, self.M):
                # maybe you can skip the loop because you start from i and i already takes care of it
                # if j == m - 1:
                #     continue
                if j == m - 1:
                    continue
                elif i == j:
                    
                    col_end_idx     = col_start_idx + self.p_vec[j, 0] * p_m 
                    # print('(i,j) = ',(i+1,j+1))
                    # print('row_start_idx = ', row_start_idx)
                    # print('row_end_idx = ', row_end_idx)
                    # print('col_start_idx = ', col_start_idx)
                    # print('col_end_idx = ', col_end_idx)
                    if t == 4:
                    # prnt('shape of Pnn = ', self.P_nn(m, i+1, t).shape)
                        print(f'eigenvalues of P_nn({i+1}) = ', scipy.linalg.eigvals(self.P_nn(m, i+1, t)))
                        
                    hmatrix[row_start_idx:row_end_idx, col_start_idx:col_end_idx]   = self.P_nn(m, i+1, t)
                else:
                    col_start_idx   = col_end_idx
                    col_end_idx     = col_start_idx + self.p_vec[j, 0] * p_m

                    # print('(i,j) = ',(i+1,j+1))
                    # print('row_start_idx = ', row_start_idx)
                    # print('row_end_idx = ', row_end_idx)
                    # print('col_start_idx = ', col_start_idx)
                    # print('col_end_idx = ', col_end_idx)
                # if i == j: # you can also put the condition as if i == j:
                #     hmatrix[row_start_idx:row_end_idx, col_start_idx:col_end_idx]   = self.P_nn(m, i+1, t)
               
                # else:
                    hmatrix[row_start_idx:row_end_idx, col_start_idx:col_end_idx]       = - 2 * self.gamma_g * np.kron(self.V_nl(i+1, j+1, t), np.eye(p_m))

                    # if t == 6000:
                    #     print('Q kron R.T shape = ', hmatrix[row_start_idx:row_end_idx, col_start_idx:col_end_idx].shape)

                    hmatrix[col_start_idx:col_end_idx, row_start_idx:row_end_idx]       = hmatrix[row_start_idx:row_end_idx, col_start_idx:col_end_idx].T

            # double check all of this and simplify col_end_idx if correct
            col_start_idx       = col_end_idx
            col_end_idx         = col_start_idx + self.d_vec[m - 1, 0] * p_m

                # if t == 6000:
                #     print('Q.T kron R')
                #     print( np.kron(self.Q_mn(m, i+1, t).T, R))
                #     print('Q')
                #     print(self.Q_mn(m, i+1, t))
                #     print('m = ', m, 'n = ', i+1)
                # print('(i,j) = ',(i,j))
                # print('row_start_idx = ', row_start_idx)
                # print('row_end_idx = ', row_end_idx)
                # print('col_start_idx = ', col_start_idx)
                # print('col_end_idx = ', col_end_idx)
                # print('LHS shape = ', hmatrix[row_start_idx:row_end_idx, col_start_idx:col_end_idx].shape)
                # print('Q_mn_shape = ', self.Q_mn(m, i+1, t).shape)
                # print('gamma_g = ', self.gamma_g)
                # RHS     = self.gamma_g * np.kron(self.Q_mn(m, i+1, t).T, R)
                # print('RHS shape = ', RHS.shape)
            hmatrix[row_start_idx:row_end_idx, col_start_idx:col_end_idx]           = self.gamma_g * np.kron(self.Q_mn(m, i+1, t).T, R)
            hmatrix[col_start_idx:col_end_idx, row_start_idx: row_end_idx]          = eta_g * np.kron(self.Q_mn(m, i+1, t), R.T)

        
        hmatrix[col_start_idx:col_end_idx, col_start_idx:col_end_idx]                   = (1 - 2 * eta_l * lambda_l) * np.eye(p_m * d_m) - np.kron(self.N(m, t), M)

        if t == 1241:
            print(f'eigenvalues of I - N kron M ({m})  = ', scipy.linalg.eigvals(hmatrix[col_start_idx:col_end_idx, col_start_idx:col_end_idx]))
            print(f'eigenvalues of H({m}) = ', scipy.linalg.eigvals(hmatrix))
        # if t == 8200:
            # print(f'Hmatrix({m}) = ', hmatrix)
            # print('shape of H = ', hmatrix.shape)
        return hmatrix

    def SR_List(self, m):

        rho_vec         = np.zeros((self.T, 1))

        for t in range(self.T):
            H           = self.HMatrix(m, t)

            eig_values      = scipy.linalg.eigvals(H)

            # if t == 6000:
            #     print(f'eigenvalues of H({m}) = ', eig_values)

            rho_vec[t,0]    = max(abs(eig_values))

        return rho_vec
    
    def check_SR_condn(self, m):

        rho_vec         = self.SR_List(m)

        return np.all(rho_vec < 1)
    
    def check_SR_condn_all(self):
        return all(self.check_SR_condn(m) for m in range(1, self.M + 1))
    

