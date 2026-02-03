import numpy as np
import pandas as pd
import scipy.linalg
import matplotlib.pyplot as plt

# Importing csv file
def load_csv_as_matrix(file_path):
    # Read the CSV file using pandas
    df = pd.read_csv(file_path, header = None)
    
    # Convert the DataFrame to a NumPy matrix
    matrix = df.to_numpy()
    
    return matrix


class KalmanFilter:
    def __init__(self, A, B, C, Q, R, P, x):
        self.A = A  # State transition matrix
        self.B = B  # Control matrix
        self.C = C  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Estimate error covariance
        self.x = x  # State estimate

    def predict(self, u=0):
        # Predict the state and covariance
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def residual(self, y):
        return y - self.C @ self.x

    def update(self, z):
        # Update the state with a new measurement
        # y = z - self.C @ self.x
        y = self.residual(z)
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.C @ self.P

    def get_state(self):
        return self.x
    

    
