import numpy as np
from matplotlib import pyplot as plt

class kalman:
    def __init__(self, A, B, C, Q, R, x0, P0):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.x_hat = x0
        self.P_hat = P0
        self.K = np.zeros((2,1))
        self.x_hat_hist = []
        self.P_hat_hist = []
        self.x_hist = []
        self.P_hist = []
        self.z_hist = []

    def update(self, z):
        self.x_hat = self.A @ self.x + self.B @ z
        self.P_hat = self.A @ self.P @ self.A.T + self.Q
        self.K = self.P_hat @ self.C.T @ np.linalg.inv(self.C @ self.P_hat @ self.C.T + self.R)
        self.x = self.x_hat + self.K @ (z - self.C @ self.x_hat)
        self.P = (np.identity(2) - self.K @ self.C) @ self.P_hat
        self.x_hat_hist.append(self.x_hat)
        self.P_hat_hist.append(self.P_hat)
        self.x_hist.append(self.x)
        self.P_hist.append(self.P)
        self.z_hist.append(z)

    def plot(self):
        plt.plot(self.x_hat_hist)
        plt.plot(self.P_hat_hist)
        plt.plot(self.x_hist)
        plt.plot(self.P_hist)
        plt.plot(self.z_hist)
        plt.show()