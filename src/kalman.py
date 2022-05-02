import numpy as np
from matplotlib import pyplot as plt

class Kalman:
    def __init__(self, measurements, v, x0, P0):
        self.measurements = measurements
        self.x = x0
        self.P = P0
        self.v = v # velocity vector measured in distance pr timestep
        self.Q = None
        self.R = None
        self.x = None
        self.kalmanEstimates = self.positionUpdate()
    
    def kalmanPosition(self, measurement, x_prediction, P_prediction):
        newData = 1
        """
        Kalman filter for position
        input:
            measurement: measured x,y,z
            x_prediction: x,y,z prediction from previous iteration
            P_prediction: covariance matrix from previous iteration
            Q: process noise covariance matrix
            R: measurement noise covariance matrix
            k: iteration number
        output:
            x_prediction: x,y,z prediction from current iteration
            P_prediction: covariance matrix from current iteration
        """
        
        A = np.eye(3)
        B = self.v 
        print(A @ x_prediction)

        # Prediction
        x_prediction = A @ x_prediction + B
        P_prediction = A @ P_prediction @ A.T + self.Q


        # Update
        if measurement[0] != 0:
            H = np.eye(3)
            K = P_prediction @ H.T @ np.linalg.inv(H @ P_prediction @ H.T + self.R)
            x_prediction = x_prediction + K @ (measurement - H @ x_prediction)
            P_prediction = (np.eye(3) - K @ H) @ P_prediction
        
        ############### USE THIS WHEN ACTUAL LISTS ARE INPUT ###############
        # if newData:
        #     H = np.eye(3)
        #     K = P_prediction @ H.T @ np.linalg.inv(H @ P_prediction @ H.T + R)
        #     x_prediction = x_prediction + K @ (measurement - H @ x_prediction)
        #     P_prediction = (np.eye(3) - K @ H) @ P_prediction


        return x_prediction, P_prediction

    def positionUpdate(self):
        print(self.x)
        print(self.measurements.T[i])
        kalman_estimates = np.zeros_like(self.measurements)
        kalman_estimates[0] = self.x
        newData = 1
        for i in range(len(self.measurements)):

            self.x, self.P = self.kalmanPosition(self.measurements.T[i], self.x, self.P)
            kalman_estimates[i+1] = self.x
        return kalman_estimates
    








    
