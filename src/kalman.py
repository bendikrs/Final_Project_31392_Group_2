import numpy as np
from matplotlib import pyplot as plt

class Kalman:
    def __init__(self, measurements,x0,P0):
        self.measurements = measurements
        self.x = x0.T
        self.P = P0
        self.v = None # velocity vector measured in distance pr timestep
        self.Q = np.eye(3)
        self.R = None
        self.x = None

    
    def kalmanFilter(self,measurement, newData,x_prediction, P_prediction):
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

        # Prediction
        x_prediction = A @ x_prediction + B
        P_prediction = A @ P_prediction @ A.T + self.Q

        if newData: 
            # Update
            H = np.eye(3)
            K = P_prediction @ H.T @ np.linalg.inv(H @ P_prediction @ H.T + self.R)
            x_prediction = x_prediction + K @ (measurement - H @ x_prediction)
            P_prediction = (np.eye(3) - K @ H) @ P_prediction


        return x_prediction, P_prediction

    def positionUpdate(self):
        estimates = []
        newData = 1
        for i in self.measurements.T:
            if i == np.zeros(3):
                newData = 0
            newData = 1
            self.x, self.P = self.kalmanFilter(i,newData,self.x,self.P)
            estimates.append(self.x)
        return estimates


    
