import numpy as np
from matplotlib import pyplot as plt

class Kalman:
    def __init__(self, measurements, v, x0, P0, R):
        """
        Kalman filter for position
        input:
            measurements: measured x,y,z in a 3xn array
            v: velocity vector
            x0: initial x,y,z position (measurements.T[0] f.ex)
            P0: initial covariance matrix
            R: measurement noise covariance matrix
        """

        self.measurements = measurements
        self.x = x0
        self.P = P0
        self.v = v # velocity vector measured in distance pr timestep
        self.Q = np.eye(3)
        self.R = R

        # self.kalmanEstimates = self.positionUpdate()
    
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
        kalman_estimates = np.zeros_like(self.measurements).T
        kalman_estimates[0] = self.x
        newData = 1
        for i in range(len(self.measurements[0])-1):
            self.x, self.P = self.kalmanPosition(self.measurements.T[i], self.x, self.P)
            kalman_estimates[i+1] = self.x
        return kalman_estimates

    


def get_kalman_data(results):
    """
    gets data in bendik format
    returns kalman estimated positions 3 x n
    """
    measurements = np.zeros((len(results), 3))
    v = (np.array([results[-50][0][7]]) - np.array([results[0][0][7]]))[0] / 130
    # print(results)
    for i in range(0,len(results)):
        if len(results [i]):

            if len(np.array(results[i][0][7])):
                x = np.array(results[i][0][7])
            else:
                x = np.array([[0,0,0]])
            measurements[i] = np.array(x)
            
    r_xy = 1
    r_z = 1
    x_init = measurements[0]

    P_init = np.eye(3)
    Q_init = np.eye(3)
    R_init = np.array([[r_xy,0,0],[0,r_xy,0],[0,0,r_z]])

    kalman = Kalman(measurements.T, v, x_init, P_init, R_init)

    kalman_estimates = kalman.positionUpdate()

    return kalman_estimates.T





    
