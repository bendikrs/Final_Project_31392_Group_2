import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt



class Calibration:
    def __init__(self, img_path, nb_vertical, nb_horizontal):
        # setup for the calibration images
        self.PATH = img_path
        self.pattern_size = (nb_vertical, nb_horizontal)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.

    def calibrate(self):
        # Load the images
        images = glob.glob(self.PATH)
        assert images
        h, w = images[0].shape[:2]
        
        # perform calibration for left images only
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            #Implement findChessboardCorners here
            ret, corners = cv2.findChessboardCorners(img, self.pattern_size)

            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    
    def distort_img(self, img, Crop = False):
        pass


# some more functions




if __name == "__main__":
    # test code
