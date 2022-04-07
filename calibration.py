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
        self.cameramatrixLeft = [] # camera matrix for left camera
        self.cameramatrixRight = [] # camera matrix for right camera

    def calibrate(self):
        # Load the images
        images = glob.glob(self.PATH)
        assert images
        
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
        
        h, w = img.shape[:2]
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),1,(w,h))
    
    def distort_img(self, img, Crop = False):

        #get undistortet image
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)

        #crop if needed
        if Crop == True:
            x,y,w,h = self.roi
            dst = dst[y:y+h, x:x+w]
        return dst
# some more functions


if __name__ == "__main__":
    # test code

    calitest = Calibration("data/calibration_images/*.png", 9, 6)

    calitest.calibrate()

    dst = calitest.distort_img(cv2.imread("data/calibration_images/left-0001.png"), True)
    cv2.imshow("undistorted", dst)
