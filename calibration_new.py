import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt



class Calibration:
    def __init__(self, img_path, nb_vertical, nb_horizontal):
        # setup for the calibration images
        self.PATH = img_path
        self.pattern_size = (nb_vertical, nb_horizontal)
        
        # for distorsion
        self.mtx = None
        self.dist = None
        self.roi = None
        self.newcameramtx = None

    def calibrate(self):
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.pattern_size[0]*self.pattern_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:self.pattern_size[0],0:self.pattern_size[1]].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        
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
                objpoints.append(objp)
                imgpoints.append(corners)
        
        h, w = img.shape[:2]
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
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

    calitest = Calibration("data/calibration_images/*.png", 6, 9)
    print("1")
    calitest.calibrate()
    print("2")
    cv2.imshow("distorted", cv2.imread("data/calibration_images/left-0001.png")
    dst = calitest.distort_img(cv2.imread("data/calibration_images/left-0001.png"), True)
    print("3")
    cv2.imshow("undistorted", dst)
