import cv2
import numpy as np
from matplotlib import pyplot as plt
from calibration import *

  

def get_depth(img_left, img_right, px,py):
    # convert images to grayscale for template matching
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    min_disp = 15
    num_disp = 10 * 16
    block_size = 61
    stereo = cv2.StereoBM_create(numDisparities = num_disp, blockSize = block_size)
    stereo.setMinDisparity(min_disp)
    stereo.setDisp12MaxDiff(200)
    stereo.setUniquenessRatio(0)
    stereo.setSpeckleRange(20)
    stereo.setSpeckleWindowSize(1)

    disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    disp = cv2.bilateralFilter(disp,9,20,75)
    depth = 653 * 120 / (disp[py,px] * 1000)
    return depth, disp

def get_depth_wls(img_left, img_right, px,py):

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    lmbda = 8000.0
    sigma = 3
    min_disp = 15
    num_disp = 10 * 16
    block_size = 61
    left_matcher = cv2.StereoBM_create(numDisparities = num_disp, blockSize = block_size);
    left_matcher.setMinDisparity(min_disp)
    left_matcher.setDisp12MaxDiff(200)
    left_matcher.setUniquenessRatio(0)
    left_matcher.setSpeckleRange(20)
    left_matcher.setSpeckleWindowSize(1)

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher);
    left_disp = left_matcher.compute(gray_left, gray_right);
    right_disp = right_matcher.compute(gray_right, gray_left);

    # Now create DisparityWLSFilter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher);
    wls_filter.setLambda(lmbda);
    wls_filter.setSigmaColor(sigma);
    filtered_disp = wls_filter.filter(left_disp, gray_left, disparity_map_right=right_disp);
    depth = 653 * 120 /(filtered_disp[py,px]*1000)

    return depth, filtered_disp 


    

if __name__ == "__main__":
    calib = Calibration(None,None,None)
    calib.load("../data/Calibration_result.bin")

    left_imgs = glob.glob("../data/conveyorImages/left/*")
    right_imgs = glob.glob("../data/conveyorImages/right/*")
    assert left_imgs
    assert right_imgs
    
    fig = plt.figure() 
    ax = fig.add_subplot(1,1,1)
    for i in range(len(left_imgs)):
        left_img = cv2.imread(left_imgs[i])
        left_img = calib.left_remap(left_img)
        right_img = cv2.imread(right_imgs[i])
        right_img = calib.right_remap(right_img)
    
        depth, disp = get_depth_wls(left_img,right_img, 100,100)
   
        ax.clear(); ax.imshow(disp);
        plt.pause(0.05)
    plt.show()



