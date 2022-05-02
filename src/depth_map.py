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
    depth = 653 * 120 / (disp[py,px] * 1000)
    return depth, disp


    

if __name__ == "__main__":
    calib = Calibration(None,None,None)
    calib.load("../data/Calibration_result.bin")

    left_imgs = glob.glob("../../left/*")
    right_imgs = glob.glob("../../right/*")
    assert left_imgs
    assert right_imgs
    
    fig = plt.figure() 
    ax = fig.add_subplot(1,1,1)
    for i in range(len(left_imgs)):
        left_img = cv2.imread(left_imgs[i])
        left_img = calib.left_remap(left_img)
        right_img = cv2.imread(right_imgs[i])
        right_img = calib.right_remap(right_img)
    
        depth, disp = get_depth(left_img,right_img, 100,100)
   
        ax.clear(); ax.imshow(disp);
        plt.pause(0.05)
    plt.show()



