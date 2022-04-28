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
    depth = 2.45 * 120 / disp[px,py]
    return depth, disp


    

if __name__ == "__main__":
    calib = Calibration(None,None,None)
    calib.load("../Calibration_result.bin")

    left_imgs = glob.glob("../data/conveyorImages/left/*")
    right_imgs = glob.glob("../data/conveyorImages/right/*")
    assert left_imgs
    assert right_imgs
    
    left_img = cv2.imread(left_imgs[135])
    left_img = calib.left_remap(left_img)
    right_img = cv2.imread(right_imgs[135])
    right_img = calib.right_remap(right_img)
    
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,18))
    ax[0].imshow(left_img)
    ax[0].set_title('left image')
    ax[1].imshow(right_img)
    ax[1].set_title('right image')

    
    depth, disp = get_depth(left_img,right_img, 100,100)
    
    plt.figure(figsize=(18,18))
    plt.gray()
    plt.imshow(disp)
    plt.show()


