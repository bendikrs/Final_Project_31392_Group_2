
import cv2
import numpy as np
from matplotlib import pyplot as plt
from src import objectDetect


cap = cv2.VideoCapture('Robots.mp4')

# read first frame
ret, prev_frame = cap.read()

## set window up
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
(h, w, d) = prev_frame.shape
r = 600.0 / w
cv2.resizeWindow('output', 600,int(h * r))


## start main loop
#while ret is True:
for i in range(200):






    cv2.imshow('output', curr_frame)
    cv2.waitKey(20)
    prev_frame = curr_frame
                   
                       
cv2.destroyAllWindows()
cap.release()
