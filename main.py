
import cv2
import numpy as np
from matplotlib import pyplot as plt
from src.predictor import Predictor
import os


# cap = cv2.VideoCapture('Robots.mp4')

# path = 'data/Stereo_conveyor_without_occlusions/left'
# imgs = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
# print(imgs[0])
# path = 'data/Stereo_conveyor_without_occlusions/left'
path = 'data/conveyorImages'
# predictor = Predictor(imgs=path, modelName='yolo5s.pt', outputPath='data/results/Stereo_conveyor_without_occlusions/left')
predictor = Predictor(imgs=path, modelName='bestest.pt', outputPath='data/results/Stereo_conveyor_without_occlusions/left')

# # read first frame
# ret, prev_frame = cap.read()

# ## set window up
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# (h, w, d) = prev_frame.shape
# r = 600.0 / w
# cv2.resizeWindow('output', 600,int(h * r))


# ## start main loop
# #while ret is True:
# for i in range(200):
#     cv2.imshow('output', curr_frame)
#     cv2.waitKey(20)
#     prev_frame = curr_frame
                   
                       
# cv2.destroyAllWindows()
# cap.release()
