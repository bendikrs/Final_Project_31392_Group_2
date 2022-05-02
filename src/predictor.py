import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
import glob
import pickle
from calibration import *
from depth_map import *

class Predictor:
    def __init__(self, left_imgs, right_imgs=None, sliceIndex=(0,-1), modelName='bestest.pt', modelPath='data/yoloModels/', outputPath='data/results'):
        if  type(left_imgs) == str:
            self.left_imgs = [os.path.join(left_imgs, f) for f in os.listdir(left_imgs) if f.endswith('.png') or f.endswith('.jpg')]
        else:
            self.left_imgs = left_imgs
        if  type(right_imgs) == str:
            self.right_imgs = [os.path.join(right_imgs, f) for f in os.listdir(right_imgs) if f.endswith('.png') or f.endswith('.jpg')]
        else:
            self.right_imgs = right_imgs
        assert len(self.left_imgs) == len(self.right_imgs)
        
        self.sliceIndex = sliceIndex
        self.left_imgs = self.left_imgs[self.sliceIndex[0]:self.sliceIndex[1]]
        self.right_imgs = self.right_imgs[self.sliceIndex[0]:self.sliceIndex[1]]
        if self.sliceIndex == (0,-1): 
            self.sliceIndex = 'all'
        else:
            self.sliceIndex = f'{self.sliceIndex[0]}-{self.sliceIndex[1]}'


        self.outputPath = outputPath
        self.modelName = modelName
        self.totalPredictions = 0
        if self.modelName == 'yolov5s.pt':
            # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s.pt', _verbose=False, force_reload=True)
            self.model = torch.hub.load('src/yolov5', 'custom', path='src/yolov5/yolov5s.pt', source='local', _verbose=False, force_reload=True)
            self.model.classes = [41, 73] # cup and book
    
        else:
            self.model = torch.hub.load('src/yolov5','custom', path=modelPath+modelName, source='local', _verbose=False, force_reload=True)

        self.model.conf = 0.32
        self.results = [] # [[[xmin, ymin, xmax, ymax, certainty, classID, className, [centerX, centerY]], ...], ...]
        self.getPredictions()


    def __str__(self):
        return f'{len(self.left_imgs)} images, {self.name}'

    def calculateDepth(self, left_img, right_img, x_px, y_px):
        calib = Calibration(None,None,None)
        calib.load("Calibration_result.bin")

        left_img = cv2.imread(left_img)
        left_img = calib.left_remap(left_img)
        right_img = cv2.imread(right_img)
        right_img = calib.right_remap(right_img)
 
        depth, disp = get_depth_wls(left_img, right_img, x_px, y_px)    

        return depth

    def getPredictions(self):
        for i in tqdm(range(len(self.left_imgs))):
            currResult = self.model(self.left_imgs[i]).pandas().xyxy[0].to_numpy()
            self.model(self.left_imgs[i]).save(labels=True, save_dir=f'{os.getcwd()}/{self.outputPath}_{self.sliceIndex}/')
            self.results.append([])
            for j, res in enumerate(currResult):
                self.results[-1].append([])
                for k, val in enumerate(res):
                    self.results[-1][-1].append(val)
                
                center = self.getCenter(*self.results[-1][-1][:4])
                self.results[-1][-1].append(center)

                z = self.calculateDepth(self.left_imgs[i], self.right_imgs[i], center[0], center[1])
                self.results[-1][-1][7].append(z)
                self.totalPredictions += 1

        # Pickle results in outputPath
        if not os.path.exists(f'{self.outputPath}_{self.sliceIndex}/results_{self.sliceIndex}.pkl'):
            with open(f'{self.outputPath}_{self.sliceIndex}/results_{self.sliceIndex}.pkl', 'wb') as f:
                pickle.dump(self.results, f)
        else:
            with open(f'{self.outputPath}_{self.sliceIndex}/results_{self.sliceIndex}.pkl', 'rb') as f:
                oldResult = pickle.load(f)

            for i, res in enumerate(self.results):
                if len(res) > 0:
                    # certaintyIndex = res.index(max([x[4] for x in res])) # Find index of prediction with highest certainty
                    # oldResult[i].append(res[certaintyIndex])
                    self.totalPredictions += 1
                    oldResult[i][0] = res[0]
            self.results = oldResult

            with open(f'{self.outputPath}/results.pkl', 'wb') as f:
                pickle.dump(self.results, f)


        print(f'Saved results to {self.outputPath}/results.pkl, {len(self.results)} images, added {self.totalPredictions} predictions')




    def getCenter(self, xmin, ymin, xmax, ymax):
        return [int((xmin + xmax)/2), int((ymin + ymax)/2)]


def makeVideo(imgPath, picklePath, videoName='results.avi'):
    '''Makes a video with coordinates and bounding boxes added to the frames.
    input:
        imgFiles: list of image filepaths
        picklePath: path to pickle file containing results, or the results list
        videoName: name of video to be saved
    output:
        video: saved videofile in current directory
    '''
    slicing = picklePath.split('_')[-1]
    if slicing == 'all':
        slicing = (0,-1)
    else:
        slicing = slicing.split('-')
        slicing = [int(slicing[0]), int(slicing[1].split('.')[0])]
        print(slicing)

    imgFiles = [os.path.join(imgPath, f) for f in os.listdir(imgPath) if (f.endswith('.png') or f.endswith('.jpg'))]
    imgFiles = imgFiles[slicing[0]:slicing[1]]

    if type(picklePath) == str:
        with open(picklePath, 'rb') as f:
            results = pickle.load(f)
    else:
        results = picklePath

    assert len(imgFiles) == len(results)

    size = (1280, 720)
    fps = 30
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(videoName, fourcc, fps, size)

    for i, imgFile in tqdm(enumerate(imgFiles)):
        img = cv2.imread(imgFile) # Read image
        # print(results[i])
        if results[i]:
            text = f'Detected: x={results[i][0][7][0]}px , y={results[i][0][7][1]}px, z={results[i][0][7][2]:.2f}m'
            img = addBoundingBox(img, *results[i][0][:4], text[:8])
        else:
            text = f'Detected: x= -px , y= -px, z= -m'
        

        cv2.putText(img, 
                text, 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
        out.write(img)
    out.release()

def addBoundingBox(img, xmin, ymin, xmax, ymax, boxText, color=(0, 255, 0), thickness=2):
    '''
    input:
        img: image to add bounding box to
        xmin: xmin coordinate of bounding box
        ymin: ymin coordinate of bounding box
        xmax: xmax coordinate of bounding box
        ymax: ymax coordinate of bounding box
        boxText: text to be added to bounding box
        color: color of bounding box
        thickness: thickness of bounding box
    output:
        img: image with bounding box added
    '''
    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
    
    # check if box is too big
    x_max, y_max = 300, 300
    if xmax - xmin > x_max or ymax - ymin > y_max:
        return img

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    cv2.putText(img,
                boxText,
                (xmin, ymin-2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_4)
    return img






if __name__== '__main__':
    left_imgs = 'data/Stereo_conveyor_without_occlusions/left'
    right_imgs = 'data/Stereo_conveyor_without_occlusions/right'
    # imgs = [os.path.join(imgs, f) for f in os.listdir(imgs) if (f.endswith('.png') or f.endswith('.jpg'))]
    # imgs = imgs[0:482] # without occlusions
    # imgs = imgs[0:487] # with occlusions
    # imgs = imgs[90:100] # with occlusions

    # imgs = imgs[949:1153] # with occlusions, cup

    pred = Predictor(left_imgs, right_imgs, sliceIndex=(1064, 1244), modelName='yolov5s.pt', outputPath='data/results/Stereo_conveyor_with_occlusions/cup')
    # pred = Predictor(imgs=imgs, modelName='only_boxes.pt', outputPath='data/results/Stereo_conveyor_with_occlusions/left')
    # pred = Predictor(imgs=imgs, modelName='bestest.pt', outputPath='data/results/Stereo_conveyor_with_occlusions/left')
    

    


    # makeVideo(imgPath=left_imgs, picklePath='data/results/Stereo_conveyor_with_occlusions/cup_949-1153/results_949-1153.pkl', videoName='new_results.avi')