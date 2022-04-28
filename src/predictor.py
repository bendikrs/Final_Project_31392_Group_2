import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
import glob

class Predictor:
    def __init__(self, imgs, modelName='bestest.pt', modelPath='../data/yoloModels/', outputPath='../data/yoloPredictions/'):
        self.imgs = imgs
        self.outputPath = outputPath
        self.modelName = modelName
        self.model = torch.hub.load('yolov5','custom', path=modelPath+modelName, source='local', _verbose=False)
        self.results = [] # [[[xmin, xmax, ymin, ymax, certainty, classID, className, [centerX, centerY]], ...], ...]
        self.getPredictions()


    def __str__(self):
        return f'{len(self.imgs)} images, {self.name}'

    def getPredictions(self):
        for i, img in enumerate(self.imgs):
            currResult = self.model(img).pandas().xyxy[0].to_numpy()
            self.model(img).save(labels=True, save_dir=f'{os.getcwd()}/{self.outputPath}/{self.modelName[0:-3]}')

            self.results.append([])
            for j, res in enumerate(currResult):
                self.results[-1].append([])
                for k, val in enumerate(res):
                    self.results[-1][-1].append(val)
                
                # self.results[-1][-1].append(i)
                center = self.getCenter(*self.results[-1][-1][:4])
                self.results[-1][-1].append(center)


    def getCenter(self, xmin, xmax, ymin, ymax):
        return np.array([(xmin + xmax)/2, (ymin + ymax)/2]).astype(int)
