import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
import glob
import pickle

class Predictor:
    def __init__(self, imgs, modelName='bestest.pt', modelPath='data/yoloModels/', outputPath='data/results'):
        if  type(imgs) == str:
            self.imgs = [os.path.join(imgs, f) for f in os.listdir(imgs) if f.endswith('.png')]
        else:
            self.imgs = imgs
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
        self.results = [] # [[[xmin, xmax, ymin, ymax, certainty, classID, className, [centerX, centerY]], ...], ...]
        self.getPredictions()


    def __str__(self):
        return f'{len(self.imgs)} images, {self.name}'

    def getPredictions(self):
        for img in tqdm(self.imgs):
            currResult = self.model(img).pandas().xyxy[0].to_numpy()
            self.model(img).save(labels=True, save_dir=f'{os.getcwd()}/{self.outputPath}/')
            self.results.append([])
            for j, res in enumerate(currResult):
                self.results[-1].append([])
                for k, val in enumerate(res):
                    self.results[-1][-1].append(val)
                
                center = self.getCenter(*self.results[-1][-1][:4])
                self.results[-1][-1].append(center)

        # Pickle results in outputPath
        if not os.path.exists(f'{self.outputPath}/results.pkl'):
            with open(f'{self.outputPath}/results.pkl', 'wb') as f:
                pickle.dump(self.results, f)
        else:
            with open(f'{self.outputPath}/results.pkl', 'rb') as f:
                oldResult = pickle.load(f)

            for i, res in enumerate(self.results):
                if len(res) > 0:
                    # certaintyIndex = res.index(max([x[4] for x in res])) # Find index of prediction with highest certainty
                    # oldResult[i].append(res[certaintyIndex])
                    self.totalPredictions += 1
                    oldResult[i].append(res[0])
            self.results = oldResult

            with open(f'{self.outputPath}/results.pkl', 'wb') as f:
                pickle.dump(self.results, f)


        print(f'Saved results to {self.outputPath}/results.pkl, {len(self.results)} images, added {self.totalPredictions} predictions')




    def getCenter(self, xmin, xmax, ymin, ymax):
        return np.array([(xmin + xmax)/2, (ymin + ymax)/2]).astype(int)


if __name__== '__main__':
    imgs = 'data/Stereo_conveyor_with_occlusions/left'
    imgs = [os.path.join(imgs, f) for f in os.listdir(imgs) if f.endswith('.png')]

    # imgs = imgs[0:482] # without occlusions
    imgs = imgs[0:487] # with occlusions
    # pred = Predictor(imgs=imgs, modelName='yolov5s.pt', outputPath='data/results/Stereo_conveyor_with_occlusions/left')
    pred = Predictor(imgs=imgs, modelName='bestest.pt', outputPath='data/results/Stereo_conveyor_with_occlusions/left')
    
    # with open(f'data/results/Stereo_conveyor_without_occlusions/left/results.pkl', 'rb') as f:
    #     oldResult = pickle.load(f)
    #     print(len(oldResult))