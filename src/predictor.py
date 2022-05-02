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
        self.results = [] # [[[xmin, ymin, xmax, ymax, certainty, classID, className, [centerX, centerY]], ...], ...]
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
                self.totalPredictions += 1

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
                    oldResult[i][0] = res[0]
            self.results = oldResult

            with open(f'{self.outputPath}/results.pkl', 'wb') as f:
                pickle.dump(self.results, f)


        print(f'Saved results to {self.outputPath}/results.pkl, {len(self.results)} images, added {self.totalPredictions} predictions')




    def getCenter(self, xmin, ymin, xmax, ymax):
        return np.array([(xmin + xmax)/2, (ymin + ymax)/2]).astype(int)


def makeVideo(imgFiles, picklePath, videoName='results.avi'):
    '''Makes a video with coordinates of bounding boxes added to the frames.
    input:
        imgFiles: list of image filepaths
        picklePath: path to pickle file containing results
        videoName: name of video to be saved
    output:
        video: saved videofile in current directory
    '''
    with open(picklePath, 'rb') as f:
        results = pickle.load(f)

    size = (1280, 720)
    fps = 30
    
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(videoName, fourcc, fps, size)

    for i, imgFile in enumerate(imgFiles):
        img = cv2.imread(imgFile) # Read image
        if len(results[i][0]):
            text = f'Detected: x={results[i][0][7][0]}px , y={results[i][0][7][1]}px, z=-m'
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
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    cv2.putText(img,
                boxText,
                (xmin, ymin),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
                cv2.LINE_4)
    return img






if __name__== '__main__':
    imgs = 'data/results/Stereo_conveyor_with_occlusions/left'
    imgs = [os.path.join(imgs, f) for f in os.listdir(imgs) if (f.endswith('.png') or f.endswith('.jpg'))]
    # imgs = imgs[0:482] # without occlusions
    # imgs = imgs[0:487] # with occlusions
    # imgs = imgs[90:100] # with occlusions

    # pred = Predictor(imgs=imgs, modelName='yolov5s.pt', outputPath='data/results/Stereo_conveyor_with_occlusions/left')
    # pred = Predictor(imgs=imgs, modelName='only_boxes.pt', outputPath='data/results/Stereo_conveyor_with_occlusions/left')
    # pred = Predictor(imgs=imgs, modelName='bestest.pt', outputPath='data/results/Stereo_conveyor_with_occlusions/left')
    
    makeVideo(imgFiles=imgs, picklePath='data/results/Stereo_conveyor_with_occlusions/left/results.pkl')

    # with open(f'src/results_without_occlusions.pkl', 'rb') as f:
    #     oldResult = pickle.load(f)
    #     print(oldResult[100])