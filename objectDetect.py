import torch

# Model
model = torch.hub.load('ultralytics/yolov5','custom', path='data/yoloModels/best.pt')  # or yolov5m, yolov5l, yolov5x, custom
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# print(dir(model))

# Images
img = 'data/conveyorImages/conveyorbox.png'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
# print(results.pandas().xyxy[0][1].numpy())
x0, y0, x1, y1, _, _ = results.xyxy[0][0].numpy().astype(float)
print(results)
print(x0, y0, x1, y1) 
