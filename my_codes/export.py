from ultralytics import YOLO

# Load a model
model = YOLO('/home/Yang/Project/ICH/YOLOV8.1/runs/detect/x-fracure6/weights/best1.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom trained model

# Export the model
model.export(format='onnx')