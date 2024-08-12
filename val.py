from ultralytics import YOLO

# Load a model
model = YOLO("/home/Yang/Project/ICH/YOLOV8.1/runs/detect/trian_Concat_BiFPN/weights/best.pt")
result = model.val(data='/home/Yang/Project/ICH/ICH/YOLO_ICH/ICH.yaml', name="demo", split="val", plots=True,
          device=[0, 1],save_json=False)
print(result)