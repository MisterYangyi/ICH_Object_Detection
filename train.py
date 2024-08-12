from ultralytics import YOLO

# Load a model
# model = YOLO("/home/Yang/Project/ICH/YOLOV8.1/ultralytics/cfg/models/v8/yolov8s-bifpn.yaml")
model = YOLO("/home/Yang/Project/ICH/YOLOV8.1/ultralytics/cfg/models/v8/yolov8s.yaml")

# load a pretrained model (recommended for training)
# model = YOLO("ultralytics/cfg/models/v8/yolov8x-p6.yaml")
# Train the model with 2 GPUs
# model.train(data='datasets/BL/il.yaml', imgsz=512, batch=128, device=[0, 1], patience=0, cos_lr=True, close_mosaic=0)
results = model.train(data='/home/Yang/Project/ICH/ICH/YOLO_ICH/ICH.yaml',
                      epochs=300,
                      imgsz=512,
                      batch=360,
                      pretrained=False,
                      device=[0, 1],
                      patience=100,
                      name="ICH_Brain"
                      )

# model.val(data='/home/Yang/Project/ICH/ICH/YOLO_ICH/labels/ICH.yaml', name="val", split="val", plots=True, device=[0, 1])
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
