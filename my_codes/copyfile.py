import os
from tqdm import tqdm
import shutil

src = "/home/Yang/Project/ICH/ICH/YOLO_ICH/labels/test"

dst = "/home/Yang/Project/ICH/ICH/VOC2007/JPEGImages"
with open("/home/Yang/Project/ICH/ICH/VOC2007/01_text.txt", "w") as f:
    for file in tqdm(os.listdir(src)):
        # shutil.move(os.path.join(src, file), os.path.join(dst, file))
        f.write(file[:-4] + "\n")
