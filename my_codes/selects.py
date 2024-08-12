import os

from tqdm import tqdm

if __name__ == '__main__':
    dir = "/home/Yang/Project/ICH/RetinaNet/runs/20240505161757_01_text_iou0.1/detection-results"
    dst = "/home/Yang/Project/ICH/RetinaNet/runs/20240505161757_01_text_iou0.1/detection-results2"
    os.makedirs(dst, exist_ok=True)
    for item in tqdm(os.listdir(dir)):
        with open(os.path.join(dst, item), "w") as f:
            for line in open(os.path.join(dir, item)).readlines():
                # print(line)
                # print(lines)
                conf = line.split(" ")
                if float(conf[1]) > 0.01:
                    f.write(line)
