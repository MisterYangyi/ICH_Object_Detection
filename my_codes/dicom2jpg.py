import shutil

import SimpleITK
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import pydicom
import os
from PIL import Image


def get_pixels_hu_by_simpleitk(dicom_path):
    image = SimpleITK.ReadImage(dicom_path)
    img_array = SimpleITK.GetArrayFromImage(image)
    # img_array[img_array == -2000] = 0
    return img_array


def normalize_hu(image, min_bound, max_bound):
    MIN_BOUND = min_bound
    MAX_BOUND = max_bound
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def setDicomWinWidthWinCenter(ct_array, windowWidth, windowCenter):
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


if __name__ == '__main__':
    df = pd.read_csv("key_values.csv")
    train = open("train.txt").read().strip().split()
    df = df[df["image"].isin(train)][["path", 'image']]
    save_dir = "E:\RSAN\Deal/train"
    os.makedirs(save_dir, exist_ok=True)
    for item in df.values.tolist():

        data = get_pixels_hu_by_simpleitk(item[0])
        brain = setDicomWinWidthWinCenter(data[0], 80, 40)
        # subdural = setDicomWinWidthWinCenter(data[0], 175, 50)
        # bone = setDicomWinWidthWinCenter(data[0], 3000, 500)
        # im = np.stack((bone, brain, subdural), axis=2)
        Image.fromarray(brain).save(os.path.join(save_dir, item[1]))

    # print(df)
    # exit(0)
    # with open("val.txt", "w")as f:
    #     for item in os.listdir("/home/Yang/Project/ICH/ICH/YOLO_ICH/images/val"):
    #         f.write(item + "\n")
    # f.close()
    #
    # import pandas as pd
    #
    # data = {}
    # dcm_paths = []
    # jpg_names = []
    # for root, dirs, files in os.walk(r"E:\RSAN\Deal\DICOM"):
    #     for file_path in tqdm(files):
    #         if file_path.split(".")[-1] == "dcm":
    #             file_p = os.path.join(root, file_path)
    #             dcm_paths.append(file_p)
    #             jpg_name = file_p.replace("E:\RSAN\Deal\DICOM\\", "").replace("\\", "--").replace("dcm", "jpg")
    #             jpg_names.append(jpg_name)
    # data["path"] = dcm_paths
    # data["image"] = jpg_names
    # pd.DataFrame(data=data).to_csv("key_values.csv")
    # save_dir = "/home/Yang/Project/ICH/ICH/YOLO_ICH/images/pd1"
    # os.makedirs(save_dir, exist_ok=True)
    # data = get_pixels_hu_by_simpleitk(os.path.join(root, file_path))
    # brain = setDicomWinWidthWinCenter(data[0], 80, 40)
    # subdural = setDicomWinWidthWinCenter(data[0], 175, 50)
    # bone = setDicomWinWidthWinCenter(data[0], 3000, 500)
    # im = np.stack((bone, brain, subdural), axis=2)
    # Image.fromarray(im).save(os.path.join(save_dir, file_path.replace(".dcm", ".jpg")))

    # for root, dirs, files in os.walk(r"/home/Yang/share/pd3"):
    #     for file_path in tqdm(files):
    #         if file_path.split(".")[-1] == "dcm":
    #             save_dir = "/home/Yang/Project/ICH/ICH/YOLO_ICH/images/pd1"
    #             os.makedirs(save_dir, exist_ok=True)
    #             data = get_pixels_hu_by_simpleitk(os.path.join(root, file_path))
    #             brain = setDicomWinWidthWinCenter(data[0], 80, 40)
    #             subdural = setDicomWinWidthWinCenter(data[0], 175, 50)
    #             bone = setDicomWinWidthWinCenter(data[0], 3000, 500)
    #             im = np.stack((bone, brain, subdural), axis=2)
    #             Image.fromarray(im).save(os.path.join(save_dir, file_path.replace(".dcm", ".jpg")))
