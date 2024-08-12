import os
import shutil
import xml.etree.ElementTree as ET

from tqdm import tqdm

if __name__ == '__main__':
    name = "pd2"
    items = open("/home/Yang/Project/ICH/ICH/VOC2007/ImageSets/Main/04_pd2_test.txt").read().strip().split()
    xml_path = "/home/Yang/Project/ICH/ICH/VOC2007/Annotations"
    txt_path = "/home/Yang/Project/ICH/ICH/YOLO_ICH/labels/%s" % name
    os.makedirs(txt_path, exist_ok=True)

    classes = ["EDH", "IVH", "IPH", "SAH", "SDH"]
    for item in tqdm(items):
        ## 标签转换
        tree = ET.parse(os.path.join(xml_path, item + ".xml"))
        root = tree.getroot()
        image_width = int(512)
        image_height = int(512)
        objects = root.findall('object')
        bboxes = []
        class_names = []
        for obj in objects:
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            class_name = obj.find('name').text
            c1 = round((xmin + xmax) / (image_width * 2), 6)
            c2 = round((ymin + ymax) / (image_height * 2), 6)
            c3 = round((xmax - xmin) / image_width, 6)
            c4 = round((ymax - ymin) / image_height, 6)
            if class_name in classes:
                bboxes.append([c1, c2, c3, c4])
                class_names.append(class_name)
        with open(os.path.join(txt_path, item + ".txt"), 'w') as file:
            for bbox, class_name in zip(bboxes, class_names):
                file.write(
                    f"{classes.index(class_name)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
        file.close()
