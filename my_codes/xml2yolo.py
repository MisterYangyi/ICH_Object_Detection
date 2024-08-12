import os
import shutil
import xml.etree.ElementTree as ET


def tansform_task(items, mode, voc_root, yolo_root, classes):
    yolo_labels_dir = os.path.join(yolo_root, "labels", mode)
    yolo_images_dir = os.path.join(yolo_root, "labels", mode)
    os.makedirs(yolo_labels_dir, exist_ok=True)
    os.makedirs(yolo_images_dir, exist_ok=True)
    for item in items:

        ## 标签转换
        tree = ET.parse(os.path.join(voc_root, "Annotations", item + ".xml"))
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
                print(class_name)
                bboxes.append([c1, c2, c3, c4])
                class_names.append(class_name)
        with open(os.path.join(yolo_labels_dir, item + ".txt"), 'w') as file:
            for bbox, class_name in zip(bboxes, class_names):
                file.write(
                    f"{classes.index(class_name)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
        file.close()

        ## 图片复制
        shutil.copyfile(src=os.path.join(voc_root, "JPEGImages", item + ".jpg"),
                        dst=os.path.join(yolo_images_dir, item + ".jpg"))


if __name__ == '__main__':
    voc_root = r"G:\Brain\labeled\XML"
    yolo_root = r"G:\Brain\labeled\TXT"
    classes = ["EDH", "IVH", "IPH", "SAH", "SDH"]
    train_items = open(os.path.join(voc_root, "ImageSetS/Main", "train.txt")).read().strip().split()
    val_items = open(os.path.join(voc_root, "ImageSetS/Main", "val.txt")).read().strip().split()
    tansform_task(train_items, "train", voc_root, yolo_root, classes)
