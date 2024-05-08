# -*- coding: utf-8 -*-
import os
import json
import string

all_characters = string.digits + string.ascii_lowercase + string.ascii_uppercase
category_mapping = {c: i for i, c in enumerate(all_characters)}

# ISAT格式的实例分割标注文件
ISAT_FOLDER = "./dataset/Sina/annotations"
# YOLO格式的实例分割标注文件
YOLO_FOLDER = "./dataset/Sina/labels"

# 创建YoloV8标注的文件夹
if not os.path.exists(YOLO_FOLDER):
    os.makedirs(YOLO_FOLDER)

for filename in os.listdir(ISAT_FOLDER):
    if not filename.endswith(".json"):
        # 不是json格式, 跳过
        continue
    # 载入ISAT的JSON文件
    with open(os.path.join(ISAT_FOLDER, filename), "r") as f:
        isat = json.load(f)
    # 提取文件名(不带文件后缀)
    image_name = filename.split(".")[0]
    # Yolo格式的标注文件名, 后缀是txt
    yolo_filename = f"{image_name}.txt"
    # 获取标签信息
    characters = image_name.replace("-", "")
    # 写入信息
    with open(os.path.join(YOLO_FOLDER, yolo_filename), "w") as f:
        # 获取图像信息
        image_width = isat["info"]["width"]
        image_height = isat["info"]["height"]
        # 获取实例标注数据
        for c, annotation in zip(characters, isat["objects"]):
            # 从字典里面查询类别ID
            category_id = category_mapping[c]
            # 提取分割信息
            segmentation = annotation["segmentation"]
            segmentation_yolo = []
            # 遍历所有的轮廓点
            for segment in segmentation:
                # 提取轮廓点的像素坐标 x, y
                x, y = segment
                # 归一化处理
                x_center = x / image_width
                y_center = y / image_height
                # 添加到segmentation_yolo里面
                segmentation_yolo.append(f"{x_center:.4f} {y_center:.4f}")
            segmentation_yolo_str = " ".join(segmentation_yolo)
            # 添加一行Yolo格式的实例分割数据
            # 格式如下: class_id x1 y1 x2 y2 ... xn yn\n
            f.write(f"{category_id} {segmentation_yolo_str}\n")
