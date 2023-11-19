#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import requests
import cv2
import numpy as np
from urllib.request import urlretrieve, urlopen

def detect_objects(image_url, config_file="mmlab_config.json"):
    # 读取JSON文件中的内容
    with open(config_file, "r") as file:
        credentials = json.load(file)

    # 获取access_token
    access_token = credentials["access_token"]

    url = "https://platform.openmmlab.com/gw/model-inference/openapi/v1/detection"

    body = {
      "resource": image_url,
      "resourceType": "URL",
      "backend": "PyTorch",
      "requestType": "SYNC",
      "algorithm": "YOLOX"
    }

    headers = {
      'Authorization': access_token
    }

    response = requests.post(url, headers=headers, json=body)
    detection_data = response.json()["data"]["result"]["top_n_result"]

    # 下载图像
    image = np.asarray(bytearray(urlopen(image_url).read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # 在图像上绘制检测框和标签
    for item in detection_data:
        print("item: ", item)
        label = item["label"]
        score = float(item["onfidence"])
        bbox = item["coordinate"]

        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {score:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存图像
    cv2.imwrite("output.jpg", image)

    return detection_data

# 示例用法
image_url = "https://oss.openmmlab.com/web-demo/static/one.e9be6cd7.jpg"
detections = detect_objects(image_url)
print(detections)