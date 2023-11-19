#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import requests
import cv2
import numpy as np
from urllib.request import urlopen
import random

class ObjectDetector:
    def __init__(self, config_file="mmlab_config.json"):
        with open(config_file, "r") as file:
            credentials = json.load(file)
        self.access_token = credentials["access_token"]
        self.config = credentials
        self.conf_threshold = 0.45

    # @staticmethod
    def draw_detections(self, image, detections):
        for item in detections:
            label = item["label"]
            score = float(item["onfidence"])
            bbox = item["coordinate"]

            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if score < self.conf_threshold:
                continue
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, f"{label} {score:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image

    def detect_objects(self, image_url):
        url = "https://platform.openmmlab.com/gw/model-inference/openapi/v1/detection"

        body = {
          "resource": image_url,
          "resourceType": "URL",
          "backend": "PyTorch",
          "requestType": "SYNC",
          "algorithm": "YOLOX"
        }

        headers = {
          'Authorization': self.access_token
        }

        response = requests.post(url, headers=headers, json=body)
        detection_data = response.json()["data"]["result"]["top_n_result"]

        image = np.asarray(bytearray(urlopen(image_url).read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        image = self.draw_detections(image, detection_data)

        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.imwrite("output.jpg", image)

        return detection_data

# 示例用法
image_url = "https://oss.openmmlab.com/web-demo/static/one.e9be6cd7.jpg"
detector = ObjectDetector()
detections = detector.detect_objects(image_url)
print(detections)