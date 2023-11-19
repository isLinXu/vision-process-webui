#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pip3 install requests

import cv2
import json

import numpy as np
import requests
from urllib.request import urlopen

class Segmentation:
    def __init__(self, access_token, url):
        self.access_token = access_token
        self.url = url

    def image_url_to_numpy(self, image_url):
        image = np.asarray(bytearray(urlopen(image_url).read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    def segment_image(self, image_url, backend="PyTorch", request_type="SYNC", algorithm="deeplabv3plus", dataset="VOC"):
        body = {
            "resource": image_url,
            "resourceType": "URL",
            "backend": backend,
            "requestType": request_type,
            "algorithm": algorithm,
            "dataset": dataset
        }

        headers = {
            'Authorization': self.access_token
        }

        response = requests.post(self.url, headers=headers, json=body)
        print("response: ", response.json())
        result = response.json()["data"]["result"]
        return result

    def save_image(self, image_url, output_filename):
        image = self.image_url_to_numpy(image_url)
        cv2.imwrite(output_filename, image)

def main():
    # 读取 JSON 文件
    with open("mmlab_config.json", "r") as file:
        credentials = json.load(file)

    access_token = credentials["access_token"]
    url = "https://platform.openmmlab.com/gw/model-inference/openapi/v1/segmentation"

    segmentation = Segmentation(access_token, url)

    image_url = "https://oss.openmmlab.com/web-demo/static/one.d3f26af8.jpg"
    segmentation.save_image(image_url, "input_seg.jpg")

    result_url = segmentation.segment_image(image_url)
    segmentation.save_image(result_url, "output_seg.jpg")

    print(result_url)

if __name__ == "__main__":
    main()