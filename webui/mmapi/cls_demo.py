#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pip3 install requests
# pip3 install opencv-python
# https://platform.openmmlab.com/docs/zh-CN/open-api/guides/image-segmentation

import requests
import json
import cv2

class MMLabInference:
    def __init__(self, config_file_path):
        self.config = self.read_json_file(config_file_path)
        self.access_token = self.config["access_token"]

    def read_json_file(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def get_classification_result(self):
        url = "https://platform.openmmlab.com/gw/model-inference/openapi/v1/classification"
        algorithm_list = ["Swin-Transformer", "VGG", "SEResNet", "ShuffleNet v1", "ShuffleNet v2", "FP16", "MobileNetV2", "ResNet", "ResNeXt"]
        backend_list = ["PyTorch", "TensorRT", "ONNXRuntime", "OpenPPL"]
        body = {
            "resource": "https://oss.openmmlab.com/web-demo/static/one.b7608e9b.jpg",
            "resourceType": "URL",
            "requestType": "SYNC",
            "backend": "PyTorch",
            "algorithm": "Swin-Transformer",
            "dataset": "ImageNet"
        }

        headers = {
            'Authorization': self.access_token
        }

        response = requests.post(url, headers=headers, json=body)
        print("response: ", response.json())
        return response.json()['data']['result']

    def download_image(self, url, save_path):
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)

    def draw_result_on_image(self, image_path, results):
        image = cv2.imread(image_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 30
        for result in results:
            text = f"Class: {result['class']}, Confidence: {result['score']}"
            cv2.putText(image, text, (10, y), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            y += 30
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.imwrite("result.jpg", image)
        cv2.destroyAllWindows()

def main():
    mmlab_inference = MMLabInference("mmlab_config.json")
    results = mmlab_inference.get_classification_result()
    print(results)

    image_url = "https://oss.openmmlab.com/web-demo/static/one.b7608e9b.jpg"
    image_path = "input_image.jpg"
    mmlab_inference.download_image(image_url, image_path)

    mmlab_inference.draw_result_on_image(image_path, results)

if __name__ == "__main__":
    main()