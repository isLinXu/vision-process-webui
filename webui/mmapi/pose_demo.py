#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pip3 install requests

import requests

import json

import requests

with open("mmlab_config.json", "r") as file:
    credentials = json.load(file)

access_token = credentials["access_token"]

url = "https://platform.openmmlab.com/gw/model-inference/openapi/v1/pose"
# access_token = "your access token"

body = {
  "resource": "https://oss.openmmlab.com/web-demo/static/humanOne.ca88d32a.jpg",
  "resourceType": "URL",
  "requestType": "SYNC",
  "algorithm": "topdown_heatmap",
  "dataset": "COCO",
  "subType": "wholebody",
  "method": "top_down"
}

headers = {
  'Authorization': access_token
}


response = requests.post(url, headers=headers, json=body)


print(response.text)