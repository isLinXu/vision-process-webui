#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pip3 install requests

import hashlib
import random
import time
import requests
import json

class MMLabConfigToken:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config = self.read_json_file()

    def read_json_file(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        return data

    def write_json_file(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def get_nonce(self):
        pool = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        length = 10
        ret = ""
        while len(ret) < length:
            ret += random.choice(pool)
        return ret

    def get_access_token(self):
        access_key = self.config["access_key"]
        secret_key = self.config["secret_key"]
        ts = str(int(time.time()))
        nonce = self.get_nonce()

        concat_string = "uri=/api/v1/openapi/auth&ts=%s&nonce=%s&accessKey=%s&secretKey=%s" % (ts, nonce, access_key, secret_key)
        sign = hashlib.sha256(concat_string.encode("utf-8")).hexdigest()

        url = "https://platform.openmmlab.com/gw/user-service/api/v1/openapi/auth"
        headers = {
            "ts": ts,
            "nonce": nonce,
            "sign": sign,
            "accessKey": access_key
        }
        ret = requests.post(url, headers=headers)
        response_json = ret.json()
        access_token = response_json["data"]["accessToken"]
        return access_token

    def update_access_token(self):
        access_token = self.get_access_token()
        print("access_token: ", access_token)
        self.config["access_token"] = access_token
        self.write_json_file()

def main():
    mmlab_config = MMLabConfigToken("mmlab_config.json")
    mmlab_config.update_access_token()

if __name__ == "__main__":
    main()