# vision-process-webui

![](https://user-images.githubusercontent.com/59380685/265589543-e255edad-11a9-4be4-8a8d-870dcd00cc08.png)

![GitHub watchers](https://img.shields.io/github/watchers/isLinXu/vision-process-webui.svg?style=social) ![GitHub stars](https://img.shields.io/github/stars/isLinXu/vision-process-webui.svg?style=social) ![GitHub forks](https://img.shields.io/github/forks/isLinXu/vision-process-webui.svg?style=social) ![GitHub followers](https://img.shields.io/github/followers/isLinXu.svg?style=social)
 [![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge&style=flat)](https://github.com/isLinXu/vision-process-webui)  ![img](https://badgen.net/badge/icon/learning?icon=deepscan&label)![GitHub repo size](https://img.shields.io/github/repo-size/isLinXu/vision-process-webui.svg?style=flat-square) ![GitHub language count](https://img.shields.io/github/languages/count/isLinXu/vision-process-webui)  ![GitHub last commit](https://img.shields.io/github/last-commit/isLinXu/vision-process-webui) ![GitHub](https://img.shields.io/github/license/isLinXu/vision-process-webui.svg?style=flat-square)![img](https://hits.dwyl.com/isLinXu/vision-process-webui.svg)

---

# performance&demo

## classification

![](https://user-images.githubusercontent.com/59380685/265589543-e255edad-11a9-4be4-8a8d-870dcd00cc08.png)

| ![](https://user-images.githubusercontent.com/59380685/265667039-ce3f2122-4317-4c57-9bab-9ebc792ca23b.png) |
| ------------------------------------------------------------ |
| ![](https://user-images.githubusercontent.com/59380685/265667095-6a0d4513-cb21-42ff-b77c-723da474d0fe.png) |
| ![](https://user-images.githubusercontent.com/59380685/265667360-20438bef-91ee-4847-a5e2-16ef4e658935.png) |

---


## detection
| ![](https://user-images.githubusercontent.com/59380685/265492490-9353cd87-052d-4dcb-9115-afb7954c00dd.png) | ![](https://user-images.githubusercontent.com/59380685/265493664-939d5c5f-f571-4a84-b6e9-6193f4613f37.png) | ![](https://user-images.githubusercontent.com/59380685/265493715-e920d82e-c85d-43e1-a7ae-c0a706c0bb95.png) | ![](https://user-images.githubusercontent.com/59380685/265493821-19954089-befb-4cec-baac-688427a84589.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                          YOLOv8-det                          |                          YOLOv8-seg                          |                          YOLOv8-seg                          |                          YOLOv8-seg                          |



| ![](https://user-images.githubusercontent.com/59380685/265312963-41d535a2-f920-443e-a048-6428983fac46.png) | ![](https://user-images.githubusercontent.com/59380685/265313403-9e4937bc-a497-4806-ab9c-99a3b864f2d9.png) | ![](https://user-images.githubusercontent.com/59380685/265313486-6a3785ee-0202-4a4f-9816-23dbb0a3588c.png) |
|:----------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------:|
|                                                                                                            |                                                                                                            |                                                                                                            |
|                                                                                                            |                                                                                                            |                                                                                                            |
|                                                   YOLOv3                                                   |                                                   YOLOv5                                                   |                                                   YOLOX                                                    |
| ![](https://user-images.githubusercontent.com/59380685/265494398-e053e543-11ec-4fc7-81ad-32bb97983fc0.png) | ![](https://user-images.githubusercontent.com/59380685/265494778-5262fb37-40f2-46df-b31e-089775d9223c.png) | ![](https://user-images.githubusercontent.com/59380685/265507024-baa0f476-4800-4bba-9129-5e2744468495.png) |
|                                                  YOLO-NAS                                                  |                                                  PP-YOLOE                                                  |                                                  RT-Detr                                                   |

---

## segmentation
| ![](https://user-images.githubusercontent.com/59380685/265508535-ce1820d2-e161-4ddf-bd7a-70c5306ee5d5.png) | ![](https://user-images.githubusercontent.com/59380685/265508607-9c07e74c-a083-4df7-bd31-38d1bb402b25.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](https://user-images.githubusercontent.com/59380685/265508557-bc5baa23-f5a0-408e-88b6-c112f9891dd8.png) | ![](https://user-images.githubusercontent.com/59380685/265508693-189b0990-149a-4fe6-bada-bb8ae7c09042.png) |
|                      mobile-sam[point]                       |                                mobile-sam[bbox]                              |

---

# support list

## classification


## detection
- [x] **yolov3**([src](https://docs.ultralytics.com/models/yolov3/) | [code](webui/yolov3_ui.py))
- [ ] **yolov4**([src](https://docs.ultralytics.com/models/yolov4/) | [code](webui/yolov4_ui.py))
- [x] **yolov5**([src](https://docs.ultralytics.com/models/yolov5/) | [code](webui/det/yolov5_ui.py))
- [ ] **yolov6**([src](https://docs.ultralytics.com/models/yolov6/) | [code](webui/det/yolov6_ui.py))
- [ ] **yolov7**([src](https://docs.ultralytics.com/models/yolov7/) | [code](webui/yolov7_ui.py))
- [x] **yolox**([src](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/detection_models/yolox.py) | [code](webui/yolox_ui.py)))
- [x] **ppyoloe**([src](https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/training/models/detection_models/pp_yolo_e) | [code](webui/ppyoloe_ui.py)))
- [x] **yolo-nas**([src](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md) | [code](webui/yolonas_ui.py)))
- [x] **yolov8**([src](https://docs.ultralytics.com/models/yolov8/) | [code](webui/det/yolov8_ui.py))
- [x] **rtdetr-l**([src](https://docs.ultralytics.com/models/rtdetr/) | [code](webui/det/rt_detr_ui.py)))

---

## segmentation

- [x] **mobile_sam**([src](https://docs.ultralytics.com/models/mobile-sam/) | [code](webui/seg/mobilesam_ui.py))
- [x] **fast_sam**([src](https://docs.ultralytics.com/models/fast-sam/) | [code](webui/seg/fastsam_ui.py))


