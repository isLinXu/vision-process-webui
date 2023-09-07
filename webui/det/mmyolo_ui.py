import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou
def nms(bboxes, iou_threshold, threshold=0):
    bboxes = [box for box in bboxes if box[4] > threshold]

    bboxes = sorted(bboxes, key=lambda x: -x[4])
    bboxes_after_nmn = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes if bb_intersection_over_union(chosen_box, box) < iou_threshold]
        bboxes_after_nmn.append(chosen_box)

    return bboxes_after_nmn

def predict():
    pass


config_file = '/Users/gatilin/PycharmProjects/detection-webui/webui/det/mmyolo/configs/yolov5/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco.py'
checkpoint_file = '/Users/gatilin/PycharmProjects/detection-webui/webui/det/mmyolo/weights/yolov5_ins_n-v61_syncbn_fast_8xb16-300e_coco_instance_20230424_104807-84cc9240.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
result = inference_detector(model, '/Users/gatilin/PycharmProjects/detection-webui/images/demo.jpg')
# result = inference_detector(model, '/Users/gatilin/PycharmProjects/detection-webui/images/bus.jpg')
print("result", result)

pred_instances = result.pred_instances
print("pred_instances", pred_instances)
score = np.array(pred_instances.scores)
label = np.array(pred_instances.labels)
bbox = np.array(pred_instances.bboxes)
# print("score", score, "label", label, "bbox", bbox)
print("score:", len(score), "label:", len(label), "bbox:", len(bbox))

# 置信度阈值
CONF_THRES = 0.3
pred_instance = result.pred_instances.cpu().numpy()
# bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
# print("bboxes1:", bboxes)
bboxes = bbox
bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > CONF_THRES)]
bboxes = bboxes[nms(bboxes, 0.3)][:, :4].astype('int')
print('bboxes', bboxes)
# return info: {'status' : ['EMPTY', 'FULL', 'AWAY', …] #seat}


'''  
import cv2
img = cv2.imread(imgs[0])
colors = [(0,0,0),(0,255,0),(0,255,255),(0,0,255)]
for box in bbox[0]:
    xmin, ymin, xmax, ymax, _, cls = map(int, box)
    cv2.rectangle(img, (xmin, ymin),(xmax,ymax),colors[cls+1])
pdb.set_trace()
cv2.imwrite('check.png',img)
'''













# 取出前100个最大的score对应的bbox和label
# idxs = np.argsort(score)[::-1][:100]
# print("idxs", idxs)
#
# bbox = bbox[idxs]
# label = label[idxs]

# img = cv2.imread('/Users/gatilin/PycharmProjects/detection-webui/images/demo.jpg')
#
# # 绘制bbox
# for i in range(len(bbox)):
#     x1, y1, x2, y2 = bbox[i]
#     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#     cv2.putText(img, str(label[i]), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
# # 显示结果
# cv2.imshow('result', img)
# cv2.waitKey(0)

# import cv2
# import numpy as np
#
# # 从结果中提取数据
# img_path = "/Users/gatilin/PycharmProjects/detection-webui/images/demo.jpg"
# bboxes = np.array([[613.4585, 100.8939, 622.7244, 114.1862],
#                    [265.4710, 112.8605, 274.7048, 126.2006],
#                    [241.4698, 108.8725, 250.7095, 122.2139],
#                    ...,
#                    [165.4496, 96.8771, 174.7426, 110.1879],
#                    [569.4405, 100.8696, 578.7538, 114.2099],
#                    [249.4495, 108.8866, 258.7365, 122.1980]])
# labels = np.array([32, 32, 32, ..., 32, 32, 32])
# scores = np.array([0.2649, 0.2648, 0.2648, ..., 0.2642, 0.2642, 0.2642])
#
# # 读取图像
# image = cv2.imread(img_path)
#
# # 循环遍历所有 bbox
# for i, bbox in enumerate(bboxes):
#     # 提取 bbox 坐标
#     x1, y1, x2, y2 = bbox.astype(int)
#
#     # 绘制矩形
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#     # 绘制文本
#     label = f"{labels[i]}: {scores[i]:.2f}"
#     cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
# # 显示图像
# cv2.imshow("Output", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()