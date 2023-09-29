'''
pip install mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/${CUDA}/${Torch}/index.html
pip3 install torch_blade_cpu==3.27.0+1.6.0 -f https://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/release/repo.html
pip install pai-easycv
'''
import cv2
from easycv.predictors.classifier import TorchClassifier

output_ckpt = 'work_dirs/classification/cifar10/swintiny/epoch_350_export.pth'
tcls = TorchClassifier(output_ckpt)

img = cv2.imread('aeroplane_s_000004.png')
# input image should be RGB order
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
output = tcls.predict([img])
print(output)