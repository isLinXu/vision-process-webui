import os
# os.system("pip install modelscope")
# os.system("pip install tensorflow")

import cv2
from modelscope.pipelines import pipeline

# portrait_matting = pipeline('portrait-matting')
# result = portrait_matting('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_matting.png')
# cv2.imwrite('result.png', result['output_img'])

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

general_recognition = pipeline(
            task=Tasks.general_recognition,
            model='damo/cv_resnest101_general_recognition',
            device='cpu')
result = general_recognition('https://pailitao-image-recog.oss-cn-zhangjiakou.aliyuncs.com/mufan/img_data/maas_test_data/dog.png')