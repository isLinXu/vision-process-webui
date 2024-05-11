# sh
set -e
pip install -r requirements.txt
cd mmcv-2.0.0
pip install -r requirements/optional.txt
MMCV_WITH_OPS=1 pip install -e .