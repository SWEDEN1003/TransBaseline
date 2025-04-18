python train.py \
--config_file configs/Market/vit_transreid.yml \
MODEL.DEVICE_ID "('0')" \
DATASETS.ROOT_DIR "('/home/weidong.shi1/data/')" \
OUTPUT_DIR "('./results/market/baseline')"


# 2025-04-17 03:34:43,238 transreid.train INFO: Validation Results - Epoch: 120
# 2025-04-17 03:34:43,239 transreid.train INFO: mAP: 88.0%
# 2025-04-17 03:34:43,239 transreid.train INFO: CMC curve, Rank-1  :94.8%
# 2025-04-17 03:34:43,239 transreid.train INFO: CMC curve, Rank-5  :98.4%
# 2025-04-17 03:34:43,239 transreid.train INFO: CMC curve, Rank-10 :99.1%
