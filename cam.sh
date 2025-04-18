# export PYTHONPATH=$PYTHONPATH:$PWD
python cam.py \
--config_file configs/Market/vit_transreid.yml \
DATASETS.ROOT_DIR "('/home/weidong.shi1/data/')" \
MODEL.DEVICE_ID "('0')" \
OUTPUT_DIR "('./results/market/cam')" \
TEST.WEIGHT "('results/market/baseline/transformer_120.pth')"
