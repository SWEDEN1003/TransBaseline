
savepath=./results/market/ranklist
rm -rf ${savepath}
python ranklist.py \
--config_file configs/Market/vit_transreid.yml \
DATASETS.ROOT_DIR "('/home/weidong.shi1/data/')" \
MODEL.DEVICE_ID "('0')" \
OUTPUT_DIR "('${savepath}')" \
TEST.WEIGHT "('results/market/baseline/transformer_120.pth')"
echo Finsh Excute Feature!!!
python demo.py --output ${savepath} --input ${savepath}/pytorch_result.mat --nums 100
echo Finsh generate Ranklist!!!
