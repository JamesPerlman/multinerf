export CUDA_VISIBLE_DEVICES=3
export DATA_DIR=/mnt/e/2022/nerf-library/food/1b-pcporkbun
python -m train \
  --gin_configs=configs/360_glo4.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/mipnerf360'" \
  --logtostderr
