
export CUDA_VISIBLE_DEVICES=0

DATA_DIR=/mnt/e/2022/nerf-library/SulliKovaWedding2022/car-modelA-clean
CHECKPOINT_DIR=${DATA_DIR}/checkpoints/refnerf

mkdir -p $CHECKPOINT_DIR
# rm -rf "$CHECKPOINT_DIR"/*
python -m train \
  --gin_configs=configs/ngp_refnerf.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr
