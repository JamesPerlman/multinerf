
export CUDA_VISIBLE_DEVICES=0

export DATA_DIR=/mnt/e/2022/nerf-library/SulliKovaWedding2022/car-modelA-clean
export CHECKPOINT_DIR=${DATA_DIR}/checkpoints/refnerf-128

mkdir -p $CHECKPOINT_DIR
rm -rf "$CHECKPOINT_DIR"/*
python -m train \
  --gin_configs=scripts/BATCH_1/0_modelA_ref.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr
