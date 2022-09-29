
export CUDA_VISIBLE_DEVICES=2,3

export DATA_DIR=/mnt/e/2022/nerf-library/SulliKovaWedding2022/1b-elenastephen-detailed
export CHECKPOINT_DIR=$DATA_DIR/checkpoints/mipnerf360+jperf

mkdir -p $CHECKPOINT_DIR

python -m train \
  --gin_configs=scripts/BATCH_1/1_ek1_360_glo4.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr
