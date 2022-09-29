
export CUDA_VISIBLE_DEVICES=1

SCENE=
EXPERIMENT=llff
DATA_DIR=/mnt/e/2022/nerf-library/SulliKovaWedding2022/1b-elenastephen-detailed
CHECKPOINT_DIR=$DATA_DIR/checkpoints/mipnerf360

mkdir -p $CHECKPOINT_DIR
rm -rf "$CHECKPOINT_DIR"/*
python -m train \
  --gin_configs=configs/360_glo4.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr
