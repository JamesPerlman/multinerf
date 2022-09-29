export CUDA_VISIBLE_DEVICES=0

export DATA_DIR=/mnt/e/2022/nerf-library/SulliKovaWedding2022/1b-elenastephen-detailed
#export DATA_DIR=/mnt/e/2022/nerf-library/SulliKovaWedding2022/1d-gabriela-kisa-kiss
export CHECKPOINT_DIR=$DATA_DIR/checkpoints/mipnerf360+jperf

python -m render \
  --gin_configs="${CHECKPOINT_DIR}/config.gin" \
  --gin_bindings="Config.near = 1" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --gin_bindings="Config.render_dir = '${DATA_DIR}/render'" \
  --gin_bindings="Config.render_path = True" \
  --logtostderr