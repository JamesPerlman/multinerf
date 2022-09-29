export CUDA_VISIBLE_DEVICES=3
export DATA_DIR=/mnt/e/2022/nerf-library/SulliKovaWedding2022/car-modelA-clean
export CHECKPOINT_DIR=$DATA_DIR/checkpoints/refnerf
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
python -m render \
  --gin_configs="${CHECKPOINT_DIR}/config.gin" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --gin_bindings="Config.render_dir = '${DATA_DIR}/render'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.batch_size = 256" \
  --logtostderr