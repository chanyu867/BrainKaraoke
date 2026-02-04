python3 -m src.test_perf.test \
  --config ./src/config/config.toml \
  --ckpt_path ./checkpoints/model-epoch=35-val_loss=2.0383.ckpt \
  --out_dir ./test_infer_out \
  --num_workers 0