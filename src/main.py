#!/usr/bin/env python3

from absl import logging, flags, app
import sh
import time
import os
import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import tomllib

import dataset
from pipeline import Model


flags.DEFINE_string(
    'config',
    None,
    'Path to TOML config. Values override DEFAULT flags unless the flag is set on the command line.'
)
FLAGS = flags.FLAGS

def _apply_toml_config(config_path: str, FLAGS_obj):
    """Load TOML and apply into absl FLAGS.
    - TOML values override ONLY flags that are still at their default and not present on CLI.
    - CLI flags always win.
    """
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    # allow either top-level keys or [flags] table
    if isinstance(cfg.get("flags"), dict):
        cfg = cfg["flags"]

    if not isinstance(cfg, dict):
        raise ValueError("TOML config must be a table, or contain a [flags] table.")

    for key, value in cfg.items():
        if key not in FLAGS_obj:
            logging.warning("Config key '%s' is not a known flag; skipping.", key)
            continue

        flag = FLAGS_obj[key]

        # if FLAG already has been registerd on CLI command, skip
        if flag.present:
            continue

        # task: check meaning
        if not flag.using_default_value:
            continue

        try:
            setattr(FLAGS_obj, key, value) #set value from toml in FLAG
        except Exception as e:
            raise ValueError(f"Failed to set flag '{key}' from config value {value!r}: {e}") from e


def main(_):

    # --- add these lines at the very top of main() ---
    if FLAGS.config:
        _apply_toml_config(FLAGS.config, FLAGS)
        logging.info("Loaded config from %s", FLAGS.config)
    
    if FLAGS.clean_logs_dir:
        sh.rm('-r', '-f', 'logs')
        sh.mkdir('logs')
    if not torch.cuda.is_available():
        FLAGS.gpus = 0
        torch.Tensor.cuda = lambda self, *args, **kwargs: self

    if FLAGS.gpus:
        time.sleep(5)

    if not FLAGS.patient_eight:
        FLAGS.num_mel_centroids=10


    if FLAGS.OLS or FLAGS.DenseModel: #make sure output length is 1.
        assert FLAGS.use_MFCCs==True, "OLS can so far only be used with MFCCs"
        FLAGS.window_size=50
        print("Running with OLS/Dense. Re-setting window_size to 50ms")

    train_ds = dataset.get_data(split='train', hop=FLAGS.hop_in_ms)
    val_ds   = dataset.get_data(split='val',   hop=FLAGS.hop_in_ms)
    test_ds = dataset.get_data(split='test',  hop=FLAGS.hop_in_ms)
    logging.info(f'train size: {len(train_ds)}, test size: {len(test_ds)}')
    num_classes = train_ds.num_audio_classes
    sampling_rate_audio = round(FLAGS.sampling_rate_eeg * train_ds.audio_eeg_sample_ratio)

    if not FLAGS.use_MFCCs:
        class_freqs = torch.histc(train_ds.audio.float(), bins=num_classes, min=0, max=num_classes-1)
        class_freqs += FLAGS.laplace_smoothing * num_classes
        class_weights = 1. / class_freqs
        class_weights /= class_weights.sum()
        val_acc_for_mode = (train_ds.audio == test_ds.audio.mode()[0].item()).float().mean()
        logging.info(f'Validation accuracy when predicting mode: {val_acc_for_mode}')

    model = Model(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, num_classes=num_classes, sampling_rate_audio=sampling_rate_audio)

    # use gpus if they are available
    if FLAGS.gpus and torch.cuda.is_available():
        accelerator = "gpu"
        devices = FLAGS.gpus
    else:
        accelerator = "cpu"
        devices = 1


    os.makedirs("/content/drive/MyDrive/Advance_python_project/checkpoints_p2_1d", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath= "/content/drive/MyDrive/Advance_python_project/checkpoints_p2_1d",
        monitor="val_loss",
        filename="model-{epoch:02d}-{val_loss:.4f}",
        save_top_k =1,
        mode= "min",
        )

    run_name = getattr(FLAGS, "run_name", None)

    if (run_name is None) or (run_name is Ellipsis):
        run_name = "default_run"

    callbacks = []
    if FLAGS.SWA:
        callbacks.append(
            pl.callbacks.StochasticWeightAveraging(
                swa_lrs=FLAGS.learning_rate / 10,
                swa_epoch_start=max(0, FLAGS.epochs - 2),
                )
            )
    callbacks.append(checkpoint_callback)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        default_root_dir="logs",
        logger=pl.loggers.TensorBoardLogger("logs", name=str(run_name)),
        detect_anomaly=True,          # replacement for terminate_on_nan
        log_every_n_steps=10,        # replacement for row_log_interval
        num_sanity_val_steps=8,       # replacement for nb_sanity_val_steps
        gradient_clip_val=2,
        callbacks=callbacks,
        enable_progress_bar=False,
    )

    print("Starting training")
    trainer.fit(model)

    if FLAGS.final_eval:
        trainer.test(model)


if __name__ == '__main__':
    app.run(main)
