from absl import logging, flags, app
import random
import os
import torch 
import pytorch_lightning as pl
import numpy as np
# load defined functions by authors
from src import dataset, models
from two_sample_distance import pdist
import random
from pathlib import Path
import torch
import logging
logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS

class Model(pl.LightningModule):
        def __init__(self, train_ds=None, val_ds=None, test_ds=None, num_classes=None, sampling_rate_audio=None):
            super().__init__()

            #set parameters
            self.num_classes = num_classes
            self.sampling_rate_audio = sampling_rate_audio

            #set dataset
            self.train_ds = train_ds
            self.val_ds = val_ds
            self.test_ds = test_ds

            self._test_step_outputs = []
            self.input_shape = self.train_ds[0][0].shape # first batch of EEG data #seq_len_input, num_channels
            self.mel_transformer=self.train_ds.tacotron_mel_transformer
            
            if FLAGS.use_MFCCs:
                self.output_shape=self.mel_transformer.mel_spectrogram(self.train_ds[0][1].unsqueeze(0)).squeeze(0).T.shape #seq_len x mel bins ??
                self.criterion = torch.nn.MSELoss(reduction='none')

            self.seq2seq = models.RNNSeq2Seq(self.input_shape, self.output_shape)

        def loss(self, logits, y):
            if not FLAGS.use_MFCCs:
                y = y.flatten() # [bs x seq_len] -> [bs*seq_len] @YK: WHY??
                logits = logits.flatten(0, 1) #[bs x seq_len x self.num_classes] -> [bs*seq_len x self.num_classes]
            return self.criterion(logits, y) #[bs * seq_len]

        def logits_to_classes(self,logits):
            current_batch_size=logits.shape[0] #will be flags.batchsize almost always but not in the very end of validation/train
            logits=logits.reshape((current_batch_size,-1,80,FLAGS.num_mel_centroids)) # bs x seq_len x 80 x 12
            logits=torch.argmax(logits,dim=3) #bs x seq_len x 80. each entry is the class.
            return logits

        def logits_to_mel_centroids(self,logits):
            logits=self.logits_to_classes(logits)
            return self.mel_spec_discretizer.class_to_centroids(logits)

        def contrastive_loss(self, x, encoder_outputs):
            #encoder_outputs [bs x seq_len_after_conv x hidden_size * directions] 
            #x [bs x seq_len_before_conv x channels]

            patient_4=(encoder_outputs[x[:,0,214]==0,:,:])
            patient_8=(encoder_outputs[x[:,0,214]==1,:,:])

            if len(patient_4<FLAGS.batch_size) and len(patient_8<FLAGS.batch_size):
                distance_within_patient_4=torch.mean(torch.nn.functional.pdist(patient_4.flatten(1,2))) #the flatten is not nice but very fast
                distance_within_patient_8=torch.mean(torch.nn.functional.pdist(patient_8.flatten(1,2)))
                distance_across_patients=torch.mean(pdist(patient_4.flatten(1,2),patient_8.flatten(1,2)))  #(n_1, d),(n_2,d)
                delta=1
                return (distance_within_patient_4+distance_within_patient_8+distance_across_patients)**2
            else:
                return 0

        def accuracy(self, logits, y, topk=1):
            if FLAGS.use_MFCCs: # y and logits are [bs x num_frames x num_bins]
                pearson_r=torch.nn.functional.cosine_similarity(y-torch.mean(y,dim=1).unsqueeze(1),logits-torch.mean(logits,dim=1).unsqueeze(1),dim=1)   
                return pearson_r
            else:
                _, topi = torch.topk(logits, k=topk, dim=-1)
                return (y.unsqueeze(-1) == topi).float().sum(-1)

        def forward(self, x):
            pass
        
        def training_step(self, batch, batch_idx):
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            x, y = batch
            if FLAGS.use_MFCCs:
                y=self.mel_transformer.mel_spectrogram(y).transpose(1,2)
            teacher_forcing = torch.bernoulli(x.new_ones((x.shape[0],)) * FLAGS.teacher_forcing_ratio).to(dtype=torch.bool)
           
            logits,attn_matrix, encoder_outputs = self.seq2seq(x, y=y, teacher_forcing=teacher_forcing)
            loss = self.loss(logits, y)
            
            acc = self.accuracy(logits, y)

            if not FLAGS.use_MFCCs:
                acc5 = self.accuracy(logits, y, 5)

            if FLAGS.use_MFCCs: 
                #for param_group in self.trainer.optimizers[0].param_groups:
                    #current_lr=(param_group['lr'])
                
                logs = {'loss/train': loss.mean(), 'pearson_r/train': acc.mean(), 'learning_rate': current_lr}
                if batch_idx == 0:
                    MFCC_plot = dataset.create_MFCC_plot(logits[0],y[0]) #passt first sample
                    self.logger.experiment.add_image('MFCC_plot/train', MFCC_plot, global_step=self.global_step, dataformats='HWC')
                    attn_plot = models.create_attention_plot(attn_matrix)
                    self.logger.experiment.add_image('Attention_Matrix', attn_plot, global_step=self.global_step, dataformats='HWC')


            else:
                logs = {'loss/train': loss.mean(), 'acc/acc': acc.mean(), 'acc5': acc5.mean()}
                if batch_idx == 0:
                    audio_idx = random.randint(0, len(batch) - 1)
                    audio_real = dataset.audio_classes_to_signal_th(y[audio_idx])
                    audio_pred = dataset.audio_classes_to_signal_th(logits[audio_idx].argmax(-1))
                    self.logger.experiment.add_audio(
                            tag='audio_real',
                            snd_tensor=audio_real.unsqueeze(0),
                            global_step=self.global_step,
                            sample_rate=self.sampling_rate_audio,
                            )
                    self.logger.experiment.add_audio(
                            tag='audio_pred',
                            snd_tensor=audio_pred.unsqueeze(0),
                            global_step=self.global_step,
                            sample_rate=self.sampling_rate_audio,
                            )

                    audio_plot = dataset.create_audio_plot([
                            (audio_real, 'real'),
                            (audio_pred, 'predicted_tf_{}'.format(teacher_forcing[audio_idx].item())),
                            ])
                    self.logger.experiment.add_image('audio_plot', audio_plot, global_step=self.global_step, dataformats='HWC')
            #return {**logs, 'log': logs}
            self.log_dict(
                {
                    "loss/train": loss.mean(),
                    "learning_rate": current_lr,
                    # add others you want
                },
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

            return loss.mean()
        def on_validation_epoch_end(self):
            val = self.trainer.callback_metrics.get("val_loss")
            if val is not None:
                logger.info(f"Epoch {self.current_epoch}: val_loss={val.item():.4f}")



        def validation_step(self, batch, batch_idx):

            SAVE_DIR = Path("/content/drive/MyDrive/Advance_python_project/mel_preds_p3")  # <-- change to your folder

            x, y = batch  # y is ground-truth audio waveform at first (before mel conversion)

            # 1) Build target mel (ground-truth) if using MFCCs/mel
            y = self.mel_transformer.mel_spectrogram(y).transpose(1, 2)

            # 2) Predict (no teacher forcing)
            logits, attn_matrix, encoder_outputs = self.seq2seq(x)

            # 3) Loss
            loss = self.loss(logits, y)

            # 4) Metrics
            acc = self.accuracy(logits, y)

            acc5 = acc.new_zeros((1, 1))

            # 5) Build outs + (optionally) save mels for WaveGlow
            if FLAGS.use_MFCCs:
                every_kth = 1  # keep as you had it

                y_mel = y                                                    # [B, T, n_mels]
                pred_mel = logits                                            # [B, T, n_mels]

                outs = {
                    "loss": loss,
                    "acc": acc,
                    "acc5": acc5,
                    "targets": y_mel[:, ::every_kth, :],
                    "predictions": pred_mel[:, ::every_kth, :],
                }

                # -------- SAVE ONE SAMPLE PER EPOCH (WaveGlow-friendly) --------
                # Only save from rank 0 in DDP (safe for single GPU too).
                if batch_idx == 0 and getattr(self.trainer, "is_global_zero", True):
                    SAVE_DIR.mkdir(parents=True, exist_ok=True)

                    idx = 0  # save first item in batch (change if you want random)

                    gt = y_mel[idx].detach().cpu().float()        # [T, n_mels]
                    pr = pred_mel[idx].detach().cpu().float()     # [T, n_mels]

                    # WaveGlow expects [B, n_mels, T]
                    gt_wg = gt.transpose(0, 1).unsqueeze(0)       # [1, n_mels, T]
                    pr_wg = pr.transpose(0, 1).unsqueeze(0)       # [1, n_mels, T]

                    out_path = SAVE_DIR / f"epoch{self.current_epoch:03d}_val_mel.pt"
                    torch.save(
                        {
                            "gt_mel": gt_wg,
                            "pred_mel": pr_wg,
                            "epoch": int(self.current_epoch),
                            "batch_idx": int(batch_idx),
                        },
                        out_path,
                    )
                    logger.info(f"[validation_step] Saved mel tensors for WaveGlow -> {out_path}")
                # ---------------------------------------------------------------

                # Optional plotting (kept from your original)
                # if batch_idx == 0:
                #     outs["MFCC_plot_val"] = dataset.create_MFCC_plot(pred_mel[0], y_mel[0])

            else:
                # No MFCCs branch from your original (audio classes)
                outs = {"loss": loss, "acc": acc, "acc5": acc5}

                if batch_idx == 0:
                    audio_idx = random.randint(0, len(batch) - 1)
                    outs["val_audio_real"] = dataset.audio_classes_to_signal_th(y[audio_idx])
                    outs["val_audio_pred"] = dataset.audio_classes_to_signal_th(logits[audio_idx].argmax(-1))
                    outs["val_audio_plot"] = dataset.create_audio_plot(
                        [(outs["val_audio_real"], "real"), (outs["val_audio_pred"], "predicted")]
                    )

            self.log("val_loss", loss.mean(), on_epoch=True, prog_bar=True)
            return loss.mean()
    
        def test_step(self, batch, batch_idx, dataloader_idx):

            x, y = batch
            if FLAGS.use_MFCCs:
                y=self.mel_transformer.mel_spectrogram(y).transpose(1,2)

            logits, attn_matrix, encoder_outputs = self.seq2seq(x)  # no teacher forcing
            preds = logits.argmax(-1)
            return {
                    'audio_real': dataset.audio_classes_to_signal_th(y).flatten(), 
                    'audio_pred': dataset.audio_classes_to_signal_th(preds).flatten()}

        def on_test_epoch_end(self):
            if not self._test_step_outputs:
                return

            # If your test_step returns per-dataloader lists, you may need to adapt this.
            audio_real = torch.cat([o["audio_real"] for o in self._test_step_outputs], 0).cpu().numpy()
            audio_pred = torch.cat([o["audio_pred"] for o in self._test_step_outputs], 0).cpu().numpy()

            os.makedirs("results", exist_ok=True)
            np.save(f"results/test_bs_{FLAGS.batch_size}", np.concatenate((audio_real, audio_pred)))

            self._test_step_outputs.clear()


        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                    self.train_ds, 
                    shuffle=True,
                    drop_last=True,
                    num_workers=3,
                    batch_size=FLAGS.batch_size)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(
                    self.val_ds, 
                    shuffle=False,
                    drop_last=False,
                    num_workers=3,
                    batch_size=FLAGS.batch_size)

        def test_dataloader(self):
            train_ds_full_hop = dataset.get_data('train')
            return [torch.utils.data.DataLoader(ds, shuffle=False, drop_last=False, num_workers=3,
                    batch_size=FLAGS.batch_size) for ds in (train_ds_full_hop, self.test_ds)]

        def configure_optimizers(self): 
            optim = next(o for o in dir(torch.optim) if o.lower() == FLAGS.optim.lower()) # "Adam"
            optimizer=getattr(torch.optim, optim)(self.parameters(), lr=FLAGS.learning_rate) # optimizer object
            optimizer=torch.optim.AdamW(self.parameters(),lr=FLAGS.learning_rate,weight_decay=0.001)
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=FLAGS.lr_milestones, gamma=0.5)
            return [optimizer], [scheduler]