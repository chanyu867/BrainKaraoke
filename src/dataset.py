#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
from absl import flags, logging
from pathlib import Path
import numpy as np
import torch 
from matplotlib import pyplot as plt
from stft import STFT
from scipy.signal import resample_poly
try:
    import torchaudio.functional as taF
except Exception:
    torchaudio = None
    taF = None
import logging
logger = logging.getLogger(__name__)

# load defined functions by authors
from audio_processing import dynamic_range_compression
from audio_processing import dynamic_range_decompression

FLAGS = flags.FLAGS

def mu_law(x): #for audio conversion from regression to classification
    return np.sign(x)*np.log(1+255*np.abs(x))/np.log(1+255)

def mu_law_inverse(x):
    return np.sign(x)*(1./255)*(np.power(1.+255, np.abs(x)) - 1.)

def audio_signal_to_classes(audio):
    audio=np.floor(128*mu_law(audio))
    audio = np.clip(audio, -128., 128.) / 128.
    audio = (audio + 1.) / 2.
    audio = np.round(audio * (FLAGS.num_audio_classes - 1)).astype(np.int64)
    return audio

def audio_classes_to_signal_th(audio):
    audio = audio.detach().cpu().numpy()
    audio = audio.astype(np.float32) / (FLAGS.num_audio_classes - 1)
    audio = audio * 2. - 1.
    audio = audio * 128.
    audio = mu_law_inverse(audio / 128.)
    audio = torch.from_numpy(audio)
    return audio

class GlobalMelSpecDiscretizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.centroids=torch.load('global_centroids_kh4.pt')

    def mel_to_class(self,melspecs):
        centroids=self.centroids.repeat(melspecs.shape[0]).view(-1,FLAGS.num_mel_centroids).cuda()
        distances=torch.abs((melspecs.unsqueeze(1)-centroids.unsqueeze(-1).unsqueeze(-1))).transpose(1,2)  # bs x seq_len x 12 x mel_bins
        cluster_assignments=torch.argmin(distances,dim=2) #closest enter of each entry # bs x seq_len x mel_bins
        
        return cluster_assignments
    def class_to_centroids(self,cluster_assignments):
        values=torch.zeros_like(cluster_assignments).float()
        for i in range(len(self.centroids)):
            values[cluster_assignments==i]=self.centroids[i].cuda()
        return values

    def mel_to_centroids(self,melspecs): 
        cluster_assignments=self.mel_to_class(melspecs)
        return self.class_to_centroids(cluster_assignments)

class LocalMelSpecDiscretizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.centroids=torch.load("local_centroids_kh4.pt").cuda()

    def mel_to_class(self,melspecs):
        #compute distances
        melspecs=melspecs.transpose(1,2) # from [bs x seq x bins] to [bs x bins x seq]

        distances=torch.abs((melspecs.transpose(0,1).unsqueeze(1)-self.centroids.unsqueeze(-1).unsqueeze(-1)))
        cluster_assignments=torch.argmin(distances,dim=1) #[bins x bs x seq_len]
        
        return cluster_assignments.transpose(0,1).transpose(1,2) # [bs x seq_len x bins]
    def class_to_centroids(self,cluster_assignments):
        cluster_assignments=cluster_assignments.transpose(1,2) # from [bs x seq_len x bins] to [bs x bins x seq]
        cluster_assignments=cluster_assignments.transpose(0,1) # [bins x bs x seq]
        values=torch.zeros_like(cluster_assignments).float() # [bins x bs x seq]
        for i in range(len(self.centroids)):
            for j in range(self.centroids.shape[1]):
                indices_to_replace=(cluster_assignments==j)[i] #bs x seq_len
                values[i,indices_to_replace]=self.centroids[i,j]

        return values.transpose(0,1).transpose(1,2) #[bs x seq_len x bins ]

    def mel_to_centroids(self,melspecs): 
        cluster_assignments=self.mel_to_class(melspecs)
        return self.class_to_centroids(cluster_assignments)

class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,  
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0): 
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length) # hop and window length are in samples.

        n_freqs = filter_length // 2 + 1

        if taF is not None:
            mel_basis = taF.melscale_fbanks(
                n_freqs=n_freqs,
                f_min=mel_fmin,
                f_max=mel_fmax,
                n_mels=n_mel_channels,
                sample_rate=sampling_rate,
                norm=None,
                mel_scale="htk",
            ).T
        else:
            # last-resort: keep your original librosa path if you still want it
            import librosa
            mel_basis = librosa.filters.mel(
                sr=sampling_rate,
                n_fft=filter_length,
                n_mels=n_mel_channels,
                fmin=mel_fmin,
                fmax=mel_fmax,
                htk=True,
                norm=None,
            )
            mel_basis = torch.from_numpy(mel_basis).float()

        self.register_buffer("mel_basis", mel_basis.float())


    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        y_det = y.detach()
        assert torch.min(y_det) >= -1
        assert torch.max(y_det) <= 1

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.detach()
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        if (getattr(FLAGS, "OLS", False) or getattr(FLAGS, "DenseModel", False)):
            return mel_output[:, :, 3].unsqueeze(-1)
        else:
            return mel_output

def create_audio_plot(audios_with_labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for audio, label in audios_with_labels:
        ax.plot(audio.detach().cpu().numpy(), label=label)
    ax.legend(loc='best')
    fig.canvas.draw()

    # RGBA buffer -> numpy array (H, W, 4)
    buf = np.asarray(fig.canvas.buffer_rgba())

    # drop alpha -> RGB (H, W, 3)
    rgb = buf[..., :3].copy()

    return torch.from_numpy(rgb).float() / 255.0


def create_MFCC_plot(MFCCs, targets):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True)
    ax[0].imshow(targets.detach().cpu().numpy(), cmap='viridis',aspect='auto') 
    
    ax[1].imshow(MFCCs.detach().cpu().numpy(), cmap='viridis',aspect='auto') 
    fig.canvas.draw()

    # buffer_rgba() gives an (H, W, 4) uint8 array (RGBA)
    buf = np.asarray(fig.canvas.buffer_rgba())

    # convert to RGB (H, W, 3)
    data = buf[..., :3].copy()

    return torch.from_numpy(data.transpose(1, 0, 2)).float() / 255.0



def get_data(split='train', hop=None):
    data_dir = Path(FLAGS.data_dir) #->google drive contents

    #load audio file
    logger.info(f"Loading audio data from: {data_dir / FLAGS.audio_file}")
    audio=np.load(str(data_dir / FLAGS.audio_file))
        
    #load SEEG data
    logger.info(f"Loading sEEG data from: {data_dir / FLAGS.eeg_file}")
    eeg = np.load(str(data_dir / FLAGS.eeg_file)).T

    if eeg.ndim == 2 and eeg.shape[0] < eeg.shape[1]:
        eeg = eeg.T

    audio_eeg_sample_ratio = audio.shape[0] / eeg.shape[0]
    logger.info(f"Audio to EEG sample ratio: {audio_eeg_sample_ratio}, eeg: {eeg.shape}, audio: {audio.shape}")
    
    if not FLAGS.use_MFCCs:
        audio = audio_signal_to_classes(audio)
    
    num_train_samples = round(len(eeg) * FLAGS.train_test_split)
    num_train_samples_audio = round(len(audio) * FLAGS.train_test_split)

    num_val = round(num_train_samples * FLAGS.train_val_split)
    num_val_audio = round(num_train_samples_audio * FLAGS.train_val_split)

    if split == 'train':
        eeg = eeg[:num_val]
        audio = audio[:num_val_audio]
    elif split == 'val':
        eeg = eeg[num_val:num_train_samples]
        audio = audio[num_val_audio:num_train_samples_audio]
    elif split == 'test':
        eeg = eeg[num_train_samples:]
        audio = audio[num_train_samples_audio:]
    else:
        raise ValueError(f"Unknown split: {split}")

    eeg = torch.from_numpy(eeg).float()

    if FLAGS.use_MFCCs:
        audio = torch.from_numpy(audio).float()
    else:
        audio = torch.from_numpy(audio).long()

    logger.info(f"Created {split} dataset with {len(eeg)} samples.")

    return EEGAudioDataset(eeg, audio, FLAGS.num_audio_classes,audio_eeg_sample_ratio, hop=hop)


class EEGAudioDataset(torch.utils.data.Dataset):
    def __init__(self, eeg, audio, num_audio_classes,audio_eeg_sample_ratio, hop=None):
        super().__init__()

        self.audio_eeg_sample_ratio=audio_eeg_sample_ratio
        self.sampling_rate_audio=FLAGS.sampling_rate_eeg*self.audio_eeg_sample_ratio

        self.eeg = eeg
        self.audio = audio

        self.num_audio_classes = num_audio_classes #only meaningful if direct audio is synthesized

        window_size_eeg=FLAGS.window_size / 1000 * FLAGS.sampling_rate_eeg #-> 200/1000 * 1024 = 200
        self.window_size_eeg = round(window_size_eeg)
        self.versatz_eeg=round(window_size_eeg*FLAGS.versatz_windows)
        self.window_size_audio=round(window_size_eeg * self.audio_eeg_sample_ratio)

        self.hop =  self.window_size_eeg if hop is None else int(hop/1000*FLAGS.sampling_rate_eeg)
        self.tacotron_mel_transformer=TacotronSTFT() #all default values are used, i.e. ~ 50ms window size, 12.5 ms hop, 80 mel bins, 8000hz max frequency.
        logging.info(f'''
        num_audio_classes: {self.num_audio_classes},
        window_size_eeg: {self.window_size_eeg}, 
        versatz_eeg: {self.versatz_eeg}, 
        window_size_audio: {self.window_size_audio},
        hop: {self.hop}
        ''')

    def __len__(self):
        # maximum allowed start index for idx_eeg (in EEG samples)
        max_start = len(self.eeg) - (2 * self.versatz_eeg + self.window_size_eeg)

        if max_start < 0:
            return 0

        hop = max(1, int(self.hop))

        return (max_start // hop) + 1

    def __getitem__(self, idx):
        idx_eeg = idx*self.hop+self.versatz_eeg
        idx_audio = round(idx_eeg*self.audio_eeg_sample_ratio) #(includes versatz in audio)
        eeg = self.eeg[idx_eeg-self.versatz_eeg:idx_eeg+self.window_size_eeg+self.versatz_eeg]
        start = idx_audio
        end = idx_audio + self.window_size_audio

        if start < 0 or end > self.audio.shape[0]:
            raise IndexError(
                f"Audio slice out of bounds: [{start}:{end}] of {self.audio.shape[0]}. "
                f"(idx={idx}, idx_eeg={idx_eeg}, ratio={self.audio_eeg_sample_ratio:.4f}, "
                f"window_size_audio={self.window_size_audio})"
            )

        audio = self.audio[start:end]

        return eeg, audio

