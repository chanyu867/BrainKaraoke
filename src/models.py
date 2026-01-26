#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')

from absl import logging,flags
import torch 
import numpy as np
import math
import torch.nn.functional as F
from matplotlib import pyplot as plt

# Flag definitions
flags.DEFINE_integer('hidden_size', 128, '')
flags.DEFINE_integer('n_layers', 3, '')
flags.DEFINE_integer('n_layers_decoder', 1, '')
flags.DEFINE_float('dropout', 0.5, '')
flags.DEFINE_integer('n_pos', 32, '')

flags.DEFINE_bool('concat_pos', True, 'If True, concatenate positional encoding; else do not concat.')
flags.DEFINE_bool('use_bahdanau_attention', True, '')
flags.DEFINE_bool('convolve_eeg_1d', False, '')
flags.DEFINE_bool('convolve_eeg_2d', False, '')
flags.DEFINE_bool('convolve_eeg_3d', False, '')
flags.DEFINE_bool('pre_and_postnet', True, '')
flags.DEFINE_integer('pre_and_postnet_dim', 256, '')

FLAGS = flags.FLAGS

# --- FIX: Updated for Matplotlib 3.8+ ---
def create_attention_plot(attention_matrix):
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax.imshow(attention_matrix.detach().cpu().numpy(), cmap='viridis',aspect='auto') 
    fig.canvas.draw()
    
    # NEW FIX: Use buffer_rgba and remove alpha channel
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    data = data.reshape((h, w, 4))
    data = data[:, :, :3] # Keep RGB
    
    return torch.from_numpy(data).float() / 255
# ----------------------------------------

class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_length, dim):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=FLAGS.dropout)

        pe = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)) 
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        self.register_buffer('pe', pe) 

    def forward(self, x):
        pos = self.pe[:x.shape[1]]
        pos = torch.stack([pos]*x.shape[0], 0) 
        if getattr(FLAGS, 'concat_pos', True):
            x = torch.cat((x, pos), -1)
        else:
            # no-concat mode (checkpoint-compatible)
            if x.shape[-1] == pos.shape[-1]:
                x = x + pos
            # else: do nothing

        return self.dropout(x)

class LocationLayer(torch.nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = self.location_dense(processed_attention.transpose(1, 2))
        return processed_attention

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class Hybdrid_Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_layer = LinearNorm(FLAGS.hidden_size, FLAGS.hidden_size,bias=False,w_init_gain='tanh')
        self.v = LinearNorm(FLAGS.hidden_size, 1, bias=False) 
        self.memory_layer = LinearNorm(FLAGS.hidden_size * 2, FLAGS.hidden_size,bias=False,w_init_gain='tanh') 
        self.location_layer = LocationLayer(32,31,FLAGS.hidden_size) 
  
    def compute_kv(self,encoder_outputs):
        return self.memory_layer(encoder_outputs)  

    def compute_context(self,inp_emb,hiddens,attn_kv,attention_weights_cat):
        attn_q= self.query_layer(hiddens[-1]).unsqueeze(1) 
        attn_locs=self.location_layer(attention_weights_cat) 
        attn_scores=torch.softmax(self.v(torch.tanh(attn_q+attn_kv+attn_locs)), dim=1).squeeze(-1) 
        return torch.bmm(attn_scores.unsqueeze(1), attn_kv), attn_scores.unsqueeze(1) 

class Yannic_Attention(torch.nn.Module):
    def __init__(self,embedding_size):
        super().__init__()

        self.attention_q = torch.nn.Linear(FLAGS.hidden_size * FLAGS.n_layers_decoder + embedding_size + FLAGS.n_pos, FLAGS.hidden_size)
        self.attention_kv = torch.nn.Linear(FLAGS.hidden_size * 2, FLAGS.hidden_size) 

    def compute_kv(self,encoder_outputs):
        return self.attention_kv(encoder_outputs)  

    def compute_context(self,inp_emb,hiddens,attn_kv):
        attn_q = self.attention_q(torch.cat((inp_emb, hiddens.transpose(1, 0).flatten(1).unsqueeze(1)), -1))
        attn_scores = F.softmax(torch.bmm(attn_q, attn_kv.transpose(2, 1)), -1) 
        return torch.bmm(attn_scores, attn_kv),attn_scores 

class Prenet(torch.nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = torch.nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.elu(linear(x)), p=FLAGS.dropout, training=True)
        return x


class Postnet(torch.nn.Module):
    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = torch.nn.ModuleList()
        self.postnet_embedding_dim=FLAGS.pre_and_postnet_dim 
        self.postnet_kernel_size=5
        self.n_mel_channels=80 
        self.postnet_n_convolutions=5
        self.convolutions.append(
            torch.nn.Sequential(
                ConvNorm(self.n_mel_channels, self.postnet_embedding_dim,
                         kernel_size=self.postnet_kernel_size, stride=1,
                         padding=int((self.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                torch.nn.BatchNorm1d(self.postnet_embedding_dim))
        )

        for i in range(1, self.postnet_n_convolutions - 1):
            self.convolutions.append(
                torch.nn.Sequential(
                    ConvNorm(self.postnet_embedding_dim,
                             self.postnet_embedding_dim,
                             kernel_size=self.postnet_kernel_size, stride=1,
                             padding=int((self.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    torch.nn.BatchNorm1d(self.postnet_embedding_dim))
            )

        self.convolutions.append(
            torch.nn.Sequential(
                ConvNorm(self.postnet_embedding_dim, self.n_mel_channels,
                         kernel_size=self.postnet_kernel_size, stride=1,
                         padding=int((self.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                torch.nn.BatchNorm1d(self.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), FLAGS.dropout, self.training)
        x = F.dropout(self.convolutions[-1](x), FLAGS.dropout, self.training)
        return x          

class RNNSeq2Seq(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.input_length, self.input_channels = input_shape 
        self.output_length, self.num_classes = output_shape 
        self.embedding_size = self.num_classes if FLAGS.use_MFCCs else FLAGS.hidden_size
        self.input_position_encoder = PositionalEncoding(self.input_length, FLAGS.n_pos)
        self.output_position_encoder = PositionalEncoding(self.output_length, FLAGS.n_pos) 

        self.context_for_prediction=True
        self.input_dim=self.input_channels

        self.encoder = torch.nn.GRU( 
                self.input_dim + FLAGS.n_pos, 
                hidden_size=FLAGS.hidden_size, 
                num_layers=FLAGS.n_layers,
                bidirectional=True,
                batch_first=True,
                dropout=FLAGS.dropout
                )

        self.hidden_to_hidden = torch.nn.Linear(FLAGS.hidden_size*FLAGS.n_layers*2, FLAGS.hidden_size*FLAGS.n_layers_decoder)

        if not FLAGS.use_MFCCs:
            self.audio_embedding = torch.nn.Embedding(self.num_classes+1, self.embedding_size) 

        prenet_dim=FLAGS.pre_and_postnet_dim  
        self.prenet=Prenet(self.num_classes,[prenet_dim, prenet_dim]) 

        if FLAGS.pre_and_postnet:
            self.embedding_size=prenet_dim
        self.decoder = torch.nn.GRU(
                FLAGS.hidden_size + self.embedding_size + FLAGS.n_pos,
                hidden_size=FLAGS.hidden_size,
                num_layers=FLAGS.n_layers_decoder,
                bidirectional=False,
                batch_first=True
                )

        if self.context_for_prediction:
            self.decoder_classifier = torch.nn.Sequential(LinearNorm(2*FLAGS.hidden_size, FLAGS.hidden_size),torch.nn.ELU(),
            torch.nn.Dropout(FLAGS.dropout),LinearNorm(FLAGS.hidden_size, self.num_classes))
        else:
            self.decoder_classifier = LinearNorm(FLAGS.hidden_size, self.num_classes)

        self.postnet=Postnet()

        if FLAGS.use_bahdanau_attention:
            self.attention=Hybdrid_Attention()
        else:
            self.attention=Yannic_Attention(self.embedding_size)

    def forward(self, x, y=None, teacher_forcing=None):
        assert (y is None) == (teacher_forcing is None)

        # reverse time order of input for the encoder
        x=torch.flip(x,[2])

        encoder_outputs, hiddens = self.encoder(self.input_position_encoder(x)) 

        hiddens = self.hidden_to_hidden(hiddens.transpose(1, 0).flatten(1)).reshape((-1, FLAGS.n_layers_decoder, FLAGS.hidden_size)).transpose(1, 0).contiguous()
        attn_kv=self.attention.compute_kv(encoder_outputs) 

        inp = x.new_zeros((x.shape[0], 1,self.num_classes)) if FLAGS.use_MFCCs else x.new_ones((x.shape[0], 1)).long() * self.num_classes # SOS
        if FLAGS.pre_and_postnet:
            inp = self.prenet(inp)

        logits = []
        attn_matrix=[]
        attn_scores=encoder_outputs.data.new_zeros((encoder_outputs.shape[0],1,encoder_outputs.shape[1]))
        attn_scores_cum=torch.zeros_like(attn_scores)
        for i in range(self.output_length):

            inp_emb = inp if FLAGS.use_MFCCs else self.audio_embedding(inp) 
            inp_emb = self.output_position_encoder(inp_emb) 

            attention_scores_cat = torch.cat((attn_scores,attn_scores_cum), dim=1)
            if FLAGS.use_bahdanau_attention:
                attn_context,attn_scores=self.attention.compute_context(inp_emb,hiddens,attn_kv,attention_scores_cat)
            else:
                attn_context,attn_scores=self.attention.compute_context(inp_emb,hiddens,attn_kv)

            attn_scores_cum+= attn_scores
            attn_matrix.append(attn_scores[0]) 

            decoder_input=torch.cat((inp_emb, attn_context), -1) 
            decoder_outputs, hiddens = self.decoder(decoder_input, hiddens)

            if self.context_for_prediction:
                decoder_outputs=torch.cat([decoder_outputs, attn_context], 2) 
            else:
                decoder_outputs=decoder_outputs 

            l = self.decoder_classifier(decoder_outputs) 
          
            if FLAGS.use_MFCCs and FLAGS.pre_and_postnet:
                mel_outputs_postnet = self.postnet(l.transpose(1,2)).transpose(1,2)
                final_mel_output = l + mel_outputs_postnet
                logits.append(final_mel_output)
            else:
                logits.append(l)

            if FLAGS.use_MFCCs:
                if teacher_forcing is not None:
                    # teacher_forcing is a bool tensor [batch_size]
                    # We need to construct the input for next step
                    # l is [batch_size, 1, mel_bins]
                    # y is [batch_size, seq_len, mel_bins]
                    
                    # Select ground truth where TF is True
                    current_ground_truth = y[:, i, :].unsqueeze(1) # [bs, 1, mel_bins]
                    # Use l where TF is False, current_ground_truth where TF is True
                    # teacher_forcing needs to be broadcastable
                    mask = teacher_forcing.view(-1, 1, 1)
                    inp = torch.where(mask, current_ground_truth, l)
                else:
                    inp = l
            else:
                pass

            if FLAGS.pre_and_postnet:
                inp=self.prenet(inp.float()) 


        attn_matrix=torch.cat(attn_matrix)
        logits = torch.cat(logits, 1)
        return logits,attn_matrix, encoder_outputs

class OLSModel(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_length,self.input_dim = input_shape
        self.output_length, self.num_classes = output_shape
        self.classifier = torch.nn.Linear(self.input_length*self.input_dim, self.num_classes)
    def forward(self, x, y=None, teacher_forcing=None):
        prediction=self.classifier(x.view((x.shape[0],x.shape[1]*x.shape[2]))) 
        return prediction.unsqueeze(1) 

class DensenetModel(torch.nn.Module):
    # Dummy implementation since you are using RNNSeq2Seq usually
    def __init__(self,channels=40,new_channels_per_conv=20):
        super().__init__()
    def forward(self,x,y=None, teacher_forcing=None):
        pass