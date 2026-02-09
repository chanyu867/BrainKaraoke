#!/bin/bash
counter=6
gpu=6
waittime=1
bs=521
tfr=0.1
epochs=110
swas=140
ws=333
lr=0.001
hs=333
dro=0.1
pnpndim=256

python3 -m src.eeg_main \
    --swa_start $swas \
    --dropout $dro \
    --use_bahdanau_attention=True \
    --use_MFCCs=True \
    --hidden_size $hs --batch_size $bs \
    --pre_and_postnet=True --pre_and_postnet_dim $pnpndim\
    --epochs $epochs \
    --learning_rate $lr --window_size $ws