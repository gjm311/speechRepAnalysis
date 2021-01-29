import numpy as np
import torch
import sys
import json

sys.path.append("/../../../")
from AEspeech import AEspeech 

with open("config.json") as f:
    data = f.read()
config = json.loads(data)
import pdb

unit=config['general']['UNITS']
TIME_STEPS=config['mel_spec']['TIME_STEPS']
TIME_SHIFT=config['mel_spec']['TIME_SHIFT']
FRAME_SIZE=config['mel_spec']['FRAME_SIZE']
FS=config['general']['FS']

def reconMelSpecTransform(wav_file,rep):
    #compute 'reconstructed (recon)' mel-spectrogram using learned auto-encoder params
    aespeech=AEspeech(model='CAE',units=unit,rep=rep)
    #mat,sig_len=aespeech.compute_spectrograms(wav_file, plosives_only=0,volta=1)
    #mat=mat.float()
    #to,bot=aespeech.AE.forward(mat)
    #to=to.float()
    
    if torch.cuda.is_available():
        mat=mat.cuda()
    
    to,bot=aespeech.AE.forward(aespeech.compute_spectrograms(wav_file, plosives_only=0,volta=0).float())
    to=to.float()
    if torch.cuda.is_available():
        to=to.cuda()
    new_to=torch.zeros((to.shape[2],to.shape[0]*to.shape[3]))
    init=0
    endi=int(TIME_STEPS)
    shift=int(TIME_STEPS*(TIME_SHIFT/FRAME_SIZE))
    for fr in range(to.shape[0]):
        new_to[:,init:endi]=to[fr,:,:,:]
        init+=shift
        endi+=shift
    return new_to
