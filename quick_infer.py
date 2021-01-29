import sys
import os
import torch
import pywt
import pandas as pd
import scaleogram as scg 
from scipy.io.wavfile import read
import scipy
import librosa
from librosa.feature import melspectrogram
import torchaudio as T
import torchaudio.transforms as TT
import numpy as np
import numpy.fft
import cv2
import json
from AEspeech import AEspeech 
import argparse
import torchaudio
from scipy.signal import butter, lfilter
import pdb

from diffwave.inference import predict as diffwave_predict
from librosa.feature import melspectrogram

import toolbox.traintestsplit as tts



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



if __name__ == "__main__":
    
    if len(sys.argv)!=4:
        print("python quick_infer.py <nb-(1) or bb-(0)> <ori or recon> <path_audio>")
        sys.exit()
#    "./tedx_spanish_corpus/speech/train/"
    if sys.argv[3][0] != '/':
        sys.argv[3] = '/'+sys.argv[3]
        
    if sys.argv[3][-1] != "/":
        sys.argv[3] = sys.argv[3]+'/'
        
    if sys.argv[2] not in ['ori','recon']:
        print("python quick_infer.py <nb-(1) or bb-(0)> <ori or recon> <path_audio>")
        sys.exit()
    else:
        ori=sys.argv[2]

    if int(sys.argv[1]) not in [0,1]:
        print("python quick_infer.py <nb-(1) or bb-(0)> <ori or recon> <path_audio>")
        sys.exit()
    else:
        nb=int(sys.argv[1])
    
    with open("config.json") as f:
        data = f.read()
    config = json.loads(data)
    
    FS=config['general']['FS']
    NFFT=config['mel_spec']['NFFT']
    units=config['general']['UNITS']
    TIME_STEPS=config['mel_spec']['FULL_TIME_STEPS']
    TIME_SHIFT=config['mel_spec']['TIME_SHIFT']
    FRAME_SIZE=config['mel_spec']['FRAME_SIZE']
    
    #binary narrowband: 1 yes, 0 no (i.e. broadband)
    if nb==0:
        #broadband: higher time resolution, less frequency resolution
        NMELS=config['mel_spec']['BB_NMELS']
        HOP=int(FS*config['mel_spec']['BB_HOP'])#3ms hop (48 SAMPLES)
        WIN_LEN=int(FS*config['mel_spec']['BB_TIME_WINDOW'])#5ms time window (60 SAMPLES)
        min_filter=50
        max_filter=7000
        rep='broadband'
    elif nb==1:
        #narrowband: higher frequency resolution, less time resolution
        NMELS=config['mel_spec']['NB_NMELS']
        HOP=int(FS*config['mel_spec']['NB_HOP']) #10ms hop (160 SAMPLES)
        WIN_LEN=int(FS*config['mel_spec']['NB_TIME_WINDOW']) #30ms time window (480 SAMPLES)
        min_filter=300
        max_filter=5400
        rep='narrowband'
        
    INTERP_NMELS=config['mel_spec']['INTERP_NMELS']
    PATH=os.path.dirname(os.path.abspath(__file__)) 
    path_audio=PATH+sys.argv[3]
    save_path=PATH+"/diff_recon/"+rep+"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_dir=PATH+"/diff_models/"+rep+"/" #'/path/to/model/dir'
    model_files=os.listdir(model_dir)
    model_files.sort()
    if ori=='ori':
        model_dir=model_dir+model_files[0]
    else:
        model_dir=model_dir+model_files[1]
    
    itr=0
    for iter in range(1):
        
        while '.npy' in os.listdir(path_audio)[itr]:
            itr+=1
            
        wav_file=path_audio+os.listdir(path_audio)[itr]
                
        if ori=='ori':
            audio, sr = T.load_wav(wav_file)
            audio = torch.clamp(audio[0] / 32767.5, -1.0, 1.0)

            mel_args = {
              'sample_rate': FS,
              'win_length': WIN_LEN,
              'hop_length': HOP,
              'n_fft': NFFT,
              'f_min': 20.0,
              'f_max': FS / 2.0,
              'n_mels': NMELS,
              'power': 1.0,
              'normalized': True,
          }
            mel_spec_transform = TT.MelSpectrogram(**mel_args)

            with torch.no_grad():
                spectrogram = mel_spec_transform(audio)
                spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
                spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
        else:            
            aespeech=AEspeech(model='CAE',units=units,rep='full_'+rep)
            fs_in, signal=read(wav_file)
            fullmat=melspectrogram(signal.astype(float), sr=FS, n_fft=NFFT, win_length=WIN_LEN, hop_length=HOP, n_mels=NMELS, fmax=FS//2)
            
            fullmat=fullmat[np.where(fullmat[:,0]>0)]
            fullmat=np.abs(fullmat)
            fullmat=np.log(fullmat, dtype=np.float32)
            
            inter_mat=cv2.resize(fullmat,(TIME_STEPS,INTERP_NMELS),interpolation=cv2.INTER_CUBIC)
            ae_mat=torch.zeros(1,1,INTERP_NMELS,TIME_STEPS)
            tmat=torch.from_numpy(inter_mat)
            ae_mat[0,0,:,:]=tmat
            if torch.cuda.is_available():
                ae_mat=ae_mat.cuda()
            to, bot=aespeech.AE.forward(ae_mat)        
            if torch.cuda.is_available():
                to=to.cuda()
            spectrogram=aespeech.destandard(to).squeeze(0).squeeze(0)
        audio, sample_rate = diffwave_predict(spectrogram.float(), model_dir, ori=ori, rep=rep)
        torchaudio.save(save_path+ori+"_"+os.listdir(path_audio)[iter]+".wav", audio.cpu(),sample_rate=FS)
