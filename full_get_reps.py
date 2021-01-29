import sys
import os
import torch
import pywt
import pandas as pd
import scaleogram as scg 
from scipy.io.wavfile import read
import scipy
import numpy as np
import numpy.fft
import cv2
import json
import argparse
from scipy.signal import butter, lfilter
import pdb

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
    
    if len(sys.argv)!=2:
        print("python get_reps.py <path_audios>")
        sys.exit()
#    "./tedx_spanish_corpus/speech/"
    if sys.argv[1][0] != '/':
        sys.argv[1] = '/'+sys.argv[1]
        
    if sys.argv[1][-1] != "/":
        sys.argv[1] = sys.argv[1]+'/'

    PATH=os.path.dirname(os.path.abspath(__file__))
    PATH_AUDIO=PATH+sys.argv[1]
    PATH_BB=PATH_AUDIO+"/../reps/full_broadband/"
    PATH_NB=PATH_AUDIO+"/../reps/full_narrowband/"
    
    with open("config.json") as f:
        data = f.read()
    config = json.loads(data)
    
    FS=config['general']['FS']
    tst_percent=config['general']['ae_tst_percent']
    
    
    if not os.path.exists(PATH_BB):
        os.makedirs(PATH_BB)
    if not os.path.exists(PATH_NB):
        os.makedirs(PATH_NB)
        
    if not os.path.exists(PATH_AUDIO+'/train/'):
        split=tts.trainTestSplit(PATH_AUDIO, tst_perc=tst_percent, gen_bal=1)
        split.fileTrTstSplit()
    elif len(os.listdir(PATH_AUDIO+'/train/'))<=2:
        split=tts.trainTestSplit(PATH_AUDIO, tst_perc=tst_percent, gen_bal=1)
        split.fileTrTstSplit()

#     for trtst in ['/train/', '/test/']:
    for trtst in ['/train/']:
        audio_path=PATH_AUDIO+trtst
        bb_path=PATH_BB+trtst
        nb_path=PATH_NB+trtst
        
        hf=os.listdir(audio_path)
        hf.sort()
        if len(hf) == 0:
            print(audio_path+ " is empty...", len(hf))
            sys.exit()
        else:
            print(audio_path, len(hf))

        if not os.path.exists(bb_path):
            os.makedirs(bb_path)
        if not os.path.exists(nb_path):
            os.makedirs(nb_path)
        countbad=0
        countinf=0
        
        for j in range(len(hf)):
            print("Procesing audio", j+1, hf[j]+" of "+str(len(hf)))
            fs_in, data=read(audio_path+hf[j])
            if fs_in!=FS:
                raise ValueError(str(fs_in)+" is not a valid sampling frequency")
            
           
            if len(data.shape)>1:
                continue
            data=data-np.mean(data)
            data=data/np.max(np.abs(data))
            
            file_bb_out=bb_path+hf[j].replace(".wav", "")
            file_nb_out=nb_path+hf[j].replace(".wav", "")
            
            if not os.path.isfile(file_nb_out+"_"+str(0)+".npy"):
                BB_TIME_WINDOW=config['mel_spec']['BB_TIME_WINDOW']
                BB_HOP=config['mel_spec']['BB_HOP']
                BB_NMELS=config['mel_spec']['BB_NMELS']
                NB_TIME_WINDOW=config['mel_spec']['NB_TIME_WINDOW']
                NB_HOP=config['mel_spec']['NB_HOP']
                NB_NMELS=config['mel_spec']['NB_NMELS']

                NFFT=config['mel_spec']['NFFT']
                INTERP_NMELS=config['mel_spec']['INTERP_NMELS']
                TIME_STEPS=config['mel_spec']['FULL_TIME_STEPS']
                FRAME_SIZE=config['mel_spec']['FRAME_SIZE']
                sig_len=len(data)

                FRAME_SIZE=FS*FRAME_SIZE
                for nb,band in enumerate([file_bb_out,file_nb_out]):

                    #binary narrowband: 1 yes, 0 no (i.e. broadband)
                    if nb==0:
                        #broadband: higher time resolution, less frequency resolution
                        HOP=int(FS*BB_HOP)#3ms hop (48 SAMPLES)
                        WIN_LEN=int(FS*BB_TIME_WINDOW)#5ms time window (60 SAMPLES)
                        NMELS=BB_NMELS
                        signal=butter_bandpass_filter(data,50,7000,FS)
                    elif nb==1:
                        #narrowband: higher frequency resolution, less time resolution
                        HOP=int(FS*NB_HOP) #10ms hop (160 SAMPLES)
                        WIN_LEN=int(FS*NB_TIME_WINDOW) #30ms time window (480 SAMPLES)
                        NMELS=NB_NMELS
                        signal=butter_bandpass_filter(data,300,5000,FS)

                    init=0
                    endi=int(FRAME_SIZE)
                    
                    mat=np.zeros((1,INTERP_NMELS,TIME_STEPS), dtype=np.float32)
                    imag=melspectrogram(signal, sr=FS, n_fft=NFFT, win_length=WIN_LEN, hop_length=HOP, n_mels=NMELS, fmax=FS//2)
                    imag=imag[np.where(imag[:,0]>0)]
                    imag=cv2.resize(imag,(TIME_STEPS,INTERP_NMELS),interpolation=cv2.INTER_CUBIC)
                    imag=np.abs(imag)
                    if np.min(np.min(imag))<=0:
                        warnings.warns("There is Inf values in the Mel spectrogram")
                        continue
                    imag=np.log(imag, dtype=np.float32)
                    mat[0,:,:]=imag
                    np.save(band+".npy",mat)

