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
    PATH_BB=PATH_AUDIO+"/../reps/broadband/"
    PATH_NB=PATH_AUDIO+"/../reps/narrowband/"
    PATH_WVLT=PATH_AUDIO+"/../reps/wvlt/"
    
    with open("config.json") as f:
        data = f.read()
    config = json.loads(data)
    
    FS=config['general']['FS']
    tst_percent=config['general']['ae_tst_percent']
    
    
    if not os.path.exists(PATH_BB):
        os.makedirs(PATH_BB)
    if not os.path.exists(PATH_NB):
        os.makedirs(PATH_NB)
    if not os.path.exists(PATH_WVLT):
        os.makedirs(PATH_WVLT)
        
    if not os.path.exists(PATH_AUDIO+'/train/'):
        split=tts.trainTestSplit(PATH_AUDIO, tst_perc=tst_percent, gen_bal=1)
        split.fileTrTstSplit()
    elif len(os.listdir(PATH_AUDIO+'/train/'))<=2:
        split=tts.trainTestSplit(PATH_AUDIO, tst_perc=tst_percent, gen_bal=1)
        split.fileTrTstSplit()
    
    enrgy = {'Min broadband Scale': [], 'Max broadband Scale': [], 'Min narrowband Scale': [], 'Max narrowband Scale': [], 'Min wvlt Scale': [], 'Max wvlt Scale': []}
    minBB_en = np.inf
    maxBB_en = -np.inf
    minNB_en = np.inf
    maxNB_en = -np.inf
    maxWvlt_en = -np.inf
    minWvlt_en = np.inf
    
#     for trtst in ['/train/', '/test/']:
    for trtst in ['/train/']:
        audio_path=PATH_AUDIO+trtst
        bb_path=PATH_BB+trtst
        nb_path=PATH_NB+trtst
        wvlt_path=PATH_WVLT+trtst
        
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
        if not os.path.exists(wvlt_path):
            os.makedirs(wvlt_path)

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
            file_wvlt_out=wvlt_path+hf[j].replace(".wav", "")
            
            if not os.path.isfile(file_wvlt_out+"_"+str(0)+".npy"):              
                SNIP_LEN=config['wavelet']['SNIP_LEN']
                NBF=config['wavelet']['NBF']
                TIME_STEPS=config['wavelet']['TIME_STEPS']
                OVRLP=config['wavelet']['OVRLP']
                NFR=int(data.shape[0]*1000/(FS*SNIP_LEN))
                WV_FRAME_SIZE=int(data.shape[0]/NFR)
                SHIFT=int(WV_FRAME_SIZE*OVRLP)
                DIM=(TIME_STEPS,NBF)

                init=0
                endi=WV_FRAME_SIZE
                wv_mat=np.zeros((1,NBF,TIME_STEPS),dtype=np.float32)

                for k in range(NFR):    
                    frame=data[init:endi]                         
                    init=init+int(SHIFT)
                    endi=endi+int(SHIFT)
                    cwtmatr,_ = pywt.cwt(frame, np.arange(1,NBF+1), 'morl')
                    bicubic_img = cv2.resize(np.real(cwtmatr),DIM,interpolation=cv2.INTER_CUBIC)

                    #Looking for min/max coefficients for standardization.
                    max_curr = np.max(bicubic_img)
                    min_curr = np.min(bicubic_img)
                    if max_curr > maxWvlt_en:
                        maxWvlt_en = max_curr
                    if min_curr < minWvlt_en:
                        minWvlt_en = min_curr    

                    wv_mat[0,:,:]=bicubic_img
                    np.save(file_wvlt_out+"_"+str(k)+".npy",wv_mat)


#             if not os.path.isfile(file_nb_out+"_"+str(0)+".npy"):
#                 BB_TIME_WINDOW=config['mel_spec']['BB_TIME_WINDOW']
#                 BB_HOP=config['mel_spec']['BB_HOP']
#                 BB_NMELS=config['mel_spec']['BB_NMELS']
#                 NB_TIME_WINDOW=config['mel_spec']['NB_TIME_WINDOW']
#                 NB_HOP=config['mel_spec']['NB_HOP']
#                 NB_NMELS=config['mel_spec']['NB_NMELS']

#                 NFFT=config['mel_spec']['NFFT']
#                 INTERP_NMELS=config['mel_spec']['INTERP_NMELS']
#                 TIME_STEPS=config['mel_spec']['TIME_STEPS']
#                 FRAME_SIZE=config['mel_spec']['FRAME_SIZE']
#                 TIME_SHIFT=config['mel_spec']['TIME_SHIFT']
#                 sig_len=len(data)

#                 FRAME_SIZE=FS*FRAME_SIZE
#                 TIME_SHIFT=FS*TIME_SHIFT
#                 for nb,band in enumerate([file_bb_out,file_nb_out]):

#                     #binary narrowband: 1 yes, 0 no (i.e. broadband)
#                     if nb==0:
#                         #broadband: higher time resolution, less frequency resolution
#                         HOP=int(FS*BB_HOP)#3ms hop (48 SAMPLES)
#                         WIN_LEN=int(FS*BB_TIME_WINDOW)#5ms time window (60 SAMPLES)
#                         NMELS=BB_NMELS
#                         signal=butter_bandpass_filter(data,50,7000,FS)
#                     elif nb==1:
#                         #narrowband: higher frequency resolution, less time resolution
#                         HOP=int(FS*NB_HOP) #10ms hop (160 SAMPLES)
#                         WIN_LEN=int(FS*NB_TIME_WINDOW) #30ms time window (480 SAMPLES)
#                         NMELS=NB_NMELS
#                         signal=butter_bandpass_filter(data,300,5000,FS)

#                     init=0
#                     endi=int(FRAME_SIZE)
#                     nf=int(sig_len/TIME_SHIFT)-1
                    
#                     if nf>0:
#                         mat=np.zeros((1,INTERP_NMELS,TIME_STEPS), dtype=np.float32)
#                         for k in range(nf):
#                             frame=signal[init:endi]
#                             imag=melspectrogram(frame, sr=FS, n_fft=NFFT, win_length=WIN_LEN, hop_length=HOP, n_mels=NMELS, fmax=FS//2)
#                             imag=imag[np.where(imag[:,0]>0)]
#                             imag=cv2.resize(imag,(TIME_STEPS,INTERP_NMELS),interpolation=cv2.INTER_CUBIC)
#                             imag=np.abs(imag)
#                             init=init+int(TIME_SHIFT)
#                             endi=endi+int(TIME_SHIFT)
#                             if np.min(np.min(imag))<=0:
#                                 warnings.warns("There is Inf values in the Mel spectrogram")
#                                 continue
#                             imag=np.log(imag, dtype=np.float32)
#                             mat[0,:,:]=imag
#                             np.save(band+"_"+str(k)+".npy",mat)

#                             max_curr = np.max(imag)
#                             min_curr = np.min(imag)
#                             if nb==0:
#                                 if max_curr > maxBB_en:
#                                     maxBB_en = max_curr
#                                 if min_curr < minBB_en:
#                                     minBB_en = min_curr  
#                             elif nb==1:
#                                 if max_curr > maxNB_en:
#                                     maxNB_en = max_curr
#                                 if min_curr < minNB_en:
#                                     minNB_en = min_curr   
#                     else:
#                         print("WARNING, audio too short", hf[j], len(signal))
#                         countbad+=1
#     try:
#         enrgy['Min narrowband Scale'].append(minNB_en)
#         enrgy['Max narrowband Scale'].append(maxNB_en)
#         enrgy['Min broadband Scale'].append(minBB_en)
#         enrgy['Max broadband Scale'].append(maxBB_en)
#         enrgy['Min wvlt Scale'].append(minWvlt_en)
#         enrgy['Max wvlt Scale'].append(maxWvlt_en)
#         df = pd.DataFrame(data=enrgy)
#         df.to_csv(PATH+'/scales.csv')
#     except:
#         print("no luck")