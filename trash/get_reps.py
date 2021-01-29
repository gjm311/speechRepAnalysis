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
from phonetGM2 import Phonet
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


def get_plosive_audio_frames(audio,snip_len,pkl_path):
    phon=Phonet()    
    audio_len=len(audio)
    
    if os.path.isfile(pkl_path):
        wav_df=pd.read_pickle(pkl_path)
    else:
        wav_df=phon.get_phon_wav(audio_file=audio, phonclass="all", feat_file=pkl_path)
    
    unique_df=wav_df.loc[data_stor['phoneme'].diff() != 0]
    
    num_frames=unique_df['phoneme'].value_counts().filter(['p','t','k']).sum()
    frames=np.zeros((num_frames,snip_len))
    plosive_starts=unique_df['time'][unique_df.phoneme.isin(['p','t','k'])].to_numpy()*16000/1000

    for itr,start_time in enumerate(plosive_starts):
        start_time=int(start_time)
        if start_time>(audio_len-snip_len):
            new_start=snip_len-(audio_len-start_time)
            frames[itr,:]=audio[new_start:]
        else:
            frames[itr,:]=audio[start_time:start_time+snip_len]
    
    return frames


def get_frames(audio, audio_name, snip_len, phon_book_path, pkl_path):
    snip_len=int(snip_len*16000/1000)
    phon_book_pkl=phon_book_path+'/phon_book.pkl'
    pkl_path=pkl_path+'/'+audio_name+'.pkl' 
    
    if os.path.isfile(phon_book_pkl):
        phon_book=pd.read_pickle(phon_book_pkl)
        if audio_name in phon_book['audio_names'].to_numpy():
            try:
                with open(phon_book_path+'/'+audio_name+'.npy', 'rb') as f:
                    frames=np.load(f)
            except:
                frames=get_plosive_audio_frames(audio,snip_len,pkl_path)
                with open(phon_book_path+'/'+audio_name+'.npy', 'wb') as f:
                    np.save(f, frames)
                    
            return frames            
        else:
            phon_dic={'audio_names':audio_name}
            phon_book=phon_book.append(pd.Series(phon_dic,name=phon_book.index.max()+1))
            phon_book.to_pickle(phon_book_pkl)
            
            frames=get_plosive_audio_frames(audio,snip_len,pkl_path)
            with open(phon_book_path+'/'+audio_name+'.npy', 'wb') as f:
                np.save(f, frames)
            return frames            
    else:
        phon_book=pd.DataFrame([audio_name],columns=['audio_names'])
        phon_book.to_pickle(phon_book_pkl)
        frames=get_plosive_audio_frames(audio,snip_len,pkl_path)
        with open(phon_book_path+'/'+audio_name+'.npy', 'wb') as f:
            np.save(f, frames)
        return frames 
     
    
    

if __name__ == "__main__":
    
    if len(sys.argv)!=2:
        print("python get_rep.py <path_audios>")
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
    PATH_FRAMES_SPEC=PATH_AUDIO+"/../frames/spec/"
    PATH_FRAMES_WVLT=PATH_AUDIO+"/../frames/wvlt/"
    PATH_PHON_PROBS=PATH_AUDIO+"/../phonet_probs/"
    
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
    if not os.path.exists(PATH_FRAMES_SPEC):
        os.makedirs(PATH_FRAMES_SPEC)
    if not os.path.exists(PATH_FRAMES_WVLT):
        os.makedirs(PATH_FRAMES_WVLT)
    if not os.path.exists(PATH_PHON_PROBS):
        os.makedirs(PATH_PHON_PROBS)
        
    if not os.path.exists(PATH_AUDIO+'/train/'):
        split=tts.trainTestSplit(PATH_AUDIO, tst_perc=tst_percent)
        split.fileTrTstSplit()
    elif len(os.listdir(PATH_AUDIO+'/train/'))<=2:
        split=tts.trainTestSplit(PATH_AUDIO, tst_perc=tst_percent)
        split.fileTrTstSplit()
    
    enrgy = {'Min broadband Scale': [], 'Max broadband Scale': [], 'Min narrowband Scale': [], 'Max narrowband Scale': [], 'Min wvlt Scale': [], 'Max wvlt Scale': []}
    minBB_en = np.inf
    maxBB_en = -np.inf
    minNB_en = np.inf
    maxNB_en = -np.inf
    maxWvlt_en = -np.inf
    minWvlt_en = np.inf
    
    
    for trtst in ['/train/', '/test/']:
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
            
            
        if os.path.isfile(bb_path+hf[0].replace(".wav", "")+"_"+str(0)+".npy"):
            npys=os.listdir(bb_path)
            npys.sort()
            max_unlink=int(npys[-1].split('.')[0].split('_')[-1])
            for ii in range(max_unlink):
                try:
                    os.unlink(bb_path+"_".join(npys[-1].split("_")[:-1])+"_"+str(ii)+".npy")
                except:
                    continue
            
        if os.path.isfile(nb_path+hf[0].replace(".wav", "")+"_"+str(0)+".npy"):
            npys=os.listdir(nb_path)
            npys.sort()
            max_unlink=int(npys[-1].split('.')[0].split('_')[-1])
            for ii in range(max_unlink):
                try:
                    os.unlink(nb_path+"_".join(npys[-1].split("_")[:-1])+"_"+str(ii)+".npy")
                except:
                    continue
                
        if os.path.isfile(wvlt_path+hf[0].replace(".wav", "")+"_"+str(0)+".npy"):
            npys=os.listdir(wvlt_path)
            npys.sort()
            max_unlink=int(npys[-1].split('.')[0].split('_')[-1])
            for ii in range(max_unlink):
                try:
                    os.unlink(wvlt_path+"_".join(npys[-1].split("_")[:-1])+"_"+str(ii)+".npy")
                except:
                    continue
                    
                
        #loop through every audio file
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

                frames=get_frames(data, audio_name=hf[j].replace(".wav", ""), snip_len=SNIP_LEN, phon_book_path=PATH_FRAMES_WVLT, pkl_path=PATH_PHON_PROBS)

                for k,frame in enumerate(frames):
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
    
            if not os.path.isfile(file_nb_out+"_"+str(0)+".npy"):
                 
                BB_TIME_WINDOW=config['mel_spec']['BB_TIME_WINDOW']
                BB_HOP=config['mel_spec']['BB_HOP']
                BB_NMELS=config['mel_spec']['BB_NMELS']
                NB_TIME_WINDOW=config['mel_spec']['NB_TIME_WINDOW']
                NB_HOP=config['mel_spec']['NB_HOP']
                NB_NMELS=config['mel_spec']['NB_NMELS']
                SNIP_LEN=config['mel_spec']['SNIP_LEN']

                NFFT=config['mel_spec']['NFFT']
                INTERP_NMELS=config['mel_spec']['INTERP_NMELS']
                TIME_STEPS=config['mel_spec']['TIME_STEPS']
                FRAME_SIZE=config['mel_spec']['FRAME_SIZE']
                sig_len=len(data)

                FRAME_SIZE=(FS*FRAME_SIZE)/sig_len
                TIME_SHIFT=FRAME_SIZE/2

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

                    frames=get_frames(data, audio_name=hf[j].replace(".wav", ""), snip_len=SNIP_LEN, phon_book_path=PATH_FRAMES_SPEC, pkl_path=PATH_PHON_PROBS)
                    mat=np.zeros((1,INTERP_NMELS,TIME_STEPS), dtype=np.float32)
                    for k,frame in enumerate(frames):
                        imag=melspectrogram(frame, sr=FS, n_fft=NFFT, win_length=WIN_LEN, hop_length=HOP, n_mels=NMELS, fmax=FS//2)
                        imag=imag[np.where(imag[:,0]>0)]
                        imag=cv2.resize(imag,(TIME_STEPS,INTERP_NMELS),interpolation=cv2.INTER_CUBIC)
                        imag=np.abs(imag)
                        if np.min(np.min(imag))<=0:
                            warnings.warns("There is Inf values in the Mel spectrogram")
                            continue
                        imag=np.log(imag, dtype=np.float32)
                        mat[0,:,:]=imag
                        np.save(band+"_"+str(k)+".npy",mat)

                        max_curr = np.max(imag)
                        min_curr = np.min(imag)
                        if nb==0:
                            if max_curr > maxBB_en:
                                maxBB_en = max_curr
                            if min_curr < minBB_en:
                                minBB_en = min_curr  
                        elif nb==1:
                            if max_curr > maxNB_en:
                                maxNB_en = max_curr
                            if min_curr < minNB_en:
                                minNB_en = min_curr   

    try:
        enrgy['Min narrowband Scale'].append(minNB_en)
        enrgy['Max narrowband Scale'].append(maxNB_en)
        enrgy['Min broadband Scale'].append(minBB_en)
        enrgy['Max broadband Scale'].append(maxBB_en)
        enrgy['Min wvlt Scale'].append(minWvlt_en)
        enrgy['Max wvlt Scale'].append(maxWvlt_en)
        df = pd.DataFrame(data=enrgy)
        df.to_csv(PATH+'/scales.csv')
    except:
        print("no luck")





        
        
