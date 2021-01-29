import sys
import os
import torch
import pywt
import pickle
import scaleogram as scg 
from scipy.io.wavfile import read
import scipy
import numpy as np
from AEspeech import AEspeech
import json
import argparse
import pdb
from librosa.feature import melspectrogram

import toolbox.traintestsplit as tts



if __name__ == "__main__":

    if len(sys.argv)!=4:
        print("python get_reconError.py <CAE or RAE> <narrowband or broadband or wvlt or mc_fuse> <path_speech>")
        sys.exit()
    
        
    #path_speech: "./pdSpanish/speech/"
    
    if sys.argv[3][0] != '/':
        sys.argv[3] = '/'+sys.argv[3]
        
    if sys.argv[3][-1] != "/":
        sys.argv[3] = sys.argv[3]+'/'

    PATH=os.path.dirname(os.path.abspath(__file__))
    
    mod=sys.argv[1]
    rep=sys.argv[2]
    
    #LOAD CONFIG.JSON INFO
    with open("config.json") as f:
        data = f.read()
    config = json.loads(data)
    unit=config['general']['UNITS']
    
    save_path=PATH+'/pts/'+'/reconErrs/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_path=save_path+mod+'_'+rep+'.pickle'
    
    if rep=='wvlt':
        n_freqs=config['wavelet']['NBF']
    elif rep=='mc_fuse':
        n_freqs=config['mel_spec']['INTERP_NMELS']*2
    else:
        n_freqs=config['mel_spec']['INTERP_NMELS'] 
    
    data={spk:{'means':[], 'stds':[]} for spk in ['pd','hc']}
#     utters= os.listdir(PATH+sys.argv[3])
    utters=['pataka']
    
    for itr,utter in enumerate(utters):
        path_utter=PATH+sys.argv[3]+'/'+utter+'/'
    
        for spk in ['pd','hc']:
            path_audio=path_utter+spk+'/'
            dirNames=os.listdir(path_audio)
            dirNames.sort()
            wav_files=[name for name in dirNames if '.wav' in name]

            num_files=len(wav_files)
            data_curr_means=np.zeros((num_files,n_freqs))
            data_curr_sds=np.zeros((num_files,n_freqs))

            for ii,wav_file in enumerate(wav_files):
                wav_file=path_audio+wav_file
                fs_in, signal=read(wav_file)

                # for mod in models:
                aespeech=AEspeech(model=mod,units=unit,rep=rep)
                error=aespeech.compute_rec_error_features(wav_file,plosives_only=0)
                data_curr_means[ii,:]=np.mean(error,axis=0)
                data_curr_sds[ii,:]=np.std(error,axis=0)
                
            if itr==0:
                data[spk]['means']=data_curr_means
                data[spk]['stds']=data_curr_sds
            else:
                data[spk]['means']=np.concatenate((data[spk]['means'],data_curr_means
),axis=0)
                data[spk]['stds']=np.concatenate((data[spk]['stds'],data_curr_sds
),axis=0)
                
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
