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
import pandas as pd
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
    
    save_path=save_path+mod+'_'+rep+'_gender.pickle'
    
    if rep=='wvlt':
        n_freqs=config['wavelet']['NBF']
    elif rep=='mc_fuse':
        n_freqs=2*config['mel_spec']['INTERP_NMELS'] 
    else:
        n_freqs=config['mel_spec']['INTERP_NMELS'] 
    
    data={spk:{gen:{'means':[], 'stds':[]} for gen in ['m','f']} for spk in ['pd','hc']}
    utters= os.listdir(PATH+sys.argv[3])
    
    mfda_path=PATH+"/pdSpanish/"
    genders=pd.read_csv(mfda_path+"metadata-Spanish_All.csv")['gender'].values
    
    for itr,utter in enumerate(utters):
        path_utter=PATH+sys.argv[3]+'/'+utter+'/'
        
        data_curr_means_pdm=np.zeros((len(genders)//4,n_freqs))
        data_curr_sds_pdm=np.zeros((len(genders)//4,n_freqs))
        data_curr_means_pdw=np.zeros((len(genders)//4,n_freqs))
        data_curr_sds_pdw=np.zeros((len(genders)//4,n_freqs))
        data_curr_means_hcm=np.zeros((len(genders)//4,n_freqs))
        data_curr_sds_hcm=np.zeros((len(genders)//4,n_freqs))
        data_curr_means_hcw=np.zeros((len(genders)//4,n_freqs))
        data_curr_sds_hcw=np.zeros((len(genders)//4,n_freqs))
        
        pdm_itr=0
        hcm_itr=0
        pdf_itr=0
        hcf_itr=0
        for s_itr,spk in enumerate(['pd','hc']):
            path_audio=path_utter+spk+'/'
            dirNames=os.listdir(path_audio)
            dirNames.sort()
            wav_files=[name for name in dirNames if '.wav' in name]

            num_files=len(wav_files)
            
            for ii,wav_file in enumerate(wav_files):
                wav_file=path_audio+wav_file
                fs_in, signal=read(wav_file)

                # for mod in models:
                aespeech=AEspeech(model=mod,units=unit,rep=rep)
                error=aespeech.compute_rec_error_features(wav_file)
                if genders[(s_itr*num_files)+ii]=='M':
                    if spk=='pd':
                        data_curr_means_pdm[pdm_itr,:]=np.mean(error,axis=0)
                        data_curr_sds_pdm[pdm_itr,:]=np.std(error,axis=0)
                        pdm_itr+=1
                    elif spk=='hc':
                        data_curr_means_hcm[hcm_itr,:]=np.mean(error,axis=0)
                        data_curr_sds_hcm[hcm_itr,:]=np.std(error,axis=0)
                        hcm_itr+=1
                elif genders[(s_itr*num_files)+ii]=='F':
                    if spk=='pd':
                        data_curr_means_pdw[pdf_itr,:]=np.mean(error,axis=0)
                        data_curr_sds_pdw[pdf_itr,:]=np.std(error,axis=0)
                        pdf_itr+=1
                    elif spk=='hc':
                        data_curr_means_hcw[hcf_itr,:]=np.mean(error,axis=0)
                        data_curr_sds_hcw[hcf_itr,:]=np.std(error,axis=0)
                        hcf_itr+=1
                
        if itr==0:
            data['pd']['m']['means']=data_curr_means_pdm
            data['pd']['m']['stds']=data_curr_sds_pdm
            data['pd']['f']['means']=data_curr_means_pdw
            data['pd']['f']['stds']=data_curr_sds_pdw
            data['hc']['m']['means']=data_curr_means_hcm
            data['hc']['m']['stds']=data_curr_sds_hcm
            data['hc']['f']['means']=data_curr_means_hcw
            data['hc']['f']['stds']=data_curr_sds_hcw
        else:
            data['pd']['m']['means']=np.concatenate((data['pd']['m']['means'],data_curr_means_pdm
),axis=0)
            data['pd']['m']['stds']=np.concatenate((data['pd']['m']['stds'],data_curr_sds_pdm
),axis=0)
            data['pd']['f']['means']=np.concatenate((data['pd']['f']['means'],data_curr_means_pdw
),axis=0)
            data['pd']['f']['stds']=np.concatenate((data['pd']['f']['stds'],data_curr_sds_pdw
),axis=0)
            data['hc']['m']['means']=np.concatenate((data['hc']['m']['means'],data_curr_means_hcm
),axis=0)
            data['hc']['m']['stds']=np.concatenate((data['hc']['m']['stds'],data_curr_sds_hcm
),axis=0)
            data['hc']['f']['means']=np.concatenate((data['hc']['f']['means'],data_curr_means_hcw
),axis=0)
            data['hc']['f']['stds']=np.concatenate((data['hc']['f']['stds'],data_curr_sds_hcw
),axis=0)
               
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
