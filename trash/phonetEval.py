#GOAL: Get correlation scores for all speech files (reconstructed vs. original).
#Reconstructed versions are obtained via autoencoders.
# -*- coding: utf-8 -*-


from AEspeech import AEspeech
import os
import sys
from scipy.io.wavfile import read
import scipy
import torch
from phonetGM2 import Phonet

if __name__=="__main__":

    if len(sys.argv)!=2:
        print("python get_spec_full.py <path_speech>")
        sys.exit()
#"/tedx_spanish_corpus/speech/test/"   

    end_path = sys.argv[1]
    if end_path[0]!='/':
        end_path='/'+end_path
        
    PATH=os.path.dirname(os.path.abspath(__file__))
    path_audio=PATH+end_path
    phon=Phonet()
    
    FS=16000
    NFFT=512  
    
    #set loop parameters
#     reps=['spec','wvlt']
    rep='spec'
    models=['CAE','RAE']
#     units=[]
    unit=256
    num_files=len(os.listdir(path_audio))
    hf=os.listdir(path_audio)  
    #for each wav_file, resample (handled in aespeech)
    for j,wav_file in enumerate(os.listdir(path_audio)):
        
        #loop through different models and possible units
#         for rep in reps:
        save_path=PATH+'/phonCSVs/'+rep+'/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for mod in models:
#                 for unit in units:

            #compute the decoded spectrograms from the autoencoder and standardize or get coeffs for wvlt representation
            aespeech=AEspeech(model=mod,units=unit,rep=rep)
            if rep=='spec':
                mat=aespeech.compute_spectrograms(path_audio+'/'+wav_file)
                mat=aespeech.standard(mat)
            if rep=='wvlt':
                mat=aespeech.compute_cwt(path_audio+'/'+wav_file)

            if torch.cuda.is_available():
                mat=mat.cuda()
            to,bot=aespeech.AE.forward(mat)
            to=aespeech.destandard(to)

            recon=to.cpu().data.numpy()
            ori=mat.cpu().data.numpy()
            if rep=='spec':
                speech_recon=aespeech.mel2speech(recon)
                speech_ori=aespeech.mel2speech(ori)
            
            reconPath_phonSave=save_path+'/'+str(unit)+'_'+mod+'_'+wav_file+'_recon.csv'
            oriPath_phonSave=save_path+'/'+str(unit)+'_'+mod+'_'+wav_file+'_original.csv'    
            phon.get_phon_wav(audio_file=speech_recon,phonclass="all",feat_file=reconPath_phonSave)
            phon.get_phon_wav(audio_file=speech_ori,phonclass="all"feat_file=oriPath_phonSave)

            print("processing file ", j+1, " from ", str(num_files), " ", hf[j])

