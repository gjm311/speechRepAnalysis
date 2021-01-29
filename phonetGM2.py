# -*- coding: utf-8 -*-
"""
Compute posteriors probabilities of phonological classes from audio files for several groups of phonemes according to the mode and manner of articulation.
Author: Camilo Vasquez-Correa 2019 (with edits made by Gabriel F Miller)
"""


import sys
import os
import pdb
import numpy as np

import python_speech_features as pyfeat
from six.moves import cPickle as pickle
from scipy.io.wavfile import read
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from keras.layers import Input, BatchNormalization, Bidirectional, GRU, Permute, Dense, TimeDistributed
from keras.utils import np_utils
from keras.models import Model
from keras import optimizers
import gc
from keras import backend as K

class Phonet:

    def __init__(self):

        self.path=os.path.dirname(os.path.abspath(__file__))
        self.GRU_size=64
        self.hidden_size=64
        self.lr=0.001
        self.recurrent_droput_prob=0.0
        self.size_frame=0.025
        self.time_shift=0.025
        self.nfilt=33
        self.len_seq=20
        self.num_labels=2
        self.nfeat=34
        self.thrplot=0.5
        self.nphonemes=22
        


    def model(self, input_size, num_labels):
        """This is the architecture used for the estimation of the phonological classes
        It consists of a 2 Bidirectional GRU layers, followed by a time-distributed dense layer

        :param input_size: size of input for the BGRU layers (number of features x sequence length).
        :param num_labels: number of labels to be recogized by the DNN (2 for phonological posteriros and 21 for the phoneme recogizer).
        :returns: A Keras model of a 2-layer BGRU neural network.
        """
        input_data=Input(shape=(input_size))
        x=input_data
        x=BatchNormalization()(x)
        x=Bidirectional(GRU(self.GRU_size, recurrent_dropout=self.recurrent_droput_prob, return_sequences=True,reset_after=False))(x)
        x=Bidirectional(GRU(self.GRU_size, recurrent_dropout=self.recurrent_droput_prob, return_sequences=True,reset_after=False))(x)
        x = TimeDistributed(Dense(self.hidden_size, activation='relu'))(x)
        x = TimeDistributed(Dense(num_labels, activation='softmax'))(x)
        modelGRU=Model(inputs=input_data, outputs=x)
        opt=optimizers.Adam(lr=self.lr)
        modelGRU.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return modelGRU

    def get_feat(self, signal, fs):
        """
        This method extracts the log-Mel-filterbank energies used as inputs
        of the model.

        :param signal: the audio signal from which to compute features. Should be an N array.
        :param fs: the sample rate of the signal we are working with, in Hz.
        :returns: A numpy array of size (NUMFRAMES by 33 log-Mel-filterbank energies) containing features. Each row holds 1 feature vector.
        """
        signal=signal-np.mean(signal)
        signal=signal/np.max(np.abs(signal))
        fill=len(signal)%int(fs*self.size_frame*self.len_seq)

        fillv=0.05*np.random.randn(fill)-0.025
        signal=np.hstack((signal,fillv))
        Fbank, energy=pyfeat.fbank(signal,samplerate=fs,winlen=self.size_frame,winstep=self.time_shift,
          nfilt=self.nfilt,nfft=512,lowfreq=0,highfreq=None,preemph=0.97)

        energy= np.expand_dims(energy, axis=1)
        feat2=np.concatenate((Fbank,energy),axis=1)
        return np.log10(feat2)

    def number2phoneme(self, seq):
        """
        Converts the prediction of the neural network for phoneme recognition to a list of phonemes.

        :param seq: sequence of integers obtained from the preiction of the neural network for phoneme recognition.
        :returns: A list of strings of the phonemes recognized for each time-frame.
        """


        list_phonemes=["a","e","i","o","u",
                        "b","d","f","x","k","l","m","n","p","r","rr","s","t",
                        "L","g","tS","sil"]
        try:
            phonemes=[list_phonemes[j] for j in seq]

            for j in range(1,len(phonemes)-1):
                if phonemes[j]!=phonemes[j-1] and phonemes[j]!=phonemes[j+1]:
                    phonemes[j]=phonemes[j-1]

            return phonemes
        except:
            print("number:*"+ seq+"*is not in the list")
            return np.nan


    def get_phon_wav(self, audio_file, phonclass, feat_file=None, save=1):
        """
        Estimate the phonological classes using the BGRU models for an audio file (.wav)

        :param audio_file: numpy signal
        :param feat_file: file (.pkl) to save the posteriors for the phonological classes
        :param phonclass: phonological class to be evaluated:("consonantal","back","anterior", "open", "close", "stop",###"nasal","continuant",  "lateral", "flap", "trill", "voice", "strident","labial", "dental", "velar", "pause", "vocalic", "all").
        :param plot_flag: True or False, whether you want plots of phonological classes or not
        :returns: A panda df created at FEAT_FILE with the posterior probabilities for the phonological classes.
        """

        if phonclass == 'all':
            keys_val=["consonantal", "back", "anterior", "open", "close", "stop", "nasal", "stop", "continuant","lateral", "flap", "trill", "voice", "strident", "labial", "dental", "velar", "pause", "vocalic"]
        else:
            keys_val=phonclass

        Models=[]
        input_size=(self.len_seq, self.nfeat)

        for l in range(len(keys_val)):
            model_file=self.path+"/models/"+keys_val[l]+".h5"
            Model=self.model(input_size, self.num_labels)
            
            Model.load_weights(model_file)
            Models.append(Model)

        Model_phonemes=self.path+"/models/phonemes.h5"
        input_size_phon=(self.len_seq, self.nfeat)
        Model_phon=self.model(input_size, self.nphonemes)
        Model_phon.load_weights(Model_phonemes)

        file_scaler=self.path+"/models/scaler.pickle"
        with open(file_scaler, 'rb') as f:
            dict_scaler = pickle.load(f)
            MU=dict_scaler["MU"]
            STD=dict_scaler["STD"]
            f.close()

#         fs, signal=read(audio_file)
#         if fs!=16000:
#             raise ValueError(str(fs)+" is not a valid sampling frequency")
        feat=self.get_feat(audio_file,16000)
        nf=int(feat.shape[0]/self.len_seq)
        start=0
        fin=self.len_seq
        Feat=[]
        for j in range(nf):
            featmat_t=feat[start:fin,:]
            Feat.append(featmat_t)
            start=start+self.len_seq
            fin=fin+self.len_seq
        Feat=np.stack(Feat, axis=0)
        Feat=Feat-MU
        Feat=Feat/STD
        df={}

        pred_mat_phon=np.asarray(Model_phon.predict(Feat))
        pred_mat_phon_seq=np.concatenate(pred_mat_phon,0)
        pred_vec_phon=np.argmax(pred_mat_phon_seq,1)
        phonemes_list=self.number2phoneme(pred_vec_phon)

        t2=np.arange(len(pred_vec_phon))*self.time_shift
        df["time"]=t2
        df["phoneme"]=phonemes_list
        for l in range(len(Models)):
            pred_mat=np.asarray(Models[l].predict(Feat))
            pred_matv=pred_mat[:,:,1]
            df[keys_val[l]]=np.hstack(pred_matv)
        
        df2=pd.DataFrame(df)
        if save==1:
            df2.to_pickle(feat_file,protocol=4)
        K.clear_session()
        gc.collect()
        return(df2)



    def get_phon_path(self, audio_path, feat_path, min_files, max_files, phonclass="all", plot_flag=False):
        """
        Estimate the phonological classes using the BGRU models for all the (.wav) audio files included inside a directory

        :param audio_path: directory with (.wav) audio files inside, sampled at 16 kHz
        :param feat_path: directory were the computed phonological posteriros will be stores as a (.csv) file per (.wav) file from the input directory
        :param phonclass: phonological class to be evaluated ("consonantal", "back", "anterior", "open", "close", "nasal", "stop",
                                                  "continuant",  "lateral", "flap", "trill", "voice", "strident",
                                                  "labial", "dental", "velar", "pause", "vocalic", "all").
        :param plot_flag: True or False, whether you want plots of phonological classes or not
        :returns: A directory with csv files created with the posterior probabilities for the phonological classes.
        """

        hf=os.listdir(audio_path)
        hf.sort()

        if not os.path.exists(feat_path):
            os.makedirs(feat_path)

        if feat_path[-1]!="/":
            feat_path=feat_path+"/"
        
        
        if len(hf) < max_files:
            max_files = len(hf)
        
        for j in range(min_files-1,max_files):
            audio_file=audio_path+hf[j]
            feat_file=feat_path+hf[j].replace(".wav", ".csv")
            print("processing file ", j+1, " from ", str(max_files), " ", hf[j])
            self.get_phon_wav(audio_file, feat_file, max_files, phonclass, plot_flag)


    def get_posteriorgram(self, audio_file):
        """
        Estimate the posteriorgram for an audio file (.wav) sampled at 16kHz

        :param audio_file: file audio (.wav) sampled at 16 kHz
        :returns: plot of the posteriorgram
        """
        if audio_file.find('.wav')==-1 and audio_file.find('.WAV')==-1:
            raise ValueError(audio_file+" is not a valid audio file")

        keys_val=["anterior","back","close","consonantal","open"]#,"continuant","dental","flap",
             # "labial","lateral","nasal","pause","stop","strident","trill","velar","vocalic","voice"]

        Models=[]
        input_size=(self.len_seq, self.nfeat)
        for l in range(len(keys_val)):
            model_file=self.path+"/models/"+keys_val[l]+".h5"
            Model=self.model(input_size, self.num_labels)
            Model.load_weights(model_file)
            Models.append(Model)

        Model_phonemes=self.path+"/models/phonemes.h5"
        input_size_phon=(self.len_seq, self.nfeat)
        Model_phon=self.model(input_size, self.nphonemes)
        Model_phon.load_weights(Model_phonemes)

        file_scaler=self.path+"/models/scaler.pickle"
        with open(file_scaler, 'rb') as f:
            dict_scaler = pickle.load(f)
            MU=dict_scaler["MU"]
            STD=dict_scaler["STD"]
            f.close()

        fs, signal=read(audio_file)
        if fs!=16000:
            raise ValueError(str(fs)+" is not a valid sampling frequency")
        feat=self.get_feat(signal,fs)
        nf=int(feat.shape[0]/self.len_seq)
        start=0
        fin=self.len_seq
        Feat=[]
        for j in range(nf):
            featmat_t=feat[start:fin,:]
            Feat.append(featmat_t)
            start=start+self.len_seq
            fin=fin+self.len_seq
        Feat=np.stack(Feat, axis=0)
        Feat=Feat-MU
        Feat=Feat/STD
        df={}

        pred_mat_phon=np.asarray(Model_phon.predict(Feat))
        pred_mat_phon_seq=np.concatenate(pred_mat_phon,0)
        pred_vec_phon=np.argmax(pred_mat_phon_seq,1)
        phonemes_list=self.number2phoneme(pred_vec_phon)

        t=np.arange(len(pred_vec_phon))*self.time_shift
        posteriors=[]
        for l in range(len(Models)):
            pred_mat=np.asarray(Models[l].predict(Feat))
            pred_matv=pred_mat[:,:,1]
            posteriors.append(np.hstack(pred_matv))

        posteriors=np.vstack(posteriors)
        plt.figure()
        plt.imshow(np.flipud(posteriors), extent=[0, t[-1], 0, len(keys_val)], aspect='auto')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Phonological class")
        plt.yticks(np.arange(len(keys_val))+0.5, keys_val)
        ini=t[0]
        for nu in range(1,len(phonemes_list)):
            if phonemes_list[nu]!=phonemes_list[nu-1] or nu==len(phonemes_list)-1:
                difft=t[nu]-ini
                plt.text(x=ini+difft/2, y=19, s="/"+phonemes_list[nu-1]+"/", color="k", fontsize=12)
                ini=t[nu]
        plt.colorbar()
        plt.show()
