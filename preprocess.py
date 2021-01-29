# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT
import sys
import json
import cv2

from librosa.feature import melspectrogram
from argparse import ArgumentParser
from scipy.io.wavfile import read
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm
from functools import partial
from itertools import repeat

from diffwave.params import params

sys.path.append("/../../../")
from AEspeech import AEspeech 
from recon_mel_spec_transform import reconMelSpecTransform

with open("config.json") as f:
    data = f.read()
config = json.loads(data)
import pdb

unit=config['general']['UNITS']
TIME_STEPS=config['mel_spec']['FULL_TIME_STEPS']
TIME_SHIFT=config['mel_spec']['TIME_SHIFT']
FRAME_SIZE=config['mel_spec']['FRAME_SIZE']
INTERP_NMELS=config['mel_spec']['INTERP_NMELS']
FS=config['general']['FS']

def transform(filename,rep,recon):    
#     filename=lst[0]
#     rep=lst[1]
#     recon=lst[2]   
    pdb.set_trace()
    if rep=='narrowband':
        rep_win_len='NB_TIME_WINDOW'
    elif rep=='broadband':
        rep_win_len='BB_TIME_WINDOW'
            
    if recon==0:
        audio, sr = T.load_wav(filename)
        if params.sample_rate != sr:
            raise ValueError(f'Invalid sample rate {sr}.')
        audio = torch.clamp(audio[0] / 32767.5, -1.0, 1.0)
        
        mel_args = {
          'sample_rate': sr,
          'win_length': int(FS*config['mel_spec'][rep_win_len]),
          'hop_length': params.hop_samples[rep],
          'n_fft': params.n_fft,
          'f_min': 20.0,
          'f_max': sr / 2.0,
          'n_mels': params.n_mels[rep],
          'power': 1.0,
          'normalized': True,
      }
        mel_spec_transform = TT.MelSpectrogram(**mel_args)

        with torch.no_grad():
            spectrogram = mel_spec_transform(audio)
            spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
            spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
            np.save(f'{filename}.spec.npy', spectrogram.cpu().numpy())
    else:
        with torch.no_grad():
            aespeech=AEspeech(model='CAE',units=unit,rep='full_'+rep)
            fs_in, signal=read(filename)
            fullmat=melspectrogram(signal.astype(float), sr=FS, n_fft=params.n_fft, win_length=int(FS*config['mel_spec'][rep_win_len]), hop_length=params.hop_samples[rep], n_mels=params.n_mels[rep], fmax=FS//2)
            
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
            
            np.save(f'{filename}.spec.npy', spectrogram.cpu().numpy())

def main(args):
  filenames = glob(f'{args.dir}/**/*.wav', recursive=True)
  rep=args.rep
  recon=int(args.recon)
#   args=[[filename,rep,recon] for filename in filenames]
  with ProcessPoolExecutor(max_workers=2) as executor:
    list(tqdm(executor.map(partial(transform,rep=rep,recon=recon),filenames), desc='Preprocessing', total=len(filenames)))
    

if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train DiffWave')
  parser.add_argument('dir',
      help='directory containing .wav files for training')
  parser.add_argument('rep', default='narrowband', help='speech representation type (narrowband,broadband).')
  parser.add_argument('recon', default=0, help='indicates if reconstructed spectrograms should be used (1->yes)')
  main(parser.parse_args())
  
