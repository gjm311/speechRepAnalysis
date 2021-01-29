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
import sys
import json
sys.path.append("/../../../")
from AEspeech import AEspeech 

with open("config.json") as f:
    data = f.read()
config = json.loads(data)
FS=config['general']['FS']
NFFT=config['mel_spec']['NFFT']


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self


params = AttrDict(
    # Training params
    batch_size=16,
    learning_rate=2e-4,
    max_grad_norm=None,

    # Data params
    sample_rate=FS,
    n_mels={'broadband':config['mel_spec']['BB_NMELS'],'narrowband':config['mel_spec']['NB_NMELS'], 'recon':config['mel_spec']['INTERP_NMELS']},
    n_fft=NFFT,
    hop_samples={'broadband':int(FS*config['mel_spec']['BB_HOP']),'narrowband':int(FS*config['mel_spec']['NB_HOP'])},
    crop_mel_frames=62,  # Probably an error in paper.
    
    # Model params
    residual_layers=30,
    proj={'broadband':512, 'narrowband':384,'recon':512},
    residual_channels=64,
    dilation_cycle_length=10,
    stride={'broadband':6,'narrowband':12},
    noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
)
