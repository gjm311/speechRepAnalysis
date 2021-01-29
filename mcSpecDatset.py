
import torch.utils.data as data
import os
import numpy as np

import torch

class mcSpecDataset(data.Dataset):
    def __init__(self, datafolder1, datafolder2):
        self.datafolder1 = datafolder1
        self.datafolder2 = datafolder2
        self.image_files_list1 =os.listdir(datafolder1)
        self.image_files_list2 =os.listdir(datafolder2)


    def __len__(self):
        return len(self.image_files_list1)

    def __getitem__(self, idx):
        img_name1 = self.datafolder1+self.image_files_list1[idx]
        img_name2 = self.datafolder2+self.image_files_list2[idx]
        mat={'bb':np.load(img_name1),'nb':np.load(img_name2)}      

        return mat