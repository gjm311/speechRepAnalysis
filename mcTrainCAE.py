from mcSpecDatset import mcSpecDataset
import time
import torch
import numpy as np
import pandas as pd
import os
import sys
from mcCAE import mcCAEn
import toolbox.traintestsplit as tts
import json
import argparse
import pdb

def standard(tensor, minval, maxval):
    temp=tensor-minval
    return temp/(maxval-minval)

def destandard(tensor, minval, maxval):
    temp=tensor*(maxval-minval)
    return temp+minval

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path)
    epoch = checkpoint_dict['epoch']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading)
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, epoch))
    return model, optimizer, epoch    
    

if __name__=="__main__":

    PATH=os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) != 2:
        print("python mcTrainCAE.py <bottleneck_sizes>")
        sys.exit()
    #rep_path: "./tedx_spanish_corpus/reps/'wvlt or broadband or narrowband'/train/"    
        
    
    reps=['broadband','narrowband']
    path_reps=[PATH+'/tedx_spanish_corpus/reps/'+rep+'/train/' for rep in reps]
        
    for rpathItr,path_rep in enumerate(path_reps):
        if rpathItr==0:
            split=tts.trainTestSplit(path_rep,file_type='.npy', tst_perc=0.3, gen_bal=1)
            if len(os.listdir(path_rep+'train/')) != 0 or len(os.listdir(path_rep+'test/')) != 0:
                split.wavReset()
            split.fileTrTstSplit()
            assign_path=path_rep+'/trTst_idsNPY.pkl'
        else:
            split=tts.trainTestSplit(path_rep,file_type='.npy', tst_perc=0.3, assign_path=assign_path,gen_bal=1)
            if len(os.listdir(path_rep+'train/'))!=0 or len(os.listdir(path_rep+'test/'))!=0:
                split.wavReset()
            split.fileTrTstSplit()

                
    with open("config.json") as f:
        data = f.read()
    config = json.loads(data)
    
    FS=config['general']['FS']
    N_EPOCHS=config['CAE']['N_EPOCHS']
    LR=config['CAE']['LR']
    NUM_W=config['CAE']['NUM_W']
    BATCH_SIZE=config['CAE']['BATCH_SIZE']
    BATCH_SIZE=4
    
    BOTTLE_SIZE=int(sys.argv[1])
    
    SCALERS = pd.read_csv("scales.csv")
      
    model=mcCAEn(BOTTLE_SIZE)
        
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        model.cuda()
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
        
    save_path = PATH+"/pts/mc_fuse/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        
    epochs=np.arange(N_EPOCHS)
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
            
    if os.path.isfile(save_path+str(BOTTLE_SIZE)+'_CAE.pt'):
        model, optimizer, epoch = load_checkpoint(save_path+str(BOTTLE_SIZE)+'_CAE.pt', model,
                                                      optimizer)
        epoch+=1
        if epoch>=N_EPOCHS:
            model=mcCAEn(BOTTLE_SIZE)
            epochs=np.arange(N_EPOCHS)
        else:
            epochs=np.arange(epoch,N_EPOCHS)
        
    criterion = torch.nn.MSELoss()
    
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        model.cuda()
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


    valid_loss_min = np.Inf # set initial "min" to infinity
    
            
    BB_PATH_TRAIN=path_reps[0]+"/train/"
    BB_PATH_TEST=path_reps[0]+"/test/"
    NB_PATH_TRAIN=path_reps[1]+"/train/"
    NB_PATH_TEST=path_reps[1]+"/test/"
    
    train=mcSpecDataset(BB_PATH_TRAIN,NB_PATH_TRAIN)
    test=mcSpecDataset(BB_PATH_TEST,NB_PATH_TEST)
    train_loader=torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, drop_last=True, num_workers=NUM_W)
    test_loader=torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, drop_last=True, num_workers=NUM_W)
    
    BB_MIN_SCALER= float(SCALERS['Min broadband Scale']) #MIN value of total energy.
    BB_MAX_SCALER= float(SCALERS['Max broadband Scale'])  #MAX value of total energy.
    NB_MIN_SCALER= float(SCALERS['Min narrowband Scale']) #MIN value of total energy.
    NB_MAX_SCALER= float(SCALERS['Max narrowband Scale'])  #MAX value of total energy.

    for epoch in epochs:
        start=time.time()
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
        model.train() # prep model for training
        
        for data in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            
            bb_data=standard(data['bb'], BB_MIN_SCALER, BB_MAX_SCALER)
            nb_data=standard(data['nb'], NB_MIN_SCALER, NB_MAX_SCALER)
            bb_data=bb_data.float()
            nb_data=nb_data.float()
            
            if torch.cuda.is_available():
                bb_data=bb_data.cuda()
                nb_data=nb_data.cuda()

            bb_data_out,nb_data_out=model.forward(bb_data,nb_data)

            if torch.cuda.is_available():
                bb_data_out=bb_data_out.cuda()
                nb_data_out=nb_data_out.cuda()
            
            data_cat=torch.cat((bb_data,nb_data),dim=0)
            data_out_cat=torch.cat((bb_data_out,nb_data_out),dim=0)
            loss = criterion(data_out_cat, data_cat)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*data_cat.size(0)
            
        ######################    
        # validate the model #
        ######################
        

        model.eval() # prep model for evaluation
        for data_val in test_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            bb_data_val=standard(data_val['bb'], BB_MIN_SCALER, BB_MAX_SCALER)
            nb_data_val=standard(data_val['nb'], NB_MIN_SCALER, NB_MAX_SCALER)
            
            bb_data_val=bb_data_val.float()
            nb_data_val=nb_data_val.float()
            
            if torch.cuda.is_available():
                bb_data_val=bb_data_val.cuda()
                nb_data_val=nb_data_val.cuda()

            bb_data_val_out, nb_data_val_out=model.forward(bb_data_val,nb_data_val)

            if torch.cuda.is_available():
                bb_data_val_out=bb_data_val_out.cuda()
                nb_data_val_out=nb_data_val_out.cuda()
                
            data_val_cat=torch.cat((bb_data_val,nb_data_val),dim=0)
            data_val_out_cat=torch.cat((bb_data_val_out,nb_data_val_out),dim=0)
            loss = criterion(data_val_out_cat, data_val_cat)
            valid_loss += loss.item()*data_val_cat.size(0)
        
            
        train_loss = train_loss/len(train_loader.dataset['broadband'])
        valid_loss = valid_loss/len(test_loader.dataset['broadband'])


        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTime: {:.6f}'.format(
            epoch+1, 
            train_loss,
            valid_loss,
            time.time()-start
            ))


        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            model.load_state_dict(model.state_dict())
            torch.save({'model': model.state_dict(),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'learning_rate': LR}, save_path+'/'+str(BOTTLE_SIZE)+'_CAE.pt')
            valid_loss_min = valid_loss
        if epoch==0:
            f=open(save_path+'/loss_'+str(BOTTLE_SIZE)+'_CAE.csv', "w")
        else:
            f=open(save_path+'/loss_'+str(BOTTLE_SIZE)+'_CAE.csv', "a")
        f.write(str(train_loss)+", "+str(valid_loss)+"\n")
        f.close()





