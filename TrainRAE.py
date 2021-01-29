from SpecDatset import SpecDataset
import time
import torch
import numpy as np
import pandas as pd
import os
import sys
from RAE import RAEn
from wvRAE import wvRAEn
import toolbox.traintestsplit as tts
import json
import argparse


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
    if len(sys.argv)!=4:
        print("python TrainRAE.py <bottleneck_sizes> <rep path> <chkpt_path>")
        sys.exit()
    #repPath: "./tedx_spanish_corpus/reps/'wvlt or broadband or narrowband'/train/" 
    #chkpt_path: "./pts/checkpoints/" 
    
    if sys.argv[3]:
        if sys.argv[3][0] !='/':
            chkpt_path = PATH+'/'+sys.argv[3]
        if sys.argv[3][-1] !='/':
            chkpt_path = PATH+sys.argv[3]+'/'
    else:
        chkpt_path=''
        
    if sys.argv[2][0] !='/':
        sys.argv[2] = '/'+sys.argv[2]
    if sys.argv[2][-1] !='/':
        sys.argv[2] = sys.argv[2]+'/'

    path_rep=PATH+sys.argv[2]
    if 'wvlt' in sys.argv[2]:
        rep_typ='wvlt'
    elif 'narrowband' in sys.argv[2]:
        rep_typ='narrowband'
    elif 'broadband' in sys.argv[2]:
        rep_typ='broadband'
    else:
        print("Please correct directory path input...")
        sys.exit()
    
    if not os.path.exists(path_rep+'/train/') or not os.path.exists(path_rep+'/test/'):
        split=tts.trainTestSplit(path_rep,file_type='.npy', tst_perc=0.1, gen_bal=1)
        split.fileTrTstSplit()

    elif len(os.listdir(path_rep+'/train/')) == 0:  
        split=tts.trainTestSplit(path_rep,file_type='.npy', tst_perc=0.1, gen_bal=1)
        split.fileTrTstSplit()
   
    with open("config.json") as f:
        data = f.read()
    config = json.loads(data)
    
    FS=config['general']['FS']
    NUM_W=config['RAE']['NUM_W']
    N_EPOCHS=config['RAE']['N_EPOCHS']
    NTRAIN=config['RAE']['NTRAIN']
    NVAL=config['RAE']['NVAL']
    LR=config['RAE']['LR']
    BATCH_SIZE=config['RAE']['BATCH_SIZE']
    
    PATH_TRAIN=path_rep+"/train/"
    PATH_TEST=path_rep+"/test/"
    BOTTLE_SIZE=int(sys.argv[1])
    SCALERS = pd.read_csv("scales.csv")
    if rep_typ=='narrowband' or rep_typ=='broadband':
        MIN_SCALER= float(SCALERS['Min '+rep_typ+' Scale']) #MIN value of total energy.
        MAX_SCALER= float(SCALERS['Max '+rep_typ+' Scale'])  #MAX value of total energy.
    else:
        MIN_SCALER=-10.429498640058068
        MAX_SCALER=10.460396126590783
    
    
    train=SpecDataset(PATH_TRAIN)
    test=SpecDataset(PATH_TEST)
   
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, drop_last=True, num_workers=NUM_W)
    test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, drop_last=True, num_workers=NUM_W)
    
    if rep_typ=='broadband' or rep_typ=='narrowband':
        model=RAEn(BOTTLE_SIZE)
    elif rep_typ=='wvlt':
        model=wvRAEn(BOTTLE_SIZE)
    
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        model.cuda()
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    
    save_path = PATH+"/pts/"+rep_typ+'/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        
    criterion = torch.nn.MSELoss()    
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    epochs=np.arange(N_EPOCHS)
         
    if os.path.isfile(save_path+str(BOTTLE_SIZE)+'_RAE.pt'):
        model, optimizer, epoch = load_checkpoint(save_path+str(BOTTLE_SIZE)+'_RAE.pt', model,
                                                      optimizer)
        epoch += 1  # next iteration is iteration + 1
        if epoch>=N_EPOCHS:
            if rep_typ=='broadband' or rep_typ=='narrowband':
                model=CAEn(BOTTLE_SIZE)
            elif rep_typ=='wvlt':
                model=wvCAEn(BOTTLE_SIZE)
            epochs=np.arange(N_EPOCHS)
        else:
            epochs=np.arange(epoch,N_EPOCHS)
    
    
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        model.cuda()
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    valid_loss_min = np.Inf # set initial "min" to infinity
    avg_train_losses=[]
    avg_valid_losses=[]
    
    for epoch in range(N_EPOCHS):
        start=time.time()
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
        cdata=0
        model.train() # prep model for training
        for data in train_loader:

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
#             if rep_typ=='broadband' or rep_typ=='narrowband':
            data=standard(data, MIN_SCALER, MAX_SCALER)

            data=data.float()

            if torch.cuda.is_available():
                data=data.cuda()

            data_out, bottle=model.forward(data)

            if torch.cuda.is_available():
                data_out=data_out.cuda()

            loss = criterion(data_out, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

            cdata+=1

            #if cdata==NTRAIN:
            #    break
        cdata=0
        ######################    
        # validate the model #
        ######################

        model.eval() # prep model for evaluation
        for data_val in test_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
        if rep_typ=='broadband' or rep_typ=='narrowband':
                data_val=standard(data_val, MIN_SCALER, MAX_SCALER)

            data_val=data_val.float()
            if torch.cuda.is_available():
                data_val=data_val.cuda()

            data_val_out, bottle_val=model.forward(data_val)

            if torch.cuda.is_available():
                data_val_out=data_val_out.cuda()

            # calculate the loss
            loss = criterion(data_val_out, data_val)
            # update running validation loss 
            valid_loss += loss.item()*data.size(0)
            cdata+=1
            #if cdata==NVAL:
            #    break
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(test_loader.dataset)


        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

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
#             torch.save(model.state_dict(), save_path+'/'+str(BOTTLE_SIZE)+'_RAE.pt')
            model.load_state_dict(model.state_dict())
            torch.save({'model': model.state_dict(),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'learning_rate': LR}, save_path+'/'+str(BOTTLE_SIZE)+'_RAE.pt')
            valid_loss_min = valid_loss
        if epoch==0:
            f=open(save_path+'/loss_'+str(BOTTLE_SIZE)+'_RAE.csv', "w")
        else:
            f=open(save_path+'/loss_'+str(BOTTLE_SIZE)+'_RAE.csv', "a")
        f.write(str(train_loss)+", "+str(valid_loss)+"\n")
        f.close()
        





