import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import shutil
import random
import torch.utils.data as data
from pdnn import pdn
import toolbox.traintestsplit as tts
from AEspeech import AEspeech
import json
import argparse
import pdb
from sklearn import metrics

"""
Traceback (most recent call last):
  File "pdnnEval.py", line 280, in <module>
    y_pred=model.forward(X_train)
  File "/cluster/ag61iwyb/pdnn.py", line 26, in forward
    x=F.leaky_relu(self.fc1(x))
  File "/home/ag61iwyb/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/ag61iwyb/.local/lib/python3.6/site-packages/torch/nn/modules/linear.py", line 91, in forward
    return F.linear(input, self.weight, self.bias)
  File "/home/ag61iwyb/.local/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in linear
    ret = torch.addmm(bias, input, weight.t())
RuntimeError: mat1 dim 1 must match mat2 dim 0

"""


PATH=os.path.dirname(os.path.abspath(__file__))

#LOAD CONFIG.JSON INFO
with open("config.json") as f:
    info = f.read()
config = json.loads(info)
UNITS=config['general']['UNITS']
UTTERS=['pataka','kakaka','pakata','papapa','petaka','tatata']
MODELS=["CAE","RAE","ALL"]
REPS=['broadband','narrowband','wvlt']
UTTERS=['pataka']


def saveFeats(model,units,rep,wav_path,utter,save_path, spk_typ):
    global UNITS    
    # load the pretrained model with 256 units and get temp./freq. rep (spec or wvlt)
    aespeech=AEspeech(model=model,units=UNITS,rep=rep) 
    
    #compute the bottleneck and error-based features from a directory with wav files inside 
    #(dynamic: one feture vector for each 500 ms frame)
    #(global i.e. static: one feture vector per utterance)
    feat_vecs=aespeech.compute_dynamic_features(wav_path)
    #     df1, df2=aespeech.compute_global_features(wav_path)
    
    with open(save_path+'/'+rep+'_'+model+'_'+spk_typ+'Feats.pickle', 'wb') as handle:
        pickle.dump(feat_vecs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return feat_vecs
            

def getFeats(model,units,rep,wav_path,utter,spk_typ):
    global PATH
    save_path=PATH+"/"+"pdSpanish/feats/"+utter+"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if os.path.isfile(save_path+'/'+rep+'_'+model+'_'+spk_typ+'Feats.pickle'):
        with open(save_path+'/'+rep+'_'+model+'_'+spk_typ+'Feats.pickle', 'rb') as handle:
            feat_vecs = pickle.load(handle)
    else:
        feat_vecs=saveFeats(model,units,rep,wav_path,utter,save_path, spk_typ)
    
    return feat_vecs
        
    
class trainData(data.Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
class testData(data.Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)   
    
    
    

if __name__=="__main__":

    PATH=os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv)!=4:
        print("python dnnTrain.py <'CAE','RAE', or 'ALL'> <'broadband' or 'narrowband' or 'wvlt'> <pd path>")
        sys.exit()        
    #TRAIN_PATH: './pdSpanish/speech/<UTTER>/'
    
    
    if sys.argv[1] in MODELS:
        mod=sys.argv[1]
    else:
        print("python pdnnEval.py <'CAE','RAE', or 'ALL'> <'broadband' or 'narrowband' or 'wvlt'> <pd path>")
        sys.exit()
    
    if sys.argv[2] in REPS:
        rep=sys.argv[2]
    else:
        print("python pdnnEval.py <'CAE','RAE', or 'ALL'> <'broadband' or 'narrowband' or 'wvlt'> <pd path>")
        sys.exit()    
        
    if sys.argv[3][0] !='/':
        sys.argv[3] = '/'+sys.argv[3]
    if sys.argv[3][-1] !='/':
        sys.argv[3] = sys.argv[3]+'/'
    
    LR=config['dnn']['LR']
    BATCH_SIZE=config['dnn']['BATCH_SIZE']
    NUM_W=config['dnn']['NUM_W']
    N_EPOCHS=config['dnn']['N_EPOCHS']
    num_pdHc_tests=config['dnn']['tst_spks']#must be even (same # of test pds and hcs per iter)
    nv=config['dnn']['val_spks']#number of validation speakers per split

#     LRs=[10**-ex for ex in np.linspace(4,7,6)]
    if rep=='narrowband' or rep=='broadband':
        NBF=config['mel_spec']['INTERP_NMELS']
    else:
        NBF=config['wavelet']['NBF']
        
    save_path=PATH+"/pdSpanish/classResults/dnn/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
        
        
    #iterate through all pd and hc speakers for a given utterance (see UTTERS for options) and using leave ten out, train a DNN
    #(see pdnn.py) and classify one by one if PD or HC.
    train_res=[]
    test_res=[]
    lr_score_opt=0
#     lr_scores=pd.DataFrame(columns=LRs,index=np.arange(1))
#     for lrItr,LR in enumerate(LRs):
    
    nt=100-(num_pdHc_tests+nv)
    for itr in range(100//num_pdHc_tests):
        trainResultsEpo_curr=[]
        testResults_curr=pd.DataFrame({utter:{'test_loss':0, 'test_acc':0, 'tstSpk_data':{}} for utter in UTTERS})

        for u_idx,utter in enumerate(UTTERS):
            if itr==0:
                pd_path=PATH+sys.argv[3]+'/'+utter+"/pd/"
                hc_path=PATH+sys.argv[3]+'/'+utter+"/hc/"   
                pds=[name for name in os.listdir(pd_path) if '.wav' in name]
                hcs=[name for name in os.listdir(hc_path) if '.wav' in name]
                pd_files=pds
                hc_files=hcs
                pds.sort()
                hcs.sort()
                spks=pds+hcs
                num_pd=len(pds)
                num_hc=len(hcs)
                total_spks=num_pd+num_hc
                rand_range=np.arange(total_spks)
                random.shuffle(rand_range)
            else:
                pd_path=PATH+sys.argv[3]+'/'+utter+"/pd/"
                hc_path=PATH+sys.argv[3]+'/'+utter+"/hc/" 

            pdFeats=getFeats(mod,UNITS,rep,pd_path,utter,'pd')
            hcFeats=getFeats(mod,UNITS,rep,hc_path,utter,'hc')
            
            #INITIATE OR RESET model
            model=pdn(UNITS+NBF)
    
            criterion=nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr = LR)

            if torch.cuda.is_available():
                print(torch.cuda.get_device_name(0))
                model.cuda()
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

            #Get test speaker features, load test. Each test iteration (10 total), update running list of speakers such that
            #already tested speakers are not tested again.
            if u_idx==0:
                pdCurrs=[pd_files[idx] for idx in random.sample(range(0,len(pd_files)),num_pdHc_tests//2)]
                hcCurrs=[hc_files[idx] for idx in random.sample(range(0,len(hc_files)),num_pdHc_tests//2)]
                pd_files=[pd for pd in pd_files if pd not in pdCurrs]
                hc_files=[hc for hc in hc_files if hc not in hcCurrs]

                pdIds=[spks.index(pdCurr) for pdCurr in pdCurrs]
                hcIds=[spks.index(hcCurr) for hcCurr in hcCurrs]

            testDict={spk:{num[i]:{'feats':[]} for num in zip(pdIds,hcIds)} for i,spk in enumerate(['pd','hc'])}

            for pdItr in pdIds:
                pdBns=pdFeats['bottleneck'][np.where(pdFeats['wav_file']==spks[pdItr])]
                pdErrs=pdFeats['error'][np.where(pdFeats['wav_file']==spks[pdItr])]
                pdTest=np.concatenate((pdBns,pdErrs),axis=1)
                testDict['pd'][pdItr]=pdTest
            for hcItr in hcIds:
                hcBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==spks[hcItr])]
                hcErrs=hcFeats['error'][np.where(hcFeats['wav_file']==spks[hcItr])]
                hcTest=np.concatenate((hcBns,hcErrs),axis=1)
                testDict['hc'][hcItr]=hcTest

            #separate validation speakers and get features
            notTestSpksPD=[spk for spk in pds if spk not in pdCurrs]
            notTestSpksHC=[spk for spk in hcs if spk not in hcCurrs]
            validsPD=[notTestSpksPD[idx] for idx in random.sample(range(0,len(notTestSpksPD)),nv//2)]
            validsHC=[notTestSpksHC[idx] for idx in random.sample(range(0,len(notTestSpksHC)),nv//2)]
            valids=validsPD+validsHC
            valIds=[spks.index(valid) for valid in valids]
            valDict={num:{'feats':[]} for num in valIds}

            #getting bottle neck features and reconstruction error for validation speakers
            for ii,val in enumerate(valids):
                vitr=valIds[ii]
                if vitr<num_pd:
                    feats=pdFeats
                else:
                    feats=hcFeats
                valBns=feats['bottleneck'][np.where(feats['wav_file']==spks[vitr])]
                valErrs=feats['error'][np.where(feats['wav_file']==spks[vitr])]
                valTest=np.concatenate((valBns,valErrs),axis=1)
                valDict[vitr]=valTest
                
            trainResults_epo= pd.DataFrame({'train_loss':0.0, 'train_acc':0.0,'val_loss':0.0, 'val_acc':0.0}, index=np.arange(N_EPOCHS))
            
            for epoch in range(N_EPOCHS):  
                train_loss=0.0
                train_acc=0.0

                #TRAIN dnn for each speaker individually.             
                for trainItr in rand_range:   
                    if trainItr in np.concatenate((pdIds,hcIds,valIds)):
                        continue

                    if trainItr<num_pd:
                        trainIndc=1
                        trainOpp=0
                        trainFeats=pdFeats
                    else:
                        trainIndc=0
                        trainOpp=1
                        trainFeats=hcFeats 

                    #getting bottle neck features and reconstruction error for particular training speaker
                    bns=trainFeats['bottleneck'][np.where(trainFeats['wav_file']==spks[trainItr])]
                    
                    errs=trainFeats['error'][np.where(trainFeats['wav_file']==spks[trainItr])]
                    xTrain=np.concatenate((bns,errs),axis=1)
                    #standardize data
                    xTrain=(xTrain-np.min(xTrain))/(np.max(xTrain)-np.min(xTrain))
                    yTrain=np.vstack((np.ones((xTrain.shape[0]))*trainIndc,np.ones((xTrain.shape[0]))*trainOpp)).T
                    train_data=trainData(torch.FloatTensor(xTrain), torch.FloatTensor(yTrain))
                    train_loader=torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_W)
                    start=time.time()

                    train_loss_curr=0.0
                    if len(train_loader)>0:
                        #TRAINING ON SNIPPETS OF SPEAKER UTTERANCES
                        model.train() # prep model for training
                        for X_train, y_train in train_loader:

                            #clear the gradients of all optimized variables
                            optimizer.zero_grad()

                            X_train=X_train.float()                
                            y_train=y_train.float()

                            if torch.cuda.is_available():
                                X_train,y_train=X_train.cuda(),y_train.cuda()

                            y_pred=model.forward(X_train)
                            #Find difference in probability of PD v. HC for all segments.
                            if torch.cuda.is_available():
                                y_pred=y_pred.cuda()

                            loss=criterion(y_pred, y_train)

                            loss.backward()
                            optimizer.step()
                            train_loss_curr += loss.item()*y_train.size(0)

                        #tally up train loss total for given speaker
                        train_loss+=train_loss_curr/len(train_loader.dataset)                           

                #Record train loss at end of each epoch (divide by number of train patients - 100-tst-val).
                trainResults_epo.iloc[epoch]['train_loss']=train_loss/nt
#                     print('Epoch: {} \nTraining Loss: {:.6f} Training Accuracy: {:.2f} \tTime: {:.6f}\n'.format(
#                     epoch+1,
#                     train_loss/nt,
#                     train_acc/nt,
#                     time.time()-start
#                     ))         
                
    
                #Iterate through thresholds and choose one that yields best validation acc.
                #Iterate through all num_tr training patients and classify based off difference in probability of PD/HC
                max_val_acc=0
                tr_acc=0
                num_tr=0
                if epoch==N_EPOCHS-1:
                    threshes=np.arange(-100,100)
                else:
                    threshes=[0]
                
                for thresh in threshes:
                    thresh=thresh/100
                    for trainItr in rand_range:   
                        if trainItr in np.concatenate((pdIds,hcIds,valIds)):
                            continue

                        if trainItr<num_pd:
                            trainIndc=1
                            trainOpp=0
                            trainFeats=pdFeats
                        else:
                            trainIndc=0
                            trainOpp=1
                            trainFeats=hcFeats

                        #getting bottle neck features and reconstruction error for particular training speaker
                        bns=trainFeats['bottleneck'][np.where(trainFeats['wav_file']==spks[trainItr])]
                        errs=trainFeats['error'][np.where(trainFeats['wav_file']==spks[trainItr])]
                        xTrain=np.concatenate((bns,errs),axis=1)
                        xTrain=(xTrain-np.min(xTrain))/(np.max(xTrain)-np.min(xTrain))
                        yTrain=np.vstack((np.ones((xTrain.shape[0]))*trainIndc,np.ones((xTrain.shape[0]))*trainOpp)).T

                        train_data=testData(torch.FloatTensor(xTrain))
                        train_loader=torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_W)
                        y_pred_tag=[]
                        model.eval()
                        with torch.no_grad():
                            for X_tr in train_loader:
                                yTr=np.vstack((np.ones((X_tr.shape[0]))*trainIndc,np.ones((X_tr.shape[0]))*trainOpp)).T
                                if torch.cuda.is_available():
                                    X_tr=X_tr.cuda()

                                y_tr_pred = model.forward(X_tr)

                                #Find difference in probability of PD v. HC for all segments. 
                                y_pred_tag.extend((y_tr_pred[:,0]-y_tr_pred[:,1]).cpu().detach().numpy())

                        #Wlog, if difference greater than 0 occurs more and speaker is PD, than identification is correct (1=PD,0=HC).
                        y_pred_tag=np.array(y_pred_tag)
                        if len(y_pred_tag)>0:
                            num_tr+=1

                            """THREE CLASSIFIERS BELOW"""
    #                         #Classification correct if (wlog) median prob difference indicates correct spk type.
    #                         if trainIndc==1 and np.median(y_pred_tag)>0:
    #                             tr_acc+=1
    #                         elif trainIndc==0 and np.median(y_pred_tag)<0:
    #                             tr_acc+=1               

        #                    #Classification correct if more frame probability differences indicate correct spk type. 
        #                     if trainIndc==1 and (len(y_pred_tag[np.where(y_pred_tag>0)]) >= len(y_pred_tag[np.where(y_pred_tag<0)])):
        #                         tr_acc+=1
        #                     elif trainIndc==0 and (len(y_pred_tag[np.where(y_pred_tag<0)]) >= len(y_pred_tag[np.where(y_pred_tag>0)])):
        #                         tr_acc+=1

                           #Classification is based off percent of frames classified correctly.
                            if trainIndc==1 :
                                tr_acc+=len(y_pred_tag[np.where(y_pred_tag>0)])/len(y_pred_tag)
                            elif trainIndc==0:
                                tr_acc+=len(y_pred_tag[np.where(y_pred_tag<0)])/len(y_pred_tag)
                        else:
                            continue

    #                     trainResults_epo.iloc[epoch]['train_acc']=tr_acc/num_tr

                    if np.mod(epoch,1)==0:
                        #Validate at end of each 1 epochs for nv speakers
                        val_loss=0.0
                        val_acc=0
                        num_val=0
                        for vid in valDict.keys():
                            if vid<num_pd:
                                indc=1
                                opp=0
                            else:
                                indc=0
                                opp=1
                            y_pred_tag=[]
                            xVal=valDict[vid]
                            xVal=(xVal-np.min(xVal))/(np.max(xVal)-np.min(xVal))
                            test_data=testData(torch.FloatTensor(xVal))
                            test_loader=torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_W, drop_last=False, shuffle=True) 

                            model.eval()
                            with torch.no_grad():
                                for X_test in test_loader:
                                    yTest=np.vstack((np.ones((X_test.shape[0]))*indc,np.ones((X_test.shape[0]))*opp)).T
                                    if torch.cuda.is_available():
                                        X_test=X_test.cuda()

                                    y_test_pred = model.forward(X_test)

                                    #Find difference in probability of PD v. HC for all segments. 
                                    y_pred_tag.extend((y_test_pred[:,0]-y_test_pred[:,1]).cpu().detach().numpy())
                                    if torch.cuda.is_available():
                                        loss = criterion(y_test_pred, torch.from_numpy(yTest).cuda().float())
                                    else:
                                        loss = criterion(y_test_pred, torch.from_numpy(yTest).float())
                                    val_loss+=loss.item()*X_test.size(0)


                            val_loss=val_loss/len(test_loader.dataset)
                            y_pred_tag=np.array(y_pred_tag)
                            if len(y_pred_tag)>0:
                                num_val+=1

                            """THREE CLASSIFIERS BELOW"""
            #                     #Classification correct if (wlog) median prob difference indicates correct spk type.
            #                     if indc==1 and np.median(y_pred_tag)>=0:
            #                         val_acc+=1
            #                     elif indc==0 and np.median(y_pred_tag)<=0:
            #                         val_acc+=1

            #                    #Classification correct if more frame probability differences indicate correct spk type. 
            #                     if indc==1 and (len(y_pred_tag[np.where(y_pred_tag>0)]) >= len(y_pred_tag[np.where(y_pred_tag<0)])):
            #                         val_acc+=1
            #                     elif indc==0 and (len(y_pred_tag[np.where(y_pred_tag<0)]) >= len(y_pred_tag[np.where(y_pred_tag>0)])):
            #                         val_acc+=1

                           #Classification is based off percent of frames classified correctly.
                            if indc==1 :
                                val_acc+=len(y_pred_tag[np.where(y_pred_tag>thresh)])/len(y_pred_tag)
                            elif indc==0:
                                val_acc+=len(y_pred_tag[np.where(y_pred_tag<thresh)])/len(y_pred_tag)
                            else:
                                continue
                    
                    if epoch==(N_EPOCHS-1):
                        if val_acc/num_val>max_val_acc:
                            max_val_acc=val_acc/num_val
                            opt_thresh=thresh
                            trainResults_epo.iloc[epoch]['train_acc']=tr_acc/num_tr
                            trainResults_epo.iloc[epoch]['val_loss']=val_loss/num_val
                            trainResults_epo.iloc[epoch]['val_acc']=val_acc/num_val
                    else:
                        trainResults_epo.iloc[epoch]['train_acc']=tr_acc/num_tr
                        trainResults_epo.iloc[epoch]['val_loss']=val_loss/num_val
                        trainResults_epo.iloc[epoch]['val_acc']=val_acc/num_val
                        
                print('Train Loss: {:.6f} Train Accuracy: {}\nValidation Loss: {:.6f} Validation Accuracy: {}\n'.format(
                train_loss/num_tr,  
                tr_acc/num_tr,
                val_loss/num_val,
                val_acc/num_val,
                ))      
                    
#                     if val_loss/num_val<=valid_loss_min:
#                         torch.save(model, save_path+mod+'_'+rep+'_model.pt')
#                         valid_loss_min = val_loss/num_val
            trainResultsEpo_curr.append(trainResults_epo)



            #AFTER MODEL TRAINED FOR ALL SPEAKERS TEST MODEL ON LEFT OUT SPEAKERS  
            test_loss=0.0
            test_acc=0.0
            testResults_curr[utter]['tstSpk_data']={key:{} for spk in ['pd','hc'] for key in testDict[spk].keys()}
            num_tst=0
            for spkItr,spk in enumerate(['pd','hc']):
                dic=testDict[spk]

                for tstId in dic.keys():
                    if tstId<num_pd:
                        indc=1
                        opp=0
                    else:
                        indc=0
                        opp=1
                    y_pred_tag=[]  
                    test_loss_curr=0
                    xTest=dic[tstId]
                    xTest=(xTest-np.min(xTest))/(np.max(xTest)-np.min(xTest))
                    test_data=testData(torch.FloatTensor(xTest))
                    test_loader=torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_W, drop_last=False, shuffle=True)  

                    model.eval()
                    with torch.no_grad():
                        for X_test in test_loader:
                            yTest=np.vstack((np.ones((X_test.shape[0]))*indc,np.ones((X_test.shape[0]))*opp)).T
                            if torch.cuda.is_available():
                                X_test=X_test.cuda()

                            y_test_pred = model.forward(X_test)

                            #Find difference in probability of PD v. HC for all segments. 
                            y_pred_tag.extend((y_test_pred[:,0]-y_test_pred[:,1]).cpu().detach().numpy())

                            if torch.cuda.is_available():
                                loss = criterion(y_test_pred, torch.from_numpy(yTest).cuda().float())
                            else:
                                loss = criterion(y_test_pred, torch.from_numpy(yTest).float())
                            test_loss_curr+=loss.item()*X_test.size(0)

                    #accuracy determined on majority rule (wlog, if more frames yield probability differences greater than 0,
                    #and if speaker is PD than classification assessment is correct (1=>PD,0=>HC).
                    test_loss+=test_loss_curr/len(test_loader.dataset)
                    y_pred_tag=np.array(y_pred_tag)
                    
                    """THREE CLASSIFIERS BELOW"""
    #                     #Classification correct if (wlog) median prob difference indicates correct spk type.
    #                     if len(y_pred_tag)>0:
    #                         num_tst+=1

    #                         if indc==1 and np.median(y_pred_tag)>0:
    #                             test_acc+=1
    #                         elif indc==0 and np.median(y_pred_tag)<0:
    #                             test_acc+=1
    #                    #Classification correct if more frame probability differences indicate correct spk type. 
    #                     if indc==1:
    #                         if (len(y_pred_tag[np.where(y_pred_tag>0)]) >= len(y_pred_tag[np.where(y_pred_tag<0)])):
    #                             test_acc+=1
    #                     if indc==0:
    #                         if (len(y_pred_tag[np.where(y_pred_tag<0)]) >= len(y_pred_tag[np.where(y_pred_tag>0)])):
    #                             test_acc+=1
                    
                    if len(y_pred_tag)>0:
                        num_tst+=1
                   #Classification is based off percent of frames classified correctly.
                    if indc==1 :
                        test_acc+=len(y_pred_tag[np.where(y_pred_tag>opt_thresh)])/len(y_pred_tag)
                    elif indc==0:
                        test_acc+=len(y_pred_tag[np.where(y_pred_tag<opt_thresh)])/len(y_pred_tag)
                    else:
                        continue
                    #Store raw scores for each test speaker (probability of PD and HC as output by dnn) for ROC.
                    testResults_curr[utter]['tstSpk_data'][tstId]=y_test_pred.cpu().detach().numpy()

            #Store and report loss and accuracy for batch of test speakers.            
            testResults_curr[utter]['test_loss'],testResults_curr[utter]['test_acc']=test_loss/num_tst,test_acc/num_tst
            print('\nTest Loss: {:.3f} \tTest Acc: {:.3f} '.format(
                        test_loss/num_tst,
                        test_acc/num_tst
                ))

        trainResults_curr=pd.concat((trainResultsEpo_curr),keys=(np.arange(len(UTTERS))))      
        train_res.append(trainResults_curr)
        test_res.append(testResults_curr)


    trainResults=pd.concat(train_res,keys=(np.arange(itr+1)))
    testResults=pd.concat(test_res,keys=(np.arange(itr+1)))
        
#         #compare acc for all lrs and save highest
#         lr_score=0
#         for item in testResults:
#             for index in testResults.index:
#                 if index[1] == 'test_acc':
#                     lr_score+=testResults[item][index]/(10*len(UTTERS))
#         if lr_score>lr_score_opt:
#             trainResults.to_pickle(save_path+mod+'_'+rep+"TrainResults.pkl")
#             testResults.to_pickle(save_path+mod+'_'+rep+"TestResults.pkl") 
#             lr_score_opt=lr_score
#         lr_scores.iloc[0][LRs[lrItr]]=lr_score
#         lr_scores.to_csv(save_path+mod+'_'+rep+"lrResults.csv")

    trainResults.to_pickle(save_path+mod+'_'+rep+"_trainResults.pkl")
    testResults.to_pickle(save_path+mod+'_'+rep+"_testResults.pkl")






