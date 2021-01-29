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
from clpdnn import clpdnn
import toolbox.traintestsplit as tts
from clpAEspeech import AEspeech
import json
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import argparse
import pdb
from sklearn import metrics

PATH=os.path.dirname(os.path.abspath(__file__))
#LOAD CONFIG.JSON INFO
with open("clpConfig.json") as f:
    info = f.read()
config = json.loads(info)
UNITS=config['general']['UNITS']
UTTERS=['bola','choza','chuzo','coco','gato','jugo','mano','papa','susi']
MODELS=["CAE","RAE","ALL"]
REPS=['broadband','narrowband','wvlt']



def saveFeats(model,units,rep,wav_path,utter,save_path,spk_typ):
    global UNITS    
    # load the pretrained model with 256 units and get temp./freq. rep (spec or wvlt)
    aespeech=AEspeech(model=model,units=UNITS,rep=rep) 
    
    #compute the bottleneck and error-based features from a directory with wav files inside 
    #(dynamic: one feture vector for each 500/50 ms frame)
    #(global i.e. static: one feture vector per utterance)
    feat_vecs=aespeech.compute_dynamic_features(wav_path)
    #     df1, df2=aespeech.compute_global_features(wav_path)
    
    with open(save_path+'/'+rep+'_'+model+'_'+spk_typ+'dnnFeats.pickle', 'wb') as handle:
        pickle.dump(feat_vecs, handle, protocol=4)
    
    return feat_vecs
            

def getFeats(model,units,rep,wav_path,utter,spk_typ):
    global PATH
    save_path=PATH+"/"+"clpSpanish/feats/"+utter+"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if os.path.isfile(save_path+'/'+rep+'_'+model+'_'+spk_typ+'dnnFeats.pickle'):
        with open(save_path+'/'+rep+'_'+model+'_'+spk_typ+'dnnFeats.pickle', 'rb') as handle:
            feat_vecs = pickle.load(handle)
    else:
        feat_vecs=saveFeats(model,units,rep,wav_path,utter,save_path, spk_typ)
    
    return feat_vecs


def featAgg(mod,rep,spk_path, ef=0):
    global UNITS
    global UTTERS
    
    allClpNames=[]
    allHcNames=[]
    for u_idx,utter in enumerate(UTTERS):
        clp_path=spk_path+'/'+utter+"/clp/"
        hc_path=spk_path+'/'+utter+"/hc/"   
        clpNames=list(np.unique([name.split('_')[0] for name in os.listdir(clp_path) if '.wav' in name]))
        hcNames=list(np.unique([name.split('_')[0] for name in os.listdir(hc_path) if '.wav' in name]))
        allClpNames.extend(clpNames)
        allHcNames.extend(hcNames)
    clpNames=list(np.unique(allClpNames))
    hcNames=list(np.unique(allHcNames))
    num_clps=len(clpNames)
    num_hcs=len(hcNames)
    spkNames=clpNames+hcNames
    
    #aggregate all utterance data per speaker together.
    master_clpIds=np.arange(0,num_clps)
    master_hcIds=np.arange(num_clps,num_clps+num_hcs)
    
    spkDict={spk:{} for spk in ['clp','hc']}
    spkDict['clp']={num:[] for num in master_clpIds}
    spkDict['hc']={num:[] for num in master_hcIds}
    reps=rep
    for u_idx,utter in enumerate(UTTERS):
        clp_path=spk_path+'/'+utter+"/clp/"
        hc_path=spk_path+'/'+utter+"/hc/"   
        clpNames_curr=[name for name in os.listdir(clp_path) if '.wav' in name]
        hcNames_curr=[name for name in os.listdir(hc_path) if '.wav' in name]
        clpNames_curr.sort()
        hcNames_curr.sort()
        clp_rng=np.arange(len(clpNames_curr))
        hc_rng=np.arange(len(hcNames_curr))
        clpIds_curr=[spkNames.index(clpName.split('_')[0]) for clpName in clpNames_curr]
        hcIds_curr=[spkNames.index(hcName.split('_')[0]) for hcName in hcNames_curr]
        
        if ef==0:
            clpFeats=getFeats(mod,UNITS,rep,clp_path,utter,'clp')
            hcFeats=getFeats(mod,UNITS,rep,hc_path,utter,'hc')
            for rItr,pItr in zip(clp_rng,clpIds_curr):
                clpBns=clpFeats['bottleneck'][np.where(clpFeats['wav_file']==clpNames_curr[rItr])]
                clpErrs=clpFeats['error'][np.where(clpFeats['wav_file']==clpNames_curr[rItr])]
                if len(spkDict['clp'][pItr])==0:
                    spkDict['clp'][pItr]=np.concatenate((clpBns,clpErrs),axis=1)
                else:
                    spkDict['clp'][pItr]=np.concatenate((spkDict['clp'][pItr],np.concatenate((clpBns,clpErrs),axis=1)),axis=0)
            for rItr,hItr in zip(hc_rng,hcIds_curr):
                hcBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==hcNames_curr[rItr])]
                hcErrs=hcFeats['error'][np.where(hcFeats['wav_file']==hcNames_curr[rItr])]
                if len(spkDict['hc'][hItr])==0:
                    spkDict['hc'][hItr]=np.concatenate((hcBns,hcErrs),axis=1)
                else:
                    spkDict['hc'][hItr]=np.concatenate((spkDict['hc'][hItr],np.concatenate((hcBns,hcErrs),axis=1)),axis=0)       
        else:
            for rIdx,rep in enumerate(reps):
                clpFeats=getFeats(mod,UNITS,rep,clp_path,utter,'clp')
                hcFeats=getFeats(mod,UNITS,rep,hc_path,utter,'hc')
                
                for rItr,pItr in zip(clp_rng,clpIds_curr):
                    clpBns=clpFeats['bottleneck'][np.where(clpFeats['wav_file']==clpNames_curr[rItr])]
                    clpErrs=clpFeats['error'][np.where(clpFeats['wav_file']==clpNames_curr[rItr])]
                    if len(spkDict['clp'][pItr])==0:
                        spkDict['clp'][pItr]=np.concatenate((clpBns,clpErrs),axis=1)
                    else:
                        spkDict['clp'][pItr]=np.concatenate((spkDict['clp'][pItr],np.concatenate((clpBns,clpErrs),axis=1)),axis=0)
                for rItr,hItr in zip(hc_rng,hcIds_curr):
                    hcBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==hcNames_curr[rItr])]
                    hcErrs=hcFeats['error'][np.where(hcFeats['wav_file']==hcNames_curr[rItr])]
                    if len(spkDict['hc'][hItr])==0:
                        spkDict['hc'][hItr]=np.concatenate((hcBns,hcErrs),axis=1)
                    else:
                        spkDict['hc'][hItr]=np.concatenate((spkDict['hc'][hItr],np.concatenate((hcBns,hcErrs),axis=1)),axis=0)
    
    
    return spkDict, clpNames,hcNames

    
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
        print("python clpdnnEval.py <'CAE','RAE', or 'ALL'> <broadband, narrowband, wvlt, early_fuse, mc_fuse> <clp path>")
        sys.exit()        
    #TRAIN_PATH: './clpSpanish/speech/'
    
    
    if sys.argv[1] in MODELS:
        mod=sys.argv[1]
    else:
        print("python clpdnnEval.py <'CAE','RAE', or 'ALL'> <broadband, narrowband, wvlt, early_fuse, mc_fuse> <clp path>")
        sys.exit()
    
    if sys.argv[2] not in ['broadband', 'narrowband', 'wvlt', 'early_fuse', 'mc_fuse']:
        print("python clpdnnEval.py <'CAE','RAE', or 'ALL'> <broadband, narrowband, wvlt, early_fuse, mc_fuse> <clp path>")
        sys.exit()
    else:
        rep_typ=sys.argv[2]
    
    if sys.argv[3][0] !='/':
        sys.argv[3] = '/'+sys.argv[3]
    if sys.argv[3][-1] !='/':
        sys.argv[3] = sys.argv[3]+'/'
    spk_path=PATH+sys.argv[3]
    
    LR=config['dnn']['LR']
    BATCH_SIZE=config['dnn']['BATCH_SIZE']
    NUM_W=config['dnn']['NUM_W']
    N_EPOCHS=config['dnn']['N_EPOCHS']
    num_clpHc_tests=config['dnn']['tst_spks']#must be even (same # of test clps and hcs per iter)
    nv=config['dnn']['val_spks']#number of validation speakers per split
    num_iters=config['dnn']['num_iters']
    
#     LRs=[10**-ex for ex in np.linspace(4,7,6)]
    save_path=PATH+"/clpSpanish/classResults/dnn/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    ef=0
    if rep_typ in ['broadband','narrowband','wvlt']:
        rep=rep_typ
        if rep=='broadband' or rep=='narrowband':
            NBF=config['mel_spec']['INTERP_NMELS']
        elif rep=='wvlt':
            NBF=config['wavelet']['NBF']
    elif rep_typ=='mc_fuse':
        rep='mc_fuse'
        NBF=2*config['mel_spec']['INTERP_NMELS']
    else:
        rep=['broadband','narrowband']
        NBF=config['mel_spec']['INTERP_NMELS']
        ef=1
    
    spkDict,clpNames,hcNames=featAgg(mod,rep,spk_path,ef)
    
    clpNames.sort()
    hcNames.sort()
    spks=clpNames+hcNames
    spks.sort()
    num_spks=len(spks)
    num_clp=len(clpNames)
    num_hc=len(hcNames)
    testResults=pd.DataFrame({splItr:{'test_loss':0, 'test_acc':0, 'class_report':{}, 'tstSpk_data':{}} for splItr in range(num_iters)})     
    train_res=[]
    #split data into training and test with multiple iterations (evenly split CLP:HC)
    #Due to uneven CLP/HC populations (135:58), we sample 15 test patients of each class over 9 iterations.
    #CLP sampled w/o replacement, HC sampled w/o replacement until all patients classified, then total population replaced and resampled.
    clp_files=clpNames
    hc_files=hcNames
    for itr in range(num_iters):
        #RESET model
        model=clpdnn(UNITS+NBF)
        criterion=nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = LR)

        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
            model.cuda()
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
        

        #Get test speaker features
        clpCurrs=[clp_files[idx] for idx in random.sample(range(0,len(clp_files)),num_clpHc_tests//2)]
        clp_files=[clp for clp in clp_files if clp not in clpCurrs]
        hcCurrs=[hc_files[idx] for idx in random.sample(range(0,len(hc_files)),num_clpHc_tests//2)]
        hc_files=[hc for hc in hc_files if hc not in hcCurrs]
       
        clpIds=[spks.index(clpCurr) for clpCurr in clpCurrs]
        hcIds=[spks.index(hcCurr) for hcCurr in hcCurrs]

        testDict={spk:{} for spk in ['clp','hc']}
        testDict['clp']={num:[] for num in clpIds}
        testDict['hc']={num:[] for num in hcIds}
        
        for clpItr in clpIds:
            testDict['clp'][clpItr]=spkDict['clp'][clpItr]
        for hcItr in hcIds:
            testDict['hc'][hcItr]=spkDict['hc'][hcItr]
        
        #Separate 'nv' (equal number of clp/hc) Validation speakers and get features
        notTestSpksCLP=[spk for spk in clpNames if spk not in clpCurrs]
        notTestSpksHC=[spk for spk in hcNames if spk not in hcCurrs]
        validsCLP=[notTestSpksCLP[idx] for idx in random.sample(range(0,len(notTestSpksCLP)),nv//2)]
        validsHC=[notTestSpksHC[idx] for idx in random.sample(range(0,len(notTestSpksHC)),nv//2)]
        valids=validsCLP+validsHC
        valIds=[spks.index(valid) for valid in valids]
        valDict={num:[] for num in valIds}

        #getting bottle neck features and reconstruction error for validation speakers
        for ii,val in enumerate(valids):
            vitr=valIds[ii]
            if vitr<num_clp:
                spk_typ='clp'
            else:
                spk_typ='hc'
            valDict[vitr]=spkDict[spk_typ][vitr]

        trainResults_epo= pd.DataFrame({'train_loss':0.0, 'train_acc':0.0,'val_loss':0.0, 'val_acc':0.0}, index=np.arange(N_EPOCHS))

        #enusre same number of train clps/hcs
        num_pats_tr=num_spks-num_clpHc_tests-nv
        notTestValSpksCLP=[spk for spk in clpNames if spk not in clpCurrs+validsCLP]
        notTestValSpksHC=[spk for spk in hcNames if spk not in hcCurrs+validsHC]

        clptr_rand_range=[spks.index(trid) for trid in notTestValSpksCLP]
        hctr_rand_range=[spks.index(trid) for trid in notTestValSpksHC]
        random.shuffle(clptr_rand_range)
        random.shuffle(hctr_rand_range)
        tr_rand_range=clptr_rand_range[:num_pats_tr//2]+hctr_rand_range[:num_pats_tr//2]
        
        for epoch in range(N_EPOCHS):  
            train_loss=0.0
            train_acc=0.0
            
            #TRAIN dnn for each speaker individually.             
            for trainItr in tr_rand_range:   
                if trainItr<num_clp:
                    trainIndc=1
                    trainOpp=0
                    spk_typ='clp'
                else:
                    trainIndc=0
                    trainOpp=1
                    spk_typ='hc'

                #getting bottle neck features and reconstruction error for particular training speaker
                xTrain=spkDict[spk_typ][trainItr]
                xTrain=(xTrain-np.min(xTrain))/(np.max(xTrain)-np.min(xTrain))
                yTrain=np.vstack((np.ones((xTrain.shape[0]))*trainIndc,np.ones((xTrain.shape[0]))*trainOpp)).T
                train_data=trainData(torch.FloatTensor(xTrain), torch.FloatTensor(yTrain))
                train_loader=torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_W)

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
                        #Find difference in probability of CLP v. HC for all segments.
                        if torch.cuda.is_available():
                            y_pred=y_pred.cuda()

                        loss=criterion(y_pred, y_train)

                        loss.backward()
                        optimizer.step()
                        train_loss_curr += loss.item()*y_train.size(0)

                    #tally up train loss total for given speaker
                    train_loss+=train_loss_curr/len(train_loader.dataset)                           

            #Record train loss at end of each epoch (divide by number of train patients).
            trainResults_epo.iloc[epoch]['train_loss']=train_loss/(len(tr_rand_range)-(num_clpHc_tests+nv))
    
            
            if np.mod(epoch+1,1)==0 or epoch==0:
                #Iterate through all num_tr training patients and classify based off difference in probability of CLP/HC
                num_tr=0
                y_pred_tag=[]
                for trainItr in tr_rand_range:   
                    if trainItr in np.concatenate((clpIds,hcIds,valIds)):
                        continue

                    if trainItr<num_clp:
                        trainIndc=1
                        trainOpp=0
                        spk_typ='clp'
                    else:
                        trainIndc=0
                        trainOpp=1
                        spk_typ='hc'

                    #getting bottle neck features and reconstruction error for particular training speaker
                    
                    xTrain=spkDict[spk_typ][trainItr]
                    xTrain=(xTrain-np.min(xTrain))/(np.max(xTrain)-np.min(xTrain))
                    yTrain=np.vstack((np.ones((xTrain.shape[0]))*trainIndc,np.ones((xTrain.shape[0]))*trainOpp)).T
                    y_pred_tag_curr=[]
                    train_data=testData(torch.FloatTensor(xTrain))
                    train_loader=torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_W)
                    model.eval()
                    with torch.no_grad():
                        for X_tr in train_loader:
                            yTr=np.vstack((np.ones((X_tr.shape[0]))*trainIndc,np.ones((X_tr.shape[0]))*trainOpp)).T
                            if torch.cuda.is_available():
                                X_tr=X_tr.cuda()

                            y_tr_pred = model.forward(X_tr)

                            #Find difference in probability of CLP v. HC for all segments. 
                            y_pred_tag_curr.extend((y_tr_pred[:,0]-y_tr_pred[:,1]).cpu().detach().numpy())
                    
                    if num_tr==0:
                        indcs_vec=np.ones(len(y_pred_tag_curr))*trainIndc
                        num_tr=1
                    else:
                        indcs_vec=np.concatenate((indcs_vec,np.ones(len(y_pred_tag_curr))*trainIndc))
                        num_tr+=1
                    y_pred_tag.extend(y_pred_tag_curr)
                
                y_pred_tag=np.array(y_pred_tag).reshape(-1,1)
                clf = SGDClassifier(loss="hinge", penalty="l2")
                clf.fit(y_pred_tag, indcs_vec) 
                tr_acc=clf.score(y_pred_tag,indcs_vec)
                if epoch==N_EPOCHS-1:
                    calibrator=CalibratedClassifierCV(clf, cv='prefit')
                    modCal=calibrator.fit(y_pred_tag, indcs_vec)

            
                #Validate at end of each 1 epochs for nv speakers
                val_loss=0.0
                num_val=0
                y_pred_tag=[]
                for vid in valDict.keys():
                    if vid<num_clp:
                        indc=1
                        opp=0
                    else:
                        indc=0
                        opp=1
                    xVal=valDict[vid]
                    xVal=(xVal-np.min(xVal))/(np.max(xVal)-np.min(xVal))
                    test_data=testData(torch.FloatTensor(xVal))
                    test_loader=torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_W, drop_last=False, shuffle=True) 
                    val_loss_curr=0
                    y_pred_tag_curr=[]
                    model.eval()
                    with torch.no_grad():
                        for X_test in test_loader:
                            yTest=np.vstack((np.ones((X_test.shape[0]))*indc,np.ones((X_test.shape[0]))*opp)).T
                            if torch.cuda.is_available():
                                X_test=X_test.cuda()

                            y_test_pred = model.forward(X_test)

                            #Find difference in probability of CLP v. HC for all segments. 
                            y_pred_tag_curr.extend((y_test_pred[:,0]-y_test_pred[:,1]).cpu().detach().numpy())
                            if torch.cuda.is_available():
                                loss = criterion(y_test_pred, torch.from_numpy(yTest).cuda().float())
                            else:
                                loss = criterion(y_test_pred, torch.from_numpy(yTest).float())
                            val_loss_curr+=loss.item()*X_test.size(0)

                    if num_val==0:
                        indcs_vec=np.ones(len(y_pred_tag_curr))*indc
                        num_val+=1
                    else:
                        indcs_vec=np.concatenate((indcs_vec,np.ones(len(y_pred_tag_curr))*indc))
                        num_val+=1
                        
                    y_pred_tag.extend(y_pred_tag_curr)
                    val_loss+=val_loss_curr/len(test_loader.dataset)
                
                y_pred_tag=np.array(y_pred_tag).reshape(-1,1)
                val_acc=clf.score(y_pred_tag,indcs_vec)
                
                trainResults_epo.iloc[epoch]['train_acc']=tr_acc
                trainResults_epo.iloc[epoch]['val_loss']=val_loss/num_val
                trainResults_epo.iloc[epoch]['val_acc']=val_acc

                print('Train Loss: {:.6f} Train Accuracy: {}\nValidation Loss: {:.6f} Validation Accuracy: {}\n'.format(
                train_loss/num_tr,  
                tr_acc,
                val_loss/num_val,
                val_acc,
                ))      

        #AFTER MODEL TRAINED (FOR ALL SPEAKERS AND OVER NUM_EPOCHS), TEST MODEL ON LEFT OUT SPEAKERS  
        test_loss=0.0
        num_test=0
        y_pred_tag=[]  
        for spkItr,spk in enumerate(['clp','hc']):
            dic=testDict[spk]

            for tstId in dic.keys():
                if tstId<num_clp:
                    indc=1
                    opp=0
                else:
                    indc=0
                    opp=1
                test_loss_curr=0
                y_pred_tag_curr=[]
                xTest=dic[tstId]
                xTest=(xTest-np.min(xTest))/(np .max(xTest)-np.min(xTest))
                test_data=testData(torch.FloatTensor(xTest))
                test_loader=torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_W, drop_last=False, shuffle=True)  
                model.eval()
                with torch.no_grad():
                    for X_test in test_loader:
                        yTest=np.vstack((np.ones((X_test.shape[0]))*indc,np.ones((X_test.shape[0]))*opp)).T
                        if torch.cuda.is_available():
                            X_test=X_test.cuda()

                        y_test_pred = model.forward(X_test)

                        #Find difference in probability of CLP v. HC for all segments. 
                        y_pred_tag_curr.extend((y_test_pred[:,0]-y_test_pred[:,1]).cpu().detach().numpy())

                        if torch.cuda.is_available():
                            loss = criterion(y_test_pred, torch.from_numpy(yTest).cuda().float())
                        else:
                            loss = criterion(y_test_pred, torch.from_numpy(yTest).float())
                        test_loss_curr+=loss.item()*X_test.size(0)

                #accuracy determined on majority rule (wlog, if more frames yield probability differences greater than 0,
                #and if speaker is CLP than classification assessment is correct (1=>CLP,0=>HC).
                test_loss+=test_loss_curr/len(test_loader.dataset)
                if num_test==0:
                    indcs_vec=np.ones(len(y_pred_tag_curr))*indc
                    num_test=1
                else:
                    indcs_vec=np.concatenate((indcs_vec,np.ones(len(y_pred_tag_curr))*indc))
                    num_test+=1
                y_pred_tag.extend(y_pred_tag_curr)
                #Store raw scores for each test speaker (probability of CLP and HC as output by dnn) for ROC.
                tst_diffs=(y_test_pred[:,0]-y_test_pred[:,1]).cpu().detach().numpy().reshape(-1,1)
                testResults[itr]['tstSpk_data'][tstId]=calibrator.predict_proba(tst_diffs)

        y_pred_tag=np.array(y_pred_tag).reshape(-1,1)
        test_acc=clf.score(y_pred_tag,indcs_vec)
        #Store and report loss and accuracy for batch of test speakers.            
        testResults[itr]['test_loss'],testResults[itr]['test_acc']=test_loss/num_clpHc_tests,test_acc
        print('\nTest Loss: {:.3f} \tTest Acc: {:.3f} '.format(
                    test_loss/num_clpHc_tests,
                    test_acc
            ))
          
        train_res.append(trainResults_epo)
        
        if (len(clp_files)<num_clpHc_tests//2):
            num_clpHc_tests=len(clp_files)*2

        if not hc_files:
            hc_files=hcNames
        if len(hc_files)<num_clpHc_tests//2:
            left_ids=[lid for lid in np.arange(len(hcNames)) if hcNames[lid] not in hc_files]
            add_ids=random.sample(left_ids, (num_clpHc_tests//2)-len(hc_files))
            hc_files.extend(np.array(hcNames)[add_ids])

    trainResults=pd.concat(train_res,keys=(np.arange(1)))

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
    
    if rep_typ=='mc_fuse':
        trainResults.to_pickle(save_path+mod+'_mcFusion_trainResults.pkl', protocol=4)
        testResults.to_pickle(save_path+mod+'_mcFusion_testResults.pkl', protocol=4)
    if rep_typ in ['broadband','narrowband','wvlt']:
        trainResults.to_pickle(save_path+mod+'_'+rep+"_aggTrainResults.pkl", protocol=4)
        testResults.to_pickle(save_path+mod+'_'+rep+"_aggTestResults.pkl", protocol=4)
    if rep_typ=='early_fuse':
        trainResults.to_pickle(save_path+mod+'_earlyFusion_trainResults.pkl', protocol=4)
        testResults.to_pickle(save_path+mod+'_earlyFusion_testResults.pkl', protocol=4)
        
        
