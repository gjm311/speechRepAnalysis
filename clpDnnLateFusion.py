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
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
import toolbox.traintestsplit as tts
from clpAEspeech import AEspeech
import json
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



def saveFeats(model,units,rep,wav_path,utter,save_path, spk_typ):
    global UNITS    
    # load the pretrained model with 256 units and get temp./freq. rep (spec or wvlt)
    aespeech=AEspeech(model=model,units=UNITS,rep=rep) 
    
    #compute the bottleneck and error-based features from a directory with wav files inside 
    #(dynamic: one feture vector for each 500 ms frame)
    #(global i.e. static: one feture vector per utterance)
    feat_vecs=aespeech.compute_dynamic_features(wav_path)
    #     df1, df2=aespeech.compute_global_features(wav_path)
    
    with open(save_path+'/'+rep+'_'+model+'_'+spk_typ+'dnnFeats.pickle', 'wb') as handle:
        pickle.dump(feat_vecs, handle, protocol=, protocol=4)
    
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

    if len(sys.argv)!=3:
        print("python clpdnnLateFusion.py <'CAE','RAE', or 'ALL'> <clp path>")
        sys.exit()        
    #TRAIN_PATH: './clpSpanish/speech/<UTTER>/'
    
    
    if sys.argv[1] in MODELS:
        mod=sys.argv[1]
    else:
        print("python clpdnnLateFusion.py.py <'CAE','RAE', or 'ALL'> <clp path>")
        sys.exit()
    
    if sys.argv[2][0] !='/':
        sys.argv[2] = '/'+sys.argv[2]
    if sys.argv[2][-1] !='/':
        sys.argv[2] = sys.argv[2]+'/'
    spk_path=PATH+sys.argv[2]
    
    reps=['broadband','narrowband']
    LR=config['dnn']['LR']
    BATCH_SIZE=config['dnn']['BATCH_SIZE']
    NUM_W=config['dnn']['NUM_W']
    N_EPOCHS=config['dnn']['N_EPOCHS']
    num_clpHc_tests=config['dnn']['tst_spks']#must be even (same # of test clps and hcs per iter)
    nv=config['dnn']['val_spks']#number of validation speakers per split
    num_iters=config['dnn']['num_iters']
#     LRs=[10**-ex for ex in np.linspace(4,7,6)]

    NBF=config['mel_spec']['INTERP_NMELS']
    
    save_path=PATH+"/clpSpanish/classResults/dnn/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    testResults=pd.DataFrame({splItr:{'test_acc':0,'class_report':{}, 'tstSpk_data':{}, } for splItr in range(num_iters)})     
    train_res=[]
        
    #iterate through all clp and hc speakers for a given utterance (see UTTERS for options) and using leave ten out, train a DNN
    #(see clpdnn.py) and classify one by one if CLP or HC.
#     lr_score_opt=0
#     lr_scores=pd.DataFrame(columns=LRs,index=np.arange(1))
#     for lrItr,LR in enumerate(LRs):

    #aggregate all utterance data per speaker together.
    #aggregate all utterance data per speaker together.
    
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
    clpNames.sort()
    hcNames.sort()
    spks=clpNames+hcNames
    num_spks=len(spks)
    num_clp=len(clpNames)
    num_hc=len(hcNames)
    ntr=num_spks-(num_clpHc_tests+nv)
    
    #aggregate all utterance data per speaker together.
    master_clpIds=np.arange(0,num_clp)
    master_hcIds=np.arange(num_clp,num_clp+num_hc)
    spkDict={rep:{spk:{} for spk in ['clp','hc']} for rep in reps}
    for rep in reps:
        spkDict[rep]['clp']={num:[] for num in master_clpIds}
        spkDict[rep]['hc']={num:[] for num in master_hcIds}
        
    for u_idx,utter in enumerate(UTTERS):
        clp_path=spk_path+'/'+utter+"/clp/"
        hc_path=spk_path+'/'+utter+"/hc/"   
        clpNames_curr=[name.split('_')[0] for name in os.listdir(clp_path) if '.wav' in name]
        hcNames_curr=[name.split('_')[0] for name in os.listdir(hc_path) if '.wav' in name]
        clpNames_curr.sort()
        hcNames_curr.sort()
        clp_rng=np.arange(num_clp)
        hc_rng=np.arange(num_hc)
        clpIds_curr=[spks.index(clpName) for clpName in clpNames_curr]
        hcIds_curr=[spks.index(hcName) for hcName in hcNames_curr]
        for rIdx,rep in enumerate(reps):
            clpFeats=getFeats(mod,UNITS,rep,clp_path,utter,'clp')
            hcFeats=getFeats(mod,UNITS,rep,hc_path,utter,'hc')

            for rItr,pItr in zip(clp_rng,clpIds_curr):
                clpBns=clpFeats['bottleneck'][np.where(clpFeats['wav_file']==clpNames_curr[rItr])]
                clpErrs=clpFeats['error'][np.where(clpFeats['wav_file']==clpNames_curr[rItr])]
                if len(spkDict[rep]['clp'][pItr])==0:
                    spkDict[rep]['clp'][pItr]=np.concatenate((clpBns,clpErrs),axis=1)
                else:
                    spkDict[rep]['clp'][pItr]=np.concatenate((spkDict[rep]['clp'][pItr],np.concatenate((clpBns,clpErrs),axis=1)),axis=0)
            for rItr,hItr in zip(hc_rng,hcIds_curr):
                hcBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==hcNames_curr[rItr])]
                hcErrs=hcFeats['error'][np.where(hcFeats['wav_file']==hcNames_curr[rItr])]
                if len(spkDict[rep]['hc'][hItr])==0:
                    spkDict[rep]['hc'][hItr]=np.concatenate((hcBns,hcErrs),axis=1)
                else:
                    spkDict[rep]['hc'][hItr]=np.concatenate((spkDict[rep]['hc'][hItr],np.concatenate((hcBns,hcErrs),axis=1)),axis=0)

                    
    
    clp_files=clpNames
    hc_files=hcNames
    #split data into training and test with multiple iterations (evenly split CLP:HC) 
    for itr in range(num_iters):
        #RESET model
        nb_model=clpdnn(UNITS+NBF)
        bb_model=clpdnn(UNITS+NBF)
        criterion=nn.BCELoss()
        nb_optimizer = torch.optim.Adam(nb_model.parameters(), lr = LR)
        bb_optimizer = torch.optim.Adam(bb_model.parameters(), lr = LR)

        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
            nb_model.cuda()
            bb_model.cuda()
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
        
        #Get test speaker features
        clpCurrs=[clp_files[idx] for idx in random.sample(range(0,len(clp_files)),num_clpHc_tests//2)]
        hcCurrs=[hc_files[idx] for idx in random.sample(range(0,len(hc_files)),num_clpHc_tests//2)]
        clp_files=[clp for clp in clp_files if clp not in clpCurrs]
        hc_files=[hc for hc in hc_files if hc not in hcCurrs]
        
        
        clpIds=[spks.index(clpCurr) for clpCurr in clpCurrs]
        hcIds=[spks.index(hcCurr) for hcCurr in hcCurrs]
        
        testDict={rep:{spk:{} for spk in ['clp','hc']} for rep in reps}
        testDict['clp']={num:[] for num in clpIds}
        testDict['hc']={num:[] for num in hcIds}
        
        for rep in reps:
            for clpItr in clpIds:
                testDict[rep]['clp'][clpItr]=spkDict[rep]['clp'][clpItr]
            for hcItr in hcIds:
                testDict[rep]['hc'][hcItr]=spkDict[rep]['hc'][hcItr]
        
        #Separate 'nv' (equal number of clp/hc) Validation speakers and get features
        notTestSpksCLP=[spk for spk in clpNames if spk not in clpCurrs]
        notTestSpksHC=[spk for spk in hcNames if spk not in hcCurrs]
        validsCLP=[notTestSpksCLP[idx] for idx in random.sample(range(0,len(notTestSpksCLP)),nv//2)]
        validsHC=[notTestSpksHC[idx] for idx in random.sample(range(0,len(notTestSpksHC)),nv//2)]
        valids=validsCLP+validsHC
        valIds=[spks.index(valid) for valid in valids]
        valDict={rep:{num:[] for num in valIds} for rep in reps}

        #getting bottle neck features and reconstruction error for validation speakers
        
        for ii,val in enumerate(valids):
            for rep in reps:
                vitr=valIds[ii]
                if vitr<num_clp:
                    spk_typ='clp'
                else:
                    spk_typ='hc'
                valDict[rep][vitr]=spkDict[rep][spk_typ][vitr]

        trainResults_epo= pd.DataFrame({'nb_train_loss':0.0,'bb_train_loss':0.0, 'train_acc':0.0, 'val_acc':0.0}, index=np.arange(N_EPOCHS))

        num_pats_tr=num_spks-num_clpHc_tests-nv
        notTestValSpksCLP=[spk for spk in clpNames if spk not in clpCurrs+validsCLP]
        notTestValSpksHC=[spk for spk in hcNames if spk not in hcCurrs+validsHC]

        clptr_rand_range=[spks.index(trid) for trid in notTestValSpksCLP]
        hctr_rand_range=[spks.index(trid) for trid in notTestValSpksHC]
        random.shuffle(clptr_rand_range)
        random.shuffle(hctr_rand_range)
        tr_rand_range=clptr_rand_range[:num_pats_tr//2]+hctr_rand_range[:num_pats_tr//2]
        
        for epoch in range(N_EPOCHS):  
            train_losses=[0.0,0.0]
            
            #TRAIN dnn for each speaker individually.             
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
                
                for rtritr,rep in enumerate(reps):
                    #getting bottle neck features and reconstruction error for particular training speaker
                    xTrain=spkDict[rep][spk_typ][trainItr]
                    xTrain=(xTrain-np.min(xTrain))/(np.max(xTrain)-np.min(xTrain))
                    yTrain=np.vstack((np.ones((xTrain.shape[0]))*trainIndc,np.ones((xTrain.shape[0]))*trainOpp)).T
                    train_data=trainData(torch.FloatTensor(xTrain), torch.FloatTensor(yTrain))
                    train_loader=torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_W)
                    start=time.time()

                    train_loss_curr=0.0
                    if len(train_loader)>0:
                        #TRAINING ON SNIPPETS OF SPEAKER UTTERANCES
                        if rep=='broadband':
                            bb_model.train() # prep model for training
                        elif rep=='narrowband':
                            nb_model.train() # prep model for training
                            
                        for X_train, y_train in train_loader:
                            #clear the gradients of all optimized variables
                            if rep=='broadband':
                                bb_optimizer.zero_grad()
                            elif rep=='narrowband':
                                nb_optimizer.zero_grad()

                            X_train=X_train.float()                
                            y_train=y_train.float()

                            if torch.cuda.is_available():
                                X_train,y_train=X_train.cuda(),y_train.cuda()
                            
                            if rep=='broadband':
                                y_pred=bb_model.forward(X_train)
                            elif rep=='narrowband':
                                y_pred=nb_model.forward(X_train)
                            #Find difference in probability of CLP v. HC for all segments.
                            
                            if torch.cuda.is_available():
                                y_pred=y_pred.cuda()

                            loss=criterion(y_pred, y_train)
                            loss.backward()
                            
                            if rep=='broadband':
                                bb_optimizer.step()
                            elif rep=='narrowband':
                                nb_optimizer.step()
                            train_loss_curr += loss.item()*y_train.size(0)

                        #tally up train loss total for given speaker
                        train_losses[rtritr]+=train_loss_curr/len(train_loader.dataset)                           

            #Record train loss at end of each epoch (divide by number of train patients).
            trainResults_epo.iloc[epoch]['bb_train_loss']=train_losses[0]/len(tr_rand_range)
            trainResults_epo.iloc[epoch]['nb_train_loss']=train_losses[1]/len(tr_rand_range)
            
            if np.mod(epoch+1,1)==0 or epoch==0:
                #Iterate through thresholds and choose one that yields best validation acc.
                #Iterate through all num_tr training patients and classify based off difference in probability of CLP/HC
                tags={spk:[] for spk in tr_rand_range if spk not in np.concatenate((clpIds,hcIds,valIds))}
                frame_res=[]
                for rritr,trainItr in enumerate(tr_rand_range):   
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
                    
                    
                    for rep in reps:
                        xTrain=spkDict[rep][spk_typ][trainItr]
                        xTrain=(xTrain-np.min(xTrain))/(np.max(xTrain)-np.min(xTrain))
                        yTrain=np.vstack((np.ones((xTrain.shape[0]))*trainIndc,np.ones((xTrain.shape[0]))*trainOpp)).T
                        train_data=testData(torch.FloatTensor(xTrain))
                        train_loader=torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_W)

                        if rep=='broadband':
                            bb_y_pred_tag=[]
                            bb_model.eval()
                        elif rep=='narrowband':
                            nb_y_pred_tag=[]
                            nb_model.eval()

                        with torch.no_grad():
                            for X_tr in train_loader:
                                yTr=np.vstack((np.ones((X_tr.shape[0]))*trainIndc,np.ones((X_tr.shape[0]))*trainOpp)).T
                                if torch.cuda.is_available():
                                    X_tr=X_tr.cuda()
                                if rep=='broadband':
                                    y_tr_pred = bb_model.forward(X_tr)
                                    bb_y_pred_tag.extend((y_tr_pred[:,0]-y_tr_pred[:,1]).cpu().detach().numpy())
                                elif rep=='narrowband':
                                    y_tr_pred = nb_model.forward(X_tr)
                                    nb_y_pred_tag.extend((y_tr_pred[:,0]-y_tr_pred[:,1]).cpu().detach().numpy())


                    bb_y_pred_tag=np.array(bb_y_pred_tag)
                    nb_y_pred_tag=np.array(nb_y_pred_tag)

                    if frame_res:
                        frame_res=frame_res+tuple((x1,x2) for x1,x2 in zip(bb_y_pred_tag,nb_y_pred_tag))
                        indcs_vec=np.concatenate((indcs_vec,np.ones(len(bb_y_pred_tag))*trainIndc))
                    else:
                        frame_res=tuple((x1,x2) for x1,x2 in zip(bb_y_pred_tag,nb_y_pred_tag))
                        indcs_vec=np.ones(len(bb_y_pred_tag))*trainIndc
                    tags[trainItr]=tuple((x1,x2) for x1,x2 in zip(bb_y_pred_tag,nb_y_pred_tag))


                clf = SGDClassifier(loss="hinge", penalty="l2")
                clf.fit(frame_res, indcs_vec)
                calibrator=CalibratedClassifierCV(clf, cv='prefit')
                modCal=calibrator.fit(frame_res, indcs_vec)       
                
                preds=[]
                pred_truths=[]
                for scount,spk in enumerate(tags.keys()):
                    if scount==0:
                        preds=modCal.predict_proba(tags[spk])
                        if spk<num_clp:
                            pred_truths=np.ones(len(tags[spk]))
                        else:
                            pred_truths=np.zeros(len(tags[spk]))
                    else:
                        preds=np.concatenate((preds,modCal.predict_proba(tags[spk])))
                        if spk<num_clp:
                            pred_truths=np.concatenate((pred_truths,np.ones(len(tags[spk]))))
                        else:
                            pred_truths=np.concatenate((pred_truths,np.zeros(len(tags[spk]))))
                
                clf2 = SGDClassifier(loss="hinge", penalty="l2")
                clf2.fit(preds, pred_truths)
                tr_acc=clf2.score(preds, pred_truths)
                
                #Validate at end of each mod epochs for nv speakers
                tags={spk:[] for spk in valIds}
                frame_res=[]
                for rritr,vid in enumerate(valIds):
                    if vid<num_clp:
                        indc=1
                        opp=0
                    else:
                        indc=0
                        opp=1

                    for rep in reps:
                        xVal=valDict[rep][vid]
                        xVal=(xVal-np.min(xVal))/(np.max(xVal)-np.min(xVal))
                        test_data=testData(torch.FloatTensor(xVal))
                        test_loader=torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_W, drop_last=False, shuffle=True) 
                        if rep=='broadband':
                            bb_model.eval()
                            bb_y_pred_tag=[]
                        elif rep=='narrowband':
                            nb_model.eval()
                            nb_y_pred_tag=[]
                        with torch.no_grad():
                            for X_test in test_loader:
                                yTest=np.vstack((np.ones((X_test.shape[0]))*indc,np.ones((X_test.shape[0]))*opp)).T
                                if torch.cuda.is_available():
                                    X_test=X_test.cuda()
                                if rep=='broadband':
                                    y_test_pred = bb_model.forward(X_test) 
                                    bb_y_pred_tag.extend((y_test_pred[:,0]-y_test_pred[:,1]).cpu().detach().numpy())
                                elif rep=='narrowband':
                                    y_test_pred = nb_model.forward(X_test)
                                    nb_y_pred_tag.extend((y_test_pred[:,0]-y_test_pred[:,1]).cpu().detach().numpy())

                    bb_y_pred_tag=np.array(bb_y_pred_tag)
                    nb_y_pred_tag=np.array(nb_y_pred_tag)

                    if frame_res:
                        frame_res=frame_res+tuple((x1,x2) for x1,x2 in zip(bb_y_pred_tag,nb_y_pred_tag))
                        indcs_vec=np.concatenate((indcs_vec,np.ones(len(bb_y_pred_tag))*trainIndc))
                        frame_res=tuple((x1,x2) for x1,x2 in zip(bb_y_pred_tag,nb_y_pred_tag))
                        indcs_vec=np.ones(len(bb_y_pred_tag))*indc
                    else:
                        frame_res=tuple((x1,x2) for x1,x2 in zip(bb_y_pred_tag,nb_y_pred_tag))
                        indcs_vec=np.ones(len(bb_y_pred_tag))*indc
                    tags[vid]=tuple((x1,x2) for x1,x2 in zip(bb_y_pred_tag,nb_y_pred_tag))
                
                
                preds=[]
                pred_truths=[]
                for scount,spk in enumerate(tags.keys()):
                    if scount==0:
                        preds=modCal.predict_proba(tags[spk])
                        if spk<num_clp:
                            pred_truths=np.ones(len(tags[spk]))
                        else:
                            pred_truths=np.zeros(len(tags[spk]))
                    else:
                        preds=np.concatenate((preds,modCal.predict_proba(tags[spk])))
                        if spk<num_clp:
                            pred_truths=np.concatenate((pred_truths,np.ones(len(tags[spk]))))
                        else:
                            pred_truths=np.concatenate((pred_truths,np.zeros(len(tags[spk]))))
 
                val_acc=clf2.score(preds,pred_truths)
                    
                    

                trainResults_epo.iloc[epoch]['train_acc']=tr_acc
                trainResults_epo.iloc[epoch]['val_acc']=val_acc

                print('BB Train Loss: {:.6f} NB Train Loss: {:.6f} Train Accuracy: {} Validation Accuracy: {}\n'.format(
                train_losses[0]/ntr,
                train_losses[1]/ntr,  
                tr_acc,
                val_acc,
                ))      

        #AFTER MODEL TRAINED (FOR ALL SPEAKERS AND OVER NUM_EPOCHS), TEST MODEL ON LEFT OUT SPEAKERS  
        test_loss=0.0
        tags={spk:[] for spk in clpIds+hcIds}
        frame_res=[]
        for spkItr,spk in enumerate(['clp','hc']):
            for rritr,tstId in enumerate(testDict[list(testDict.keys())[0]][spk].keys()):
                if tstId<num_clp:
                    indc=1
                    opp=0
                else:
                    indc=0
                    opp=1
                for rep in reps:
                    xTest=testDict[rep][spk][tstId]
                    xTest=(xTest-np.min(xTest))/(np.max(xTest)-np.min(xTest))
                    test_data=testData(torch.FloatTensor(xTest))
                    test_loader=torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_W, drop_last=False, shuffle=True)  
                    if rep=='broadband':
                        bb_model.eval()
                        bb_y_pred_tag=[]
                    elif rep=='narrowband':
                        nb_model.eval()
                        nb_y_pred_tag=[]
                    with torch.no_grad():
                        for X_test in test_loader:
                            yTest=np.vstack((np.ones((X_test.shape[0]))*indc,np.ones((X_test.shape[0]))*opp)).T
                            if torch.cuda.is_available():
                                X_test=X_test.cuda()

                            if rep=='broadband':
                                y_test_pred = bb_model.forward(X_test) 
                                bb_y_pred_tag.extend((y_test_pred[:,0]-y_test_pred[:,1]).cpu().detach().numpy())
                            elif rep=='narrowband':
                                y_test_pred = nb_model.forward(X_test)
                                nb_y_pred_tag.extend((y_test_pred[:,0]-y_test_pred[:,1]).cpu().detach().numpy())

                bb_y_pred_tag=np.array(bb_y_pred_tag)
                nb_y_pred_tag=np.array(nb_y_pred_tag)

                if frame_res:
                    frame_res=frame_res+tuple((x1,x2) for x1,x2 in zip(bb_y_pred_tag,nb_y_pred_tag))
                    indcs_vec=np.concatenate((indcs_vec,np.ones(len(bb_y_pred_tag))*trainIndc))
                else:
                    frame_res=tuple((x1,x2) for x1,x2 in zip(bb_y_pred_tag,nb_y_pred_tag))
                    indcs_vec=np.ones(len(bb_y_pred_tag))*indc            
                tags[tstId]=tuple((x1,x2) for x1,x2 in zip(bb_y_pred_tag,nb_y_pred_tag))
                
        preds=[]
        pred_truths=[]
        for scount,key in enumerate(tags.keys()):
            testResults[itr]['tstSpk_data'][key]=modCal.predict_proba(tags[key])
            
            if scount==0:
                preds=modCal.predict_proba(tags[key])
                if key<num_clp:
                    pred_truths=np.ones(len(tags[key]))
                else:
                    pred_truths=np.zeros(len(tags[key]))
            else:
                preds=np.concatenate((preds,modCal.predict_proba(tags[key])))
                if key<num_clp:
                    pred_truths=np.concatenate((pred_truths,np.ones(len(tags[key]))))
                else:
                    pred_truths=np.concatenate((pred_truths,np.zeros(len(tags[key]))))

        test_acc=clf2.score(preds,pred_truths)
        #Store and report loss and accuracy for batch of test speakers.            
        testResults[itr]['test_acc']=test_acc
        testResults[itr]['class_report']=classification_report(pred_truths,calibrator.predict(preds))
        print('\nTest Acc: {:.3f} '.format(
                    test_acc
            ))
          
        train_res.append(trainResults_epo)
        
        if len(clp_files)<num_clpHc_tests//2:
            num_clpHc_tests=len(clp_files)*2
            
        if not hc_files:
            hc_files=hcNames
        if len(hc_files)<num_clpHc_tests//2:
            left_ids=[lid for lid in np.arange(num_hc) if hcNames[lid] not in hc_files]
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

    trainResults.to_pickle(save_path+mod+'_lateFusion_trainResults.pkl', protocol=4)
    testResults.to_pickle(save_path+mod+'_lateFusion_testResults.pkl', protocol=4)






