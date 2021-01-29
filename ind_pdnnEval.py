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
from sklearn.linear_model import SGDClassifier
import torch.utils.data as data
from pdnn import pdn
import toolbox.traintestsplit as tts
from AEspeech import AEspeech
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import json
import argparse
import pdb
from sklearn import metrics

PATH=os.path.dirname(os.path.abspath(__file__))
#LOAD CONFIG.JSON INFO
with open("config.json") as f:
    info = f.read()
config = json.loads(info)
UNITS=config['general']['UNITS']
UTTERS=['pataka']#,'kakaka','pakata','papapa','petaka','tatata']
MODELS=["CAE","RAE","ALL"]
REPS=['broadband','narrowband','wvlt']



def saveFeats(model,units,rep,wav_path,utter,save_path, spk_typ):
    global UNITS    
    # load the pretrained model with 256 units and get temp./freq. rep (spec or wvlt)
    aespeech=AEspeech(model=model,units=UNITS,rep=rep) 
    
    #compute the bottleneck and error-based features from a directory with wav files inside 
    #(dynamic: one feture vector for each 500 ms frame)
    #(global i.e. static: one feture vector per utterance)
    feat_vecs=aespeech.compute_dynamic_features(wav_path,rsmp=1)
    #     df1, df2=aespeech.compute_global_features(wav_path)
    
    with open(save_path+'/'+rep+'_'+model+'_'+spk_typ+'Feats.pickle', 'wb') as handle:
        pickle.dump(feat_vecs, handle, protocol=4)
    
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
        
def featAgg(mod,rep,spk_path,ef,tst=0):
    #aggregate all utterance data per speaker together.
    global UTTERS
    
    if tst==0:
        utters=UTTERS
        pdIds=np.arange(50)
        hcIds=np.arange(50,100)
        spkDict={spk:{num[i]:{'feats':[]} for num in zip(pdIds,hcIds)} for i,spk in enumerate(['pd','hc'])}
    elif tst==1:
        utters=['pataka_test']
        pdIds=np.arange(20)
        hcIds=np.arange(20,40)
        spkDict={spk:{num[i]:{'feats':[]} for num in zip(pdIds,hcIds)} for i,spk in enumerate(['pd','hc'])}
        
    reps=rep
    for u_idx,utter in enumerate(utters):
        pd_path=spk_path+'/'+utter+"/pd/"
        hc_path=spk_path+'/'+utter+"/hc/"   
        pdNames=[name for name in os.listdir(pd_path) if '.wav' in name]
        hcNames=[name for name in os.listdir(hc_path) if '.wav' in name]
        pdNames.sort()
        hcNames.sort()
        spks=pdNames+hcNames
        num_spks=len(spks)
        num_pd=len(pdNames)
        num_hc=len(hcNames)
        
        if ef==0:
            pdFeats=getFeats(mod,UNITS,rep,pd_path,utter,'pd')
            hcFeats=getFeats(mod,UNITS,rep,hc_path,utter,'hc')
            pdAll=np.unique(pdFeats['wav_file'])
            hcAll=np.unique(hcFeats['wav_file'])
            for p in pdIds:
                pdBns=pdFeats['bottleneck'][np.where(pdFeats['wav_file']==spks[p])]
                pdErrs=pdFeats['error'][np.where(pdFeats['wav_file']==spks[p])]
                if u_idx==0:
                    spkDict['pd'][p]=np.concatenate((pdBns,pdErrs),axis=1)
                else:
                    spkDict['pd'][p]=np.concatenate((spkDict['pd'][p],np.concatenate((pdBns,pdErrs),axis=1)),axis=0)

            for h in hcIds:
                hcBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==spks[h])]
                hcErrs=hcFeats['error'][np.where(hcFeats['wav_file']==spks[h])]
                if u_idx==0:
                    spkDict['hc'][h]=np.concatenate((hcBns,hcErrs),axis=1)
                else:
                    spkDict['hc'][h]=np.concatenate((spkDict['hc'][h],np.concatenate((hcBns,hcErrs),axis=1)),axis=0)
        else:
            for rIdx,rep in enumerate(reps):
                pdFeats=getFeats(mod,UNITS,rep,pd_path,utter,'pd')
                hcFeats=getFeats(mod,UNITS,rep,hc_path,utter,'hc')
                pdAll=np.unique(pdFeats['wav_file'])
                hcAll=np.unique(hcFeats['wav_file'])
                for p in pdIds:
                    pdBns=pdFeats['bottleneck'][np.where(pdFeats['wav_file']==spks[p])]
                    pdErrs=pdFeats['error'][np.where(pdFeats['wav_file']==spks[p])]
                    if u_idx==0 and rIdx==0:
                        spkDict['pd'][p]=np.concatenate((pdBns,pdErrs),axis=1)
                    else:
                        spkDict['pd'][p]=np.concatenate((spkDict['pd'][p],np.concatenate((pdBns,pdErrs),axis=1)),axis=0)

                for h in hcIds:
                    hcBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==spks[h])]
                    hcErrs=hcFeats['error'][np.where(hcFeats['wav_file']==spks[h])]
                    if u_idx==0 and rIdx==0:
                        spkDict['hc'][h]=np.concatenate((hcBns,hcErrs),axis=1)
                    else:
                        spkDict['hc'][h]=np.concatenate((spkDict['hc'][h],np.concatenate((hcBns,hcErrs),axis=1)),axis=0)

    return spkDict,pdNames,hcNames
    
    
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

    if len(sys.argv)!=4:
        print("python pdnnEvalAgg.py <'CAE','RAE'> <broadband, narrowband, wvlt, early_fuse, mc_fuse> <pd path>")
        sys.exit()        
    #TRAIN_PATH: './pdSpanish/speech/'
    
    
    if sys.argv[1] in MODELS:
        mod=sys.argv[1]
    else:
        print("python pdnnEvalAgg.py <'CAE','RAE'> <broadband, narrowband, wvlt, early_fuse, mc_fuse> <pd path>")
        sys.exit()
    
    
    if sys.argv[2] not in ['broadband', 'narrowband', 'wvlt', 'early_fuse', 'mc_fuse']:
        print("python ind_pdnnEvalAgg.py <'CAE','RAE'> <broadband, narrowband, wvlt, early_fuse, mc_fuse> <pd path>")
        sys.exit()
    else:
        rep_typ=sys.argv[2]
    
    
    if sys.argv[3][0] !='/':
        sys.argv[3] = '/'+sys.argv[3]
    if sys.argv[3][-1] !='/':
        sys.argv[3] = sys.argv[3]+'/'
    spk_path=PATH+sys.argv[3]
        
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
    
    spkDict,pdNames,hcNames=featAgg(mod,rep,spk_path,ef,tst=0)
    testDict,tst_pdNames,tst_hcNames=featAgg(mod,rep,spk_path,ef,tst=1)
    num_pd=len(pdNames)
    num_hc=len(hcNames)
    spks=pdNames+hcNames
    num_spks=len(spks)
    
    LR=config['dnn']['LR']
    BATCH_SIZE=config['dnn']['BATCH_SIZE']
    NUM_W=config['dnn']['NUM_W']
    N_EPOCHS=config['dnn']['N_EPOCHS']
    nv=config['dnn']['val_spks']#number of validation speakers per split

#     LRs=[10**-ex for ex in np.linspace(4,7,6)]
    
    save_path=PATH+"/pdSpanish/classResults/dnn/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    ntr=100
    testResults=pd.DataFrame({'test_loss':0,'test_acc':0,'tstSpk_data':{ind:[] for ind in range(40)},'class_report':{}})     
    train_res=[]
        
    #iterate through all pd and hc speakers for a given utterance (see UTTERS for options) and using leave ten out, train a DNN
    #(see pdnn.py) and classify one by one if PD or HC.
#     lr_score_opt=0
#     lr_scores=pd.DataFrame(columns=LRs,index=np.arange(1))
#     for lrItr,LR in enumerate(LRs):
    
    
    #SET model
    model=pdn(UNITS+NBF)
    criterion=nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        model.cuda()
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

#     #Get test speaker features
#     pdNames=[pd_files[idx] for idx in random.sample(range(0,len(pd_files)),num_pdHc_tests//2)]
#     hcNames=[hc_files[idx] for idx in random.sample(range(0,len(hc_files)),num_pdHc_tests//2)]
#     pd_files=[pd for pd in pd_files if pd not in pdNames]
#     hc_files=[hc for hc in hc_files if hc not in hcNames]

#     pdIds=np.arange(50)
#     hcIds=np.arange(50)

# #     testDict={spk:{num[i]:{'feats':[]} for num in zip(pdIds,hcIds)} for i,spk in enumerate(['pd','hc'])}
# #     for pdItr in pdIds:
# #         testDict['pd'][pdItr]=spkDict['pd'][pdItr]
# #     for hcItr in hcIds:
# #         testDict['hc'][hcItr]=spkDict['hc'][hcItr]

    #Separate 'nv' (equal number of pd/hc) Validation speakers and get features
#     notTestSpksPD=[spk for spk in pdNames if spk not in pdNames]
#     notTestSpksHC=[spk for spk in hcNames if spk not in hcNames]
    validsPD=[spks[idx] for idx in random.sample(range(0,50),nv//2)]
    validsHC=[spks[idx] for idx in random.sample(range(0,50),nv//2)]
    valids=validsPD+validsHC
    valIds=[spks.index(valid) for valid in valids]
    valDict={num:{'feats':[]} for num in valIds}

    #getting bottle neck features and reconstruction error for validation speakers
    for ii,val in enumerate(valids):
        vitr=valIds[ii]
        if vitr<num_pd:
            spk_typ='pd'
        else:
            spk_typ='hc'
        valDict[vitr]=spkDict[spk_typ][vitr]

    trainResults_epo= pd.DataFrame({'train_loss':0.0, 'train_acc':0.0,'val_loss':0.0, 'val_acc':0.0}, index=np.arange(N_EPOCHS))

    for epoch in range(N_EPOCHS):  
        train_loss=0.0
        train_acc=0.0
        rand_range=np.arange(num_spks)
        random.shuffle(rand_range)

        #TRAIN dnn for each speaker individually.             
        for trainItr in rand_range:   
            if trainItr in valIds:
                continue

            if trainItr<num_pd:
                trainIndc=1
                trainOpp=0
                spk_typ='pd'
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

        #Record train loss at end of each epoch (divide by number of train patients - ntr).
        trainResults_epo.iloc[epoch]['train_loss']=train_loss/ntr

        if np.mod(epoch+1,1)==0 or epoch==0:
            #Iterate through all num_tr training patients and classify based off difference in probability of PD/HC
            num_tr=0
            y_pred_tag=[]
            for trainItr in rand_range:   
                if trainItr in valIds:
                    continue

                if trainItr<num_pd:
                    trainIndc=1
                    trainOpp=0
                    spk_typ='pd'
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

                        #Find difference in probability of PD v. HC for all segments. 
                        y_pred_tag_curr.extend((y_tr_pred[:,0]-y_tr_pred[:,1]).cpu().detach().numpy())

                if num_tr==0:
                    indcs_vec=np.ones(len(y_pred_tag_curr))*trainIndc
                    num_tr=1
                else:
                    indcs_vec=np.concatenate((indcs_vec,np.ones(len(y_pred_tag_curr))*trainIndc))
                    num_tr+=1
                y_pred_tag.extend(y_pred_tag_curr)

            #fit SGD classifier on ALL training frames (for all spks/utters).
            y_pred_tag=np.array(y_pred_tag).reshape(-1,1)
            clf = SGDClassifier(loss="hinge", penalty="l2")
            clf.fit(y_pred_tag, indcs_vec) 
            #training acc. determined based on # of frames classified correctly
            tr_acc=clf.score(y_pred_tag,indcs_vec)

            #CalibratedClassifier allows for binary prediction with SGD classifier. Used for test speakers
            #and in analysis, making ultimate decision on classifying spk as pd/hc.
            if epoch==N_EPOCHS-1:
                calibrator=CalibratedClassifierCV(clf, cv='prefit')
                modCal=calibrator.fit(y_pred_tag, indcs_vec)

            #Validate at end of each epoch for nv speakers
            val_loss=0.0
            num_val=0
            y_pred_tag=[]
            for vid in valDict.keys():
                if vid<num_pd:
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

                        #Find difference in probability of PD v. HC for all segments. 
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
    for spkItr,spk in enumerate(['pd','hc']):
        dic=testDict[spk]

        for tstId in dic.keys():
            if tstId<num_pd:
                indc=1
                opp=0
            else:
                indc=0
                opp=1
            test_loss_curr=0
            y_pred_tag_curr=[]
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
                    y_pred_tag_curr.extend((y_test_pred[:,0]-y_test_pred[:,1]).cpu().detach().numpy())

                    if torch.cuda.is_available():
                        loss = criterion(y_test_pred, torch.from_numpy(yTest).cuda().float())
                    else:
                        loss = criterion(y_test_pred, torch.from_numpy(yTest).float())
                    test_loss_curr+=loss.item()*X_test.size(0)

            #accuracy determined on majority rule (wlog, if more frames yield probability differences greater than 0,
            #and if speaker is PD than classification assessment is correct (1=>PD,0=>HC).
            test_loss+=test_loss_curr/len(test_loader.dataset)
            if num_test==0:
                indcs_vec=np.ones(len(y_pred_tag_curr))*indc
                num_test=1
            else:
                indcs_vec=np.concatenate((indcs_vec,np.ones(len(y_pred_tag_curr))*indc))
                num_test+=1
            y_pred_tag.extend(y_pred_tag_curr)

            #Store raw scores for each test speaker (probability of PD and HC as output by dnn) for ROC.
#                 tst_diffs=(y_test_pred[:,0]-y_test_pred[:,1]).cpu().detach().numpy().reshape(-1,1)
            testResults['tstSpk_data'][tstId]=calibrator.predict_proba(np.array(y_pred_tag_curr).reshape(-1,1))


    y_pred_tag=np.array(y_pred_tag).reshape(-1,1)
    test_acc=clf.score(y_pred_tag,indcs_vec)
    #Store and report loss and accuracy for batch of test speakers.            
    testResults['test_loss'],testResults['test_acc']=test_loss/40,test_acc
    testResults['class_report']=classification_report(indcs_vec,calibrator.predict(y_pred_tag))
    print('\nTest Loss: {:.3f} \tTest Acc: {:.3f} '.format(
                test_loss/40,
                test_acc
        ))

    train_res.append(trainResults_epo)
    itr=0
    trainResults=pd.concat(train_res,keys=(np.arange(itr+1)))

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
    

    trainResults.to_pickle(save_path+mod+'ind_mcFusion_trainResults.pkl', protocol=4)
    testResults.to_pickle(save_path+mod+'ind_mcFusion_testResults.pkl', protocol=4)
