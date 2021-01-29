import os
import sys
import numpy as np
import pandas as pd
import pickle
import random
import pdb
from AEspeech import AEspeech
from scipy.stats import kurtosis, skew
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
import json
import argparse


PATH=os.path.dirname(os.path.abspath(__file__))

#LOAD CONFIG.JSON INFO
with open("config.json") as f:
    data = f.read()
config = json.loads(data)
UNITS=config['general']['UNITS']
UTTERS=['pataka','kakaka','pakata','papapa','petaka','tatata']
MODELS=["CAE","RAE","ALL"]
REPS=['broadband','narrowband','wvlt']
# UTTERS=['pataka']


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
       

    
if __name__=="__main__":

    if len(sys.argv)!=4:
        print("python pdsvmEval.py <'CAE','RAE', or 'ALL'> <'broadband', 'narrowband' or 'wvlt'> <pd path>")
        sys.exit()        
    #TRAIN_PATH: './pdSpanish/speech/'    
    
    if sys.argv[1] in MODELS:
        model=sys.argv[1]
    else:
        print("python pdsvmEval.py <'CAE','RAE', or 'ALL'> <'broadband', 'narrowband' or 'wvlt'> <pd path>")
        sys.exit()
    
    if sys.argv[2] in REPS:
        rep=sys.argv[2]
    else:
        print("python pdsvmEval.py <'CAE','RAE', or 'ALL'> <'broadband', 'narrowband' or 'wvlt'> <pd path>")
        sys.exit()    
          
    if sys.argv[3][0] !='/':
        sys.argv[3] = '/'+sys.argv[3]
    if sys.argv[3][-1] !='/':
        sys.argv[3] = sys.argv[3]+'/'
        
    save_path=PATH+"/pdSpanish/classResults/svm/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
#     mfda_path=PATH+"/pdSpanish/"
#     mfdas=pd.read_csv(mfda_path+"metadata-Spanish_All.csv")['M-FDA'].values
#     pd_mfdas=mfdas[0:50]
#     hc_mfdas=mfdas[50:]

    if rep=='wvlt':
        num_feats=config['wavelet']['NBF']+UNITS
    else:
        num_feats=config['mel_spec']['INTERP_NMELS']+UNITS
    
    o_itrs=config['svm']['iterations']
    results=pd.DataFrame({utter:{'train_acc':0,'test_acc':0,'bin_class':{itr:{} for itr in range(o_itrs)},'class_report':{itr:{} for itr in range(o_itrs)}} for utter in UTTERS})
 
    #Run train/test for each utterance case separately.
    for uIdx,utter in enumerate(UTTERS):
        curr_best=0
        pd_path=PATH+sys.argv[3]+'/'+utter+"/pd/"
        hc_path=PATH+sys.argv[3]+'/'+utter+"/hc/"   
        pdNames=[name for name in os.listdir(pd_path) if '.wav' in name]
        hcNames=[name for name in os.listdir(hc_path) if '.wav' in name]
        pdNames.sort()
        hcNames.sort()
        spks=pdNames+hcNames
        num_spks=len(spks)
        num_pd=len(pdNames)
        num_hc=len(hcNames)
        
        #get features
        pdFeats=getFeats(model,UNITS,rep,pd_path,utter,'pd')
        hcFeats=getFeats(model,UNITS,rep,hc_path,utter,'hc')
        pdAll=np.unique(pdFeats['wav_file'])
        hcAll=np.unique(hcFeats['wav_file'])
        pdIds=np.arange(50)
        hcIds=np.arange(50,100)
        
        #store speech frames per speaker separately
        pds=np.zeros((len(pdAll),num_feats,4))
        hcs=np.zeros((len(hcAll),num_feats,4))
        #getting bottle neck features and reconstruction error for training
        for ii,tr in enumerate(pdAll):
            tritr=pdIds[ii]
            pdTrBns=pdFeats['bottleneck'][np.where(pdFeats['wav_file']==spks[tritr])]
            pdTrBns=np.array([np.mean(pdTrBns,axis=0),np.std(pdTrBns,axis=0),skew(pdTrBns,axis=0),kurtosis(pdTrBns,axis=0)])
            pdTrErrs=pdFeats['error'][np.where(pdFeats['wav_file']==spks[tritr])]
            pdTrErrs=np.array([np.mean(pdTrErrs,axis=0),np.std(pdTrErrs,axis=0),skew(pdTrErrs,axis=0),kurtosis(pdTrErrs,axis=0)])
            pds[ii,:,:]=np.concatenate((pdTrBns,pdTrErrs),axis=1).T
            if np.isfinite(pdTrBns).all() and np.isfinite(pdTrErrs).all():
                continue
            else:
                pdTrErrs=pdTrErrs[:-1,:]
        for ii,tr in enumerate(hcAll):
            tritr=hcIds[ii]
            hcTrBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==spks[tritr])]
            hcTrBns=np.array([np.mean(hcTrBns,axis=0),np.std(hcTrBns,axis=0),skew(hcTrBns,axis=0),kurtosis(hcTrBns,axis=0)])
            hcTrErrs=hcFeats['error'][np.where(hcFeats['wav_file']==spks[tritr])]
            hcTrErrs=np.array([np.mean(hcTrErrs,axis=0),np.std(hcTrErrs,axis=0),skew(hcTrErrs,axis=0),kurtosis(hcTrErrs,axis=0)])
            hcs[ii,:,:]=np.concatenate((hcTrBns,hcTrErrs),axis=1).T
            
        #reshape and prep feature set for PCA
        pdXAll=np.reshape(pds,(pds.shape[0],num_feats*4))
        hcXAll=np.reshape(hcs,(hcs.shape[0],num_feats*4))  
        xAll=np.concatenate((pdXAll,hcXAll),axis=0)
        st_xAll=StandardScaler().fit_transform(pd.DataFrame(xAll))
        
        pca = PCA(n_components=min(xAll.shape[0],xAll.shape[1]))
        pca.fit_transform(st_xAll)
        variance = pca.explained_variance_ratio_ #calculate variance ratios
        var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
        ncs=np.count_nonzero(var<90)
        pca = PCA(n_components=ncs)
        pca_xAll=pca.fit_transform(st_xAll)

        #split data into training and test with multiple iterations (90 training, 10 test per iter and evenly split PD:HC)
        num_pdHc_tests=config['svm']['tst_spks']#must be even (same # of test pds and hcs per iter)
        val_size=config['svm']['val_size']
        nsplits=config['svm']['nsplits']
        
        i_itrs=num_spks//num_pdHc_tests
        if  np.mod(num_pdHc_tests,2)!=0:
            print("number of test spks must be even...")
            sys.exit()
        if  np.mod(num_spks,num_pdHc_tests)!=0:
            print("number of test spks must be a divisor of num_spks...")
            sys.exit()
        
        for o_itr in range(o_itrs):
            pd_files=pdNames
            hc_files=hcNames
            for itr in range(i_itrs):
                rand_range=np.arange(num_spks)
                random.shuffle(rand_range)

                pdCurrs=[pd_files[idx] for idx in random.sample(range(0,len(pd_files)),int(num_pdHc_tests/2))]
                hcCurrs=[hc_files[idx] for idx in random.sample(range(0,len(hc_files)),int(num_pdHc_tests/2))]
                pd_files=[pd for pd in pd_files if pd not in pdCurrs]
                hc_files=[hc for hc in hc_files if hc not in hcCurrs]

                pdIds=[spks.index(pdCurr) for pdCurr in pdCurrs]
                hcIds=[spks.index(hcCurr) for hcCurr in hcCurrs]

                pdTest=np.zeros((num_pdHc_tests//2,ncs))
                hcTest=np.zeros((num_pdHc_tests//2,ncs))
                for ii,pdItr in enumerate(pdIds):
                    pdTest[ii,:]=pca_xAll[pdItr,:]
                for ii,hcItr in enumerate(hcIds):
                    hcTest[ii,:]=pca_xAll[hcItr,:]

                pdTrainees=[spk for idx,spk in enumerate(pdNames) if spk not in pdCurrs]
                hcTrainees=[spk for idx,spk in enumerate(hcNames) if spk not in hcCurrs]
                pdTrainIds=[spks.index(tr) for tr in pdTrainees]
                hcTrainIds=[spks.index(tr) for tr in hcTrainees]

                pdTrain=np.zeros((num_pd-int(num_pdHc_tests/2),ncs))
                hcTrain=np.zeros((num_hc-int(num_pdHc_tests/2),ncs))
                for ii,pdItr in enumerate(pdTrainIds):
                    pdTrain[ii,:]=pca_xAll[pdItr,:]
                for ii,hcItr in enumerate(hcTrainIds):
                    hcTrain[ii,:]=pca_xAll[hcItr,:]

                xTrain=np.concatenate((pdTrain,hcTrain),axis=0)
                xTrain=(xTrain-np.min(xTrain))/(np.max(xTrain)-np.min(xTrain))
                pdYTrain=np.ones((pdTrain.shape[0])).T
                hcYTrain=np.zeros((hcTrain.shape[0])).T
                yTrain=np.concatenate((pdYTrain,hcYTrain),axis=0)

                xTest=np.concatenate((pdTest,hcTest),axis=0)
                xTest=(xTest-np.min(xTest))/(np.max(xTest)-np.min(xTest))

                pdYTest=np.ones((pdTest.shape[0])).T
                hcYTest=np.zeros((pdTest.shape[0])).T
                yTest=np.concatenate((pdYTest,hcYTest),axis=0)

                param_grid = [
#                 {'C':np.logspace(-8,8,15), 'gamma':np.logspace(-8,4,15), 'degree':[1],'kernel': ['rbf']}
              {'C':np.logspace(0,5,13), 'gamma':np.logspace(-8,-4,13), 'degree':[1],'kernel': ['rbf']},
                ]

                cv = StratifiedShuffleSplit(n_splits=nsplits, test_size=val_size, random_state=42)
        #         grid = GridSearchCV(SVC(),scoring = my_auc, param_grid=param_grid, cv=cv)
                grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv)
                grid.fit(xTrain, yTrain)
    #             grid=svm.SVC(C=grid.best_params_['C'],degree=grid.best_params_['degree'],gamma=grid.best_params_['gamma'],
    #                                 kernel=grid.best_params_['kernel'], probability=True)
    #             grid.fit(xTrain,yTrain)

#                 train_acc=grid.score(xTrain,yTrain)
#                 test_acc=grid.score(xTest,yTest)
            
                tr_bin_class=grid.predict_proba(xTrain)
                train_acc=0
                opt_thresh=0
                diffs=tr_bin_class[:,0]-tr_bin_class[:,1]
                for thresh in range(-50,50):
                    thresh=thresh/100
                    g_locs=np.where(diffs>=thresh)
                    l_locs=np.where(diffs<thresh)
                    if sum(diffs[0:pdYTrain.shape[0]])<0:
                        acc_curr=len(np.where(l_locs[0]<pdYTrain.shape[0])[0])
                        acc_curr+=len(np.where(g_locs[0]>=pdYTrain.shape[0])[0])
                    else:
                        acc_curr=len(np.where(l_locs[0]>=pdYTrain.shape[0])[0])
                        acc_curr+=len(np.where(g_locs[0]<pdYTrain.shape[0])[0])
                    
                    if acc_curr/yTrain.shape[0]>train_acc:
                        opt_thresh=thresh
                        train_acc=acc_curr/yTrain.shape[0]
                
                bin_class=grid.predict_proba(xTest)
                tst_diffs=bin_class[:,0]-bin_class[:,1]
                tst_g_locs=np.where(tst_diffs>=opt_thresh)
                tst_l_locs=np.where(tst_diffs<opt_thresh)
                if sum(diffs[0:pdYTrain.shape[0]])<0:
                    test_acc=len(np.where(tst_l_locs[0]<pdYTest.shape[0])[0])/yTest.shape[0]
                    test_acc+=len(np.where(tst_g_locs[0]>=pdYTest.shape[0])[0])/yTest.shape[0]
                else:
                    test_acc=len(np.where(tst_l_locs[0]>=pdYTest.shape[0])[0])/yTest.shape[0]
                    test_acc+=len(np.where(tst_g_locs[0]<pdYTest.shape[0])[0])/yTest.shape[0]
                
                class_report=classification_report(yTest,grid.predict(xTest))
                results[utter]['train_acc']+=train_acc*(1/(i_itrs*o_itrs))
                results[utter]['test_acc']+=test_acc*(1/(i_itrs*o_itrs))
                results[utter]['class_report'][o_itr][itr]=class_report  
                for cpi,(pdId,hcId) in enumerate(zip(pdIds,hcIds)):          
                    results[utter]['bin_class'][o_itr][pdId]=bin_class[cpi]     
                    results[utter]['bin_class'][o_itr][hcId]=bin_class[cpi+int(num_pdHc_tests/2)]

#         pdb.set_trace()
        results.to_pickle(save_path+model+'_'+rep+"Results.pkl")
    
    



    