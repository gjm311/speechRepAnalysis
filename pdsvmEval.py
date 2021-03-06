import os
import sys
import numpy as np
import pandas as pd
import pickle
import random
import pdb
import itertools
from AEspeech import AEspeech
from scipy import stats
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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.calibration import CalibratedClassifierCV
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
       
def featAgg(model,rep,spk_path,num_feats,ef,feats):
    num_utters=len(UTTERS)
    pds=np.zeros((50*num_utters,num_feats,4))
    hcs=np.zeros((50*num_utters,num_feats,4))
    if ef==0:
        reps=[rep]
    else:
        reps=rep
        
    for uIdx,utter in enumerate(UTTERS):
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
        for rIdx,rep in enumerate(reps):
            pdFeats=getFeats(model,UNITS,rep,pd_path,utter,'pd')
            hcFeats=getFeats(model,UNITS,rep,hc_path,utter,'hc')
            pdAll=np.unique(pdFeats['wav_file'])
            hcAll=np.unique(hcFeats['wav_file'])
            pdIds=np.arange(50)
            hcIds=np.arange(50,100)
            if ef==1:
                if rep=='wvlt':
                    cntr=feats[1]
                else:
                    cntr=feats[0]
            #getting bottle neck features and reconstruction error for training
            for ii,tr in enumerate(pdAll):
                tritr=pdIds[ii]
                pdTrBns=pdFeats['bottleneck'][np.where(pdFeats['wav_file']==spks[tritr])]
                pdTrBns=np.array([np.mean(pdTrBns,axis=0),np.std(pdTrBns,axis=0),skew(pdTrBns,axis=0),kurtosis(pdTrBns,axis=0)])
                pdTrErrs=pdFeats['error'][np.where(pdFeats['wav_file']==spks[tritr])]
                pdTrErrs=np.array([np.mean(pdTrErrs,axis=0),np.std(pdTrErrs,axis=0),skew(pdTrErrs,axis=0),kurtosis(pdTrErrs,axis=0)])
                if ef==0:
                    pds[(ii*num_utters)+uIdx,:,:]=np.concatenate((pdTrBns,pdTrErrs),axis=1).T
                else:
                    pds[(ii*num_utters)+uIdx,rIdx*cntr:(rIdx+1)*cntr,:]=np.concatenate((pdTrBns,pdTrErrs),axis=1).T
            for ii,tr in enumerate(hcAll):
                tritr=hcIds[ii]
                hcTrBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==spks[tritr])]
                hcTrBns=np.array([np.mean(hcTrBns,axis=0),np.std(hcTrBns,axis=0),skew(hcTrBns,axis=0),kurtosis(hcTrBns,axis=0)])
                hcTrErrs=hcFeats['error'][np.where(hcFeats['wav_file']==spks[tritr])]
                hcTrErrs=np.array([np.mean(hcTrErrs,axis=0),np.std(hcTrErrs,axis=0),skew(hcTrErrs,axis=0),kurtosis(hcTrErrs,axis=0)])
                if ef==0:
                    hcs[(ii*num_utters)+uIdx,:,:]=np.concatenate((hcTrBns,hcTrErrs),axis=1).T
                else:
                    hcs[(ii*num_utters)+uIdx,rIdx*cntr:(rIdx+1)*cntr,:]=np.concatenate((hcTrBns,hcTrErrs),axis=1).T
    
    pdXAll=np.reshape(pds,(pds.shape[0],num_feats*4))
    hcXAll=np.reshape(hcs,(hcs.shape[0],num_feats*4))  
    xAll=np.concatenate((pdXAll,hcXAll),axis=0)
    
    return xAll,pdNames,hcNames
    
    
    
if __name__=="__main__":

    if len(sys.argv)!=4:
        print("python pdsvmEval.py <'CAE','RAE'> <broadband, narrowband, wvlt, early_fuse2,early_fuse3, mc_fuse> <pd path>")
        sys.exit()        
    #TRAIN_PATH: './pdSpanish/speech/'    
    
    if sys.argv[1] in MODELS:
        mod=sys.argv[1]
    else:
        print("python pdsvmEval.py <'CAE','RAE', or 'ALL'> <broadband, narrowband, wvlt, early_fuse2,early_fuse3, mc_fuse> <pd path>")
        sys.exit()
    
    if sys.argv[2] not in ['broadband', 'narrowband', 'wvlt', 'early_fuse2', 'early_fuse3', 'mc_fuse']:
        print("python pdsvmEval.py <'CAE','RAE', or 'ALL'> <broadband, narrowband, wvlt, early_fuse2,early_fuse3, mc_fuse> <pd path>")
        sys.exit()
    else:
        rep_typ=sys.argv[2]
    
    if sys.argv[3][0] !='/':
        sys.argv[3] = '/'+sys.argv[3]
    if sys.argv[3][-1] !='/':
        sys.argv[3] = sys.argv[3]+'/'
    spk_path=PATH+sys.argv[3]
        
    save_path=PATH+"/pdSpanish/classResults/svm/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    ef=0
    if rep_typ in ['broadband','narrowband','wvlt']:
        rep=rep_typ
        feats=[]
        if rep=='wvlt':
            num_feats=config['wavelet']['NBF']+UNITS
        else:
            num_feats=config['mel_spec']['INTERP_NMELS']+UNITS
    elif rep_typ=='mc_fuse':
        feats=[]
        rep='mc_fuse'
        num_feats=2*config['mel_spec']['INTERP_NMELS']+UNITS
    elif rep_typ=='early_fuse2':
        rep=['broadband','narrowband']
        num_feats=2*(config['mel_spec']['INTERP_NMELS']+UNITS)
        feats=[config['mel_spec']['INTERP_NMELS']+UNITS]
        ef=1
    elif rep_typ=='early_fuse3':
        rep=['broadband','narrowband','wvlt']
        num_feats=2*(config['mel_spec']['INTERP_NMELS']+UNITS)+config['wavelet']['NBF']+UNITS
        feats=[config['mel_spec']['INTERP_NMELS']+UNITS, config['wavelet']['NBF']+UNITS]
        ef=1
        
    #get compressed data, n_components, and file_name list 
    xAll,pdNames,hcNames=featAgg(mod,rep,spk_path,num_feats,ef,feats)
    spks=pdNames+hcNames
    num_spks=len(spks)
    num_pd=len(pdNames)
    num_hc=len(hcNames)
    
    mfda_path=PATH+"/pdSpanish/"
    mfdas=pd.read_csv(mfda_path+"metadata-Spanish_All.csv")['M-FDA'].values
    mfda_simp=mfdas
    
    #split data into training and test with multiple iterations
    num_pdHc_tests=config['svm']['tst_spks']#must be even (same # of test pds and hcs per iter)
    nv=config['svm']['val_size']#number of validation speakers per split#must be even and a divisor of num_spks (same # of test pds and hcs per iter)
    if  np.mod(num_pdHc_tests,2)!=0:
        print("number of test spks must be even...")
        sys.exit()
    if  np.mod(100,num_pdHc_tests)!=0:
        print("number of test spks must be a divisor of 100...")
        sys.exit()
    
    num_utters=len(UTTERS)
    total_itrs=config['svm']['iterations']
    results=pd.DataFrame({'Data':{'train_acc':0,'test_acc':0,'mFDA_spear_corr':{itr:{idx:{utter:0 for utter in UTTERS} for idx in np.arange(100)} for itr in range(total_itrs)},'bin_class':{itr:{} for itr in range(total_itrs)},'class_report':{itr:{} for itr in range(total_itrs)}}})
    
    for o_itr in range(total_itrs):
        pd_files=pdNames
        hc_files=hcNames        
        
        for itr in range(int(num_spks/num_pdHc_tests)):
            pdCurrs=[pd_files[idx] for idx in random.sample(range(0,len(pd_files)),int(num_pdHc_tests/2))]
            hcCurrs=[hc_files[idx] for idx in random.sample(range(0,len(hc_files)),int(num_pdHc_tests/2))]
            pd_files=[pd for pd in pd_files if pd not in pdCurrs]
            hc_files=[hc for hc in hc_files if hc not in hcCurrs]

            pdIds=[spks.index(pdCurr) for pdCurr in pdCurrs]
            hcIds=[spks.index(hcCurr) for hcCurr in hcCurrs]
            
            pdTest=np.zeros(((num_pdHc_tests//2)*num_utters,xAll.shape[1]))
            hcTest=np.zeros(((num_pdHc_tests//2)*num_utters,xAll.shape[1]))
            for ii,pdItr in enumerate(pdIds):
                pdTest[ii*num_utters:(ii+1)*num_utters,:]=xAll[pdItr*num_utters:(pdItr+1)*num_utters,:]
            for ii,hcItr in enumerate(hcIds):
                hcTest[ii*num_utters:(ii+1)*num_utters,:]=xAll[hcItr*num_utters:(hcItr+1)*num_utters,:]

            pdTrainees=[spk for idx,spk in enumerate(pdNames) if spk not in pdCurrs]
            hcTrainees=[spk for idx,spk in enumerate(hcNames) if spk not in hcCurrs]
            pdTrainIds=[spks.index(tr) for tr in pdTrainees]
            hcTrainIds=[spks.index(tr) for tr in hcTrainees]

            pdTrain=np.zeros(((num_pd-int(num_pdHc_tests/2))*num_utters,xAll.shape[1]))
            hcTrain=np.zeros(((num_hc-int(num_pdHc_tests/2))*num_utters,xAll.shape[1]))
            for ii,pdItr in enumerate(pdTrainIds):
                pdTrain[ii*num_utters:(ii+1)*num_utters,:]=xAll[pdItr*num_utters:(pdItr+1)*num_utters,:]
            for ii,hcItr in enumerate(hcTrainIds):
                hcTrain[ii*num_utters:(ii+1)*num_utters,:]=xAll[hcItr*num_utters:(hcItr+1)*num_utters,:]
                
            xTrain=np.concatenate((pdTrain,hcTrain),axis=0)
            xTest=np.concatenate((pdTest,hcTest),axis=0)
            
            st_xTrain=StandardScaler().fit_transform(pd.DataFrame(xTrain))   
            st_xTest=StandardScaler().fit_transform(pd.DataFrame(xTest))   
            pca = PCA(n_components=min(xTrain.shape[0],xTrain.shape[1]))
            pca.fit_transform(st_xTrain)
            variance = pca.explained_variance_ratio_ #calculate variance ratios
            var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
            ncs=np.count_nonzero(var<90)
            pca = PCA(n_components=ncs)
            xTrain=pca.fit_transform(st_xTrain)
            xTest=pca.transform(st_xTest)
            
            pdYTrain=np.ones((pdTrain.shape[0])).T
            hcYTrain=np.zeros((hcTrain.shape[0])).T
            yTrain=np.concatenate((pdYTrain,hcYTrain),axis=0)
            
            pdYTest=np.ones((pdTest.shape[0])).T
            hcYTest=np.zeros((pdTest.shape[0])).T
            yTest=np.concatenate((pdYTest,hcYTest),axis=0)

            mfda_yTrain=list(itertools.chain.from_iterable(itertools.repeat(x, num_utters) for x in mfda_simp[pdTrainIds+hcTrainIds]))
            mfda_yTest=list(itertools.chain.from_iterable(itertools.repeat(x, num_utters) for x in mfda_simp[pdIds+hcIds]))
            
            param_grid = [
              {'C':np.logspace(0,5,25), 'gamma':np.logspace(-8,-4,25), 'degree':[1],'kernel': ['rbf']},
                ]

            cv = StratifiedShuffleSplit(n_splits=4, test_size=0.2, random_state=42)
            
            grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv)
            mfda_grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv)
            grid.fit(xTrain, yTrain)
            mfda_grid.fit(xTrain, mfda_yTrain)
            grid=svm.SVC(C=grid.best_params_['C'],degree=grid.best_params_['degree'],gamma=grid.best_params_['gamma'],
                                kernel=grid.best_params_['kernel'], probability=True)
            mfda_grid=svm.SVC(C=mfda_grid.best_params_['C'],degree=mfda_grid.best_params_['degree'],gamma=mfda_grid.best_params_['gamma'],
                                kernel=mfda_grid.best_params_['kernel'], probability=True)
            grid.fit(xTrain,yTrain)
            mfda_grid.fit(xTrain,mfda_yTrain)
            
            #predict probability of training, get differences and find optimal threshold
            tr_bin_class=grid.predict_proba(xTrain)
            diffs=tr_bin_class[:,0]-tr_bin_class[:,1]
            clf = SGDClassifier(loss="hinge", penalty="l2")
            diffs=np.array(diffs).reshape(-1,1)
            clf.fit(diffs, yTrain)
            train_acc=clf.score(diffs,yTrain)
            calibrator=CalibratedClassifierCV(clf, cv='prefit')
            modCal=calibrator.fit(diffs, yTrain)
            
            #predict probability of test speakers using optimal thresh
            tst_bin_class=grid.predict_proba(xTest)
            tst_diffs=tst_bin_class[:,0]-tst_bin_class[:,1]
            tst_diffs=np.array(tst_diffs).reshape(-1,1)
            test_acc=clf.score(tst_diffs,yTest)
            bin_class=calibrator.predict_proba(tst_diffs)
            
            #predict mfdas
            for idItr,curr_id in enumerate(pdIds+hcIds):
                preds=mfda_grid.predict(xTest[idItr*num_utters:(idItr+1)*num_utters,:])
                for i_utter_idx,utter in enumerate(UTTERS):
                    results['Data']['mFDA_spear_corr'][o_itr][curr_id][utter]=preds[i_utter_idx]

            class_report=classification_report(yTest,grid.predict(xTest))
            results['Data']['train_acc']+=train_acc*(1/(int(num_spks/num_pdHc_tests)*total_itrs))
            results['Data']['test_acc']+=test_acc*(1/(int(num_spks/num_pdHc_tests)*total_itrs))
            results['Data']['class_report'][o_itr][itr]=class_report  
            for cpi,(pdId,hcId) in enumerate(zip(pdIds,hcIds)):          
                results['Data']['bin_class'][o_itr][pdId]=bin_class[cpi*num_utters:(cpi+1)*num_utters]     
                results['Data']['bin_class'][o_itr][hcId]=bin_class[(cpi+num_pdHc_tests//2)*num_utters:(cpi+(num_pdHc_tests//2)+1)*num_utters]
                
                
        if rep_typ=='mc_fuse':
            results.to_pickle(save_path+mod+"_mcFusionResults.pkl", protocol=4)
        if rep_typ in ['broadband','narrowband','wvlt']:
            results.to_pickle(save_path+mod+'_'+rep+"_aggResults.pkl", protocol=4)
        if rep_typ=='early_fuse3':
            results.to_pickle(save_path+mod+"_wvlt_earlyFusionResults.pkl", protocol=4)
        elif rep_typ=='early_fuse2':
            results.to_pickle(save_path+mod+"_earlyFusionResults.pkl", protocol=4)
                



