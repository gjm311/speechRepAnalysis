import os
import sys
import numpy as np
import pandas as pd
import pickle
import random
import pdb
import itertools
from clpAEspeech import AEspeech
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import joblib
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
import json
import argparse


PATH=os.path.dirname(os.path.abspath(__file__))
#LOAD CONFIG.JSON INFO
with open("clpConfig.json") as f:
    data = f.read()
config = json.loads(data)
UNITS=config['general']['UNITS']
UTTERS=['bola','choza','chuzo','coco','gato','jugo','mano','papa','susi']
MODELS=["CAE","RAE","ALL"]
REPS=['narrowband','broadband','wvlt']


def saveFeats(model,units,rep,wav_path,utter,save_path, spk_typ):
    global UNITS    
    # load the pretrained model with 256 units and get temp./freq. rep (spec or wvlt)
    aespeech=AEspeech(model=model,units=UNITS,rep=rep) 
    
    #compute the bottleneck and error-based features from a directory with wav files inside 
    #(dynamic: one feture vector for each 500 ms frame)
    #(global i.e. static: one feture vector per utterance)
    feat_vecs=aespeech.compute_dynamic_features(wav_path)
    #     df1, df2=aespeech.compute_global_features(wav_path)
    
    with open(save_path+'/'+rep+'_'+model+'_'+spk_typ+'svmFeats.pickle', 'wb') as handle:
        pickle.dump(feat_vecs, handle, protocol=4)
    
    return feat_vecs
            

def getFeats(model,units,rep,wav_path,utter,spk_typ):
    global PATH
    save_path=PATH+"/"+"clpSpanish/feats/"+utter+"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if os.path.isfile(save_path+'/'+rep+'_'+model+'_'+spk_typ+'svmFeats.pickle'):
        with open(save_path+'/'+rep+'_'+model+'_'+spk_typ+'svmFeats.pickle', 'rb') as handle:
            feat_vecs = pickle.load(handle)
    else:
        feat_vecs=saveFeats(model,units,rep,wav_path,utter,save_path, spk_typ)
    
    return feat_vecs
       

def featAgg(model,reps,spk_path):
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
    
    clpNames_count=dict(collections.Counter([name for name in allClpNames]))
    hcNames_count=dict(collections.Counter([name for name in allHcNames]))
    clpNames=list(np.unique(allClpNames))
    hcNames=list(np.unique(allHcNames))
    
    #aggregate all spk data per rep and store pca compressed data and ncs
    rep_dict={rep:{'data':[], 'ncs':0} for rep in reps}
    for rIdx,rep in enumerate(reps):
        clpKeys=np.zeros(len(clpNames))
        hcKeys=np.zeros(len(hcNames))
        spkdict={spk:[] for spk in ['clp','hc']}
        spkdict['clp']={clp:[] for clp in clpNames_count.keys()}
        spkdict['hc']={hc:[] for hc in hcNames_count.keys()}
        for itr,name in enumerate(clpNames_count.keys()):
            spkdict['clp'][name]=[]
        for itr,name in enumerate(hcNames_count.keys()):
            spkdict['hc'][name]=[]
         
        for uIdx,utter in enumerate(UTTERS):
            clp_path=spk_path+'/'+utter+"/clp/"
            hc_path=spk_path+'/'+utter+"/hc/"                 
            clpFeats=getFeats(model,UNITS,rep,clp_path,utter,'clp')
            hcFeats=getFeats(model,UNITS,rep,hc_path,utter,'hc')
            clpAll=[c.split('_')[0] for c in np.unique(clpFeats['wav_file'])]
            hcAll=[h.split('_')[0] for h in np.unique(hcFeats['wav_file'])]
            
            for ii,tr in enumerate(clpAll):
                inner_ks=np.array([c.split('_')[0] for c in clpFeats['wav_file']])
                clpTrBns=clpFeats['bottleneck'][np.where(inner_ks==tr.split('_')[0])]
                clpTrBns=np.array([np.mean(clpTrBns,axis=0),np.std(clpTrBns,axis=0),skew(clpTrBns,axis=0),kurtosis(clpTrBns,axis=0)])
                clpTrErrs=clpFeats['error'][np.where(inner_ks==tr.split('_')[0])]
                clpTrErrs=np.array([np.mean(clpTrErrs,axis=0),np.std(clpTrErrs,axis=0),skew(clpTrErrs,axis=0),kurtosis(clpTrErrs,axis=0)])
                ovrall_idx=clpNames.index(tr.split('_')[0])
                if clpKeys[ovrall_idx]==0:
                    spkdict['clp'][tr.split('_')[0]]=np.expand_dims(np.concatenate((clpTrBns,clpTrErrs),axis=1).T,axis=0)
                    clpKeys[ovrall_idx]=1
                else:
                    new=np.concatenate((clpTrBns,clpTrErrs),axis=1).T
                    spkdict['clp'][tr.split('_')[0]]=np.concatenate((spkdict['clp'][tr.split('_')[0]],np.expand_dims(new,axis=0)),axis=0)
                    
            for ii,tr in enumerate(hcAll):
                inner_ks=np.array([h.split('_')[0] for h in hcFeats['wav_file']])
                hcTrBns=hcFeats['bottleneck'][np.where(inner_ks==tr.split('_')[0])]
                hcTrBns=np.array([np.mean(hcTrBns,axis=0),np.std(hcTrBns,axis=0),skew(hcTrBns,axis=0),kurtosis(hcTrBns,axis=0)])
                hcTrErrs=hcFeats['error'][np.where(inner_ks==tr.split('_')[0])]
                hcTrErrs=np.array([np.mean(hcTrErrs,axis=0),np.std(hcTrErrs,axis=0),skew(hcTrErrs,axis=0),kurtosis(hcTrErrs,axis=0)])
                ovrall_idx=hcNames.index(tr.split('_')[0])
                if hcKeys[ovrall_idx]==0:
                    spkdict['hc'][tr.split('_')[0]]=np.expand_dims(np.concatenate((hcTrBns,hcTrErrs),axis=1).T,axis=0)
                    hcKeys[ovrall_idx]=1
                else:
                    new=np.concatenate((hcTrBns,hcTrErrs),axis=1).T
                    spkdict['hc'][tr.split('_')[0]]=np.concatenate((spkdict['hc'][tr.split('_')[0]],np.expand_dims(new,axis=0)),axis=0)
        
        rep_dict[rep]['data']=getDataset(spkdict,clpNames,hcNames)
    
    xAlls=[rep_dict[rep]['data'] for rep in reps]
    return xAlls,clpNames,hcNames,{**clpNames_count,**hcNames_count}
    
    
def getDataset(spkdict,clpNames,hcNames):
    num_spks=len(clpNames)    
    
    for ni,name in enumerate(clpNames):
        if ni==0:
            clpFeats=spkdict['clp'][name]
        else:
            clpFeats=np.concatenate((clpFeats,spkdict['clp'][name]),axis=0)
    for ni,name in enumerate(hcNames):
        if ni==0:
            hcFeats=spkdict['hc'][name]
        else:
            hcFeats=np.concatenate((hcFeats,spkdict['hc'][name]),axis=0)
    
    clpXAll=np.reshape(clpFeats,(clpFeats.shape[0],clpFeats.shape[1]*4))
    hcXAll=np.reshape(hcFeats,(hcFeats.shape[0],hcFeats.shape[1]*4))  
    xAll=np.concatenate((clpXAll,hcXAll),axis=0)
    return xAll    
    
    
if __name__=="__main__":

    if len(sys.argv)!=4:
        print("python clpsvmLateFusion.py <'CAE','RAE', or 'ALL'> <nreps - 2 (nb/bb) or 3 (nb/bb/wvlt)> <clp path>")
        sys.exit()        
    #TRAIN_PATH: './clpSpanish/speech/'    
    
    if sys.argv[1] in MODELS:
        model=sys.argv[1]
    else:
        print("python clpsvmLateFusion.py <'CAE','RAE', or 'ALL'> <nreps - 2 (nb/bb) or 3 (nb/bb/wvlt)> <clp path>")
        sys.exit() 
    
    if int(sys.argv[2]) not in [2,3]:
        print("python clpsvmLateFusion.py <'CAE','RAE', or 'ALL'> <nreps - 2 (nb/bb) or 3 (nb/bb/wvlt)> <clp path>")
        sys.exit()
    else:
        nreps=int(sys.argv[2])
    
    if sys.argv[3][0] !='/':
        sys.argv[3] = '/'+sys.argv[3]
    if sys.argv[3][-1] !='/':
        sys.argv[3] = sys.argv[3]+'/'
    spk_path=PATH+sys.argv[3]
        
    save_path=PATH+"/clpSpanish/classResults/svm/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    reps=REPS[:nreps]   
    num_utters=len(UTTERS)
           
    #get compressed data, n_components, and file_name list 
    xAlls,clpNames,hcNames,name_count=featAgg(model,reps,spk_path)
    spks=clpNames+hcNames
    num_spks=len(spks)
    num_clps=len(clpNames)
    num_hcs=len(hcNames)
    
    #get num preceding utterances (due to each spk not performing each utter).
    prevsAll={nm:0 for nm in spks}
    for nItr,name in enumerate(spks):
        if nItr==0:
            prevsAll[name]=0
        else:
            for prev in spks[:nItr]:
                prevsAll[name]+=name_count[prev]
                
    #split data into training and test with multiple iterations
    num_clpHc_tests=config['svm']['tst_spks']#must be even (same # of test clps and hcs per iter)
    nv=config['svm']['val_size']#number of validation speakers per split    
    total_itrs=config['svm']['iterations']
    in_iters=config['svm']['in_iters']
    results=pd.DataFrame({'Data':{'train_acc':0,'test_acc':0, 'bin_class':{itr:{} for itr in range(total_itrs)},'class_report':{itr:{} for itr in range(total_itrs)}}})
    
    for o_itr in range(total_itrs):
        clp_files=clpNames
        hc_files=hcNames
        num_clpHc_tests=config['svm']['tst_spks']
        
        for itr in range(in_iters):
            #get test ids and data array index
            clpCurrs=[clp_files[idx] for idx in random.sample(range(0,len(clp_files)),int(num_clpHc_tests/2))]
            hcCurrs=[hc_files[idx] for idx in random.sample(range(0,len(hc_files)),int(num_clpHc_tests/2))]
            spkCurrs=clpCurrs+hcCurrs
            clp_files=[clp for clp in clp_files if clp not in clpCurrs]
            hc_files=[hc for hc in hc_files if hc not in hcCurrs]
            
            prevs={nm:0 for nm in spkCurrs}
            ntst_clp=0
            ntst_hc=0
            for nItr,name in enumerate(spkCurrs):
                if nItr==0:
                    prevs[name]=0
                    ntst_clp+=name_count[name]
                else:
                    if 'CLP' in name:
                        ntst_clp+=name_count[name]
                    elif 'HC' in name:
                        ntst_hc+=name_count[name]
                    for prev in spkCurrs[:nItr]:
                        prevs[name]+=name_count[prev]
                
            #get trainee ids and data array index
            clpTrainees=[spk for spk in clpNames if spk not in clpCurrs]
            hcTrainees=[spk for spk in hcNames if spk not in hcCurrs]
            rand_range=np.arange(len(hcTrainees))
            random.shuffle(rand_range)
            trainees=list(np.array(clpTrainees)[rand_range])+hcTrainees
            
            trPrevs={nm:0 for nm in trainees}
            ntr_clp=0
            ntr_hc=0
            for nItr,name in enumerate(trainees):
                if nItr==0:
                    trPrevs[name]=0
                    ntr_clp+=name_count[name]
                else:
                    if 'CLP' in name:
                        ntr_clp+=name_count[name]
                    elif 'HC' in name:
                        ntr_hc+=name_count[name]
                    for prev in trainees[:nItr]:
                        trPrevs[name]+=name_count[prev]        
            
            ntst=ntst_clp+ntst_hc
            ntr=ntr_clp+ntr_hc
            diffs=np.zeros((nreps,ntr))
            tst_diffs=np.zeros((nreps,ntst))
            
            for nrep,xAll in enumerate(xAlls):
                xTest=np.zeros((ntst,xAll.shape[1]))
                yTest=np.concatenate((np.ones(ntst_clp), np.zeros(ntst_hc)))
                xTrain=np.zeros((ntr,xAll.shape[1]))
                yTrain=np.concatenate((np.ones((ntr_clp)), np.zeros((ntr_hc))))
                for ii,tstName in enumerate(spkCurrs):
                    if ii==len(spkCurrs)-1:
                        xTest[prevs[tstName]:]=xAll[prevsAll[tstName]:prevsAll[tstName]+name_count[tstName]]
                    else:
                        xTest[prevs[tstName]:prevs[spkCurrs[ii+1]]]=xAll[prevsAll[tstName]:prevsAll[tstName]+name_count[tstName]]
                for ii,trName in enumerate(trainees):
                    if ii==len(trainees)-1:
                        xTrain[trPrevs[trName]:]=xAll[prevsAll[trName]:prevsAll[trName]+name_count[trName]]
                    else:
                        xTrain[trPrevs[trName]:trPrevs[trainees[ii+1]]]=xAll[prevsAll[trName]:prevsAll[trName]+name_count[trName]]
                
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
                                
                param_grid = [
                  {'C':np.logspace(0,5,25), 'gamma':np.logspace(-8,-4,25), 'degree':[1],'kernel': ['rbf']},
                    ]

                cv = StratifiedShuffleSplit(n_splits=4, test_size=nv, random_state=42)

                grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv)
                grid.fit(xTrain, yTrain)
                grid=svm.SVC(C=grid.best_params_['C'],degree=grid.best_params_['degree'],gamma=grid.best_params_['gamma'],
                                    kernel=grid.best_params_['kernel'], probability=True)
                grid.fit(xTrain,yTrain)
                
                tr_bin_class=grid.predict_proba(xTrain)
                diffs[nrep,:]=tr_bin_class[:,0]-tr_bin_class[:,1]
                tst_bin_class=grid.predict_proba(xTest)
                tst_diffs[nrep,:]=tst_bin_class[:,0]-tst_bin_class[:,1]
            
            
            diff_tpls=tuple((x1,x2) for x1,x2 in zip(diffs[0,:],diffs[1,:]))
            clf = SGDClassifier(loss="hinge", penalty="l2")
            clf.fit(diff_tpls, yTrain)
            tr_acc=clf.score(diff_tpls,yTrain)
            calibrator=CalibratedClassifierCV(clf, cv='prefit')
            modCal=calibrator.fit(diff_tpls, yTrain)

            tst_diff_tpls=tuple((x1,x2) for x1,x2 in zip(tst_diffs[0,:],tst_diffs[1,:]))
            tst_acc=clf.score(diff_tpls,yTrain)
            bin_class=modCal.predict_proba(tst_diff_tpls)
            
            class_report=classification_report(yTest,grid.predict(xTest))
            results['Data']['train_acc']+=tr_acc*(1/(int(num_spks/num_clpHc_tests)*total_itrs))
            results['Data']['test_acc']+=tst_acc*(1/(int(num_spks/num_clpHc_tests)*total_itrs))
            results['Data']['class_report'][o_itr][itr]=class_report  
            for cpi,(clpName,hcName) in enumerate(zip(clpCurrs,hcCurrs)):   
                if cpi == len(clpCurrs)-1:
                    results['Data']['bin_class'][o_itr][clpName]=bin_class[prevs[clpName]:prevs[hcCurrs[0]]]     
                    results['Data']['bin_class'][o_itr][hcName]=bin_class[prevs[hcName]:]
                else:
                    results['Data']['bin_class'][o_itr][clpName]=bin_class[prevs[clpName]:prevs[clpCurrs[cpi+1]]]     
                    results['Data']['bin_class'][o_itr][hcName]=bin_class[prevs[hcName]:prevs[hcCurrs[cpi+1]]]
        
            if len(clp_files)<num_clpHc_tests//2:
                num_clpHc_tests=len(clp_files)*2

            if not hc_files:
                hc_files=hcNames
            if len(hc_files)<num_clpHc_tests//2:
                left_ids=[lid for lid in np.arange(len(hcNames)) if hcNames[lid] not in hc_files]
                add_ids=random.sample(left_ids, (num_clpHc_tests//2)-len(hc_files))
                hc_files.extend(np.array(hcNames)[add_ids])
    
        if 'wvlt' in reps:
            results.to_pickle(save_path+model+"_wvlt_lateFusionResults.pkl",protocol=4)
        else:
            results.to_pickle(save_path+model+"_lateFusionResults.pkl",protocol=4)



