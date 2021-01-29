import os
import sys
import numpy as np
import pandas as pd
import pickle
import random
import pdb
# import xgboost as xgb
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


MODELS=["CAE","RAE","ALL"]
REPS=['spec','wvlt']    
UNITS=256
UTTERS=['pataka','kakaka','pakata','papapa','petaka','tatata']
# UTTERS=['pataka']
PATH=os.path.dirname(os.path.abspath(__file__))

def saveFeats(model,units,rep,wav_path,utter,save_path, spk_typ):
    global UNITS    
    # load the pretrained model with 256 units and get temp./freq. rep (spec or wvlt)
    aespeech=AEspeech(model=model,units=UNITS,rep=rep) 
    
    #compute the bottleneck and error-based features from a directory with wav files inside 
    #(dynamic: one feture vector for each 500 ms (mel-freq) or 50 ms (wvlt) frame)
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
       
    
# define scoring function 
# def custom_auc(ground_truth, predictions):
#     fpr, tpr, _ = roc_curve(ground_truth, predictions, pos_label=1)    
#     return auc(fpr, tpr)
    
    
if __name__=="__main__":

    if len(sys.argv)!=4:
        print("python dnnTrain.py <'CAE','RAE', or 'ALL'> <'spec' or 'wvlt'> <pd path>")
        sys.exit()        
    #TRAIN_PATH: './pdSpanish/speech/'    
    
    if sys.argv[1] in MODELS:
        model=sys.argv[1]
    else:
        print("python pdsvmTrain.py <'CAE','RAE', or 'ALL'> <'spec' or 'wvlt'> <pd path>")
        sys.exit()
    
    if sys.argv[2] in REPS:
        rep=sys.argv[2]
    else:
        print("python pdsvmTrain.py <'CAE','RAE', or 'ALL'> <'spec' or 'wvlt'> <pd path>")
        sys.exit()    
          
    if sys.argv[3][0] !='/':
        sys.argv[3] = '/'+sys.argv[3]
    if sys.argv[3][-1] !='/':
        sys.argv[3] = sys.argv[3]+'/'
        
    save_path=PATH+"/pdSpanish/classResults/svm/params/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    mfda_path=PATH+"/pdSpanish/"
    mfdas=pd.read_csv(mfda_path+"metadata-Spanish_All.csv")['M-FDA'].values
    pd_mfdas=mfdas[0:50]
    hc_mfdas=mfdas[50:]

    if rep=='wvlt':
        num_feats=64+256
    else:
        num_feats=128+256
    comp_range=np.arange(1,5)
#     num_feats=256
    
    pc_var_info=pd.DataFrame({utter:{'pc_var':[]} for utter in UTTERS})
    scores=[]
    for uIdx,utter in enumerate(UTTERS):
        
        curr_best=0
        pd_path=PATH+sys.argv[3]+'/'+utter+"/pd/"
        hc_path=PATH+sys.argv[3]+'/'+utter+"/hc/"   
        pds=[name for name in os.listdir(pd_path) if '.wav' in name]
        hcs=[name for name in os.listdir(hc_path) if '.wav' in name]
        pds.sort()
        hcs.sort()
        spks=pds+hcs
        num_pd=len(pds)
        num_hc=len(hcs)
        pdFeats=getFeats(model,UNITS,rep,pd_path,utter,'pd')
        hcFeats=getFeats(model,UNITS,rep,hc_path,utter,'hc')
        
        pdAll=np.unique(pdFeats['wav_file'])
        hcAll=np.unique(hcFeats['wav_file'])
        pdIds=np.arange(50)
        hcIds=np.arange(50,100)
        pds=np.zeros((len(pdAll),num_feats,4))
        hcs=np.zeros((len(hcAll),num_feats,4))
        #getting bottle neck features and reconstruction error for training
        for ii,tr in enumerate(pdAll):
            tritr=pdIds[ii]
            pdTrBns=pdFeats['bottleneck'][np.where(pdFeats['wav_file']==spks[tritr])]
            pdTrErrs=pdFeats['error'][np.where(pdFeats['wav_file']==spks[tritr])]
            pdTr=np.concatenate((pdTrBns,pdTrErrs),axis=1)
            pds[ii,:,:]=np.array([np.mean(pdTr,axis=0),np.std(pdTr,axis=0),skew(pdTr,axis=0),kurtosis(pdTr,axis=0)]).T
        for ii,tr in enumerate(hcAll):
            tritr=hcIds[ii]
            hcTrBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==spks[tritr])]
            hcTrErrs=hcFeats['error'][np.where(hcFeats['wav_file']==spks[tritr])]
            hcTr=np.concatenate((hcTrBns,hcTrErrs),axis=1)
            hcs[ii,:,:]=np.array([np.mean(hcTr,axis=0),np.std(hcTr,axis=0),skew(hcTr,axis=0),kurtosis(hcTr,axis=0)]).T

        pdXAll=np.reshape(pds,(pds.shape[0],num_feats*4))
        hcXAll=np.reshape(hcs,(hcs.shape[0],num_feats*4))  
        xAll=np.concatenate((pdXAll,hcXAll),axis=0)
        
        #standardize data and apply PCA. Choose nc based on # of corresponding eigenvectors with cum. variance>90%
        st_xAll=StandardScaler().fit_transform(pd.DataFrame(xAll))
        pca = PCA(n_components=min(xAll.shape[0],xAll.shape[1]))
        pca.fit_transform(st_xAll)
        variance = pca.explained_variance_ratio_ #calculate variance ratios
        var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
        ncs=np.count_nonzero(var>90)
        pca = PCA(n_components=ncs)
        pca_xAll=pca.fit_transform(st_xAll)
        
        #storing pca variance data
        pc_var_info[utter][0]=var
        pc_var_info.to_pickle(save_path+model+'_'+utter+'_'+rep+'_pc.pkl')
        
        #labels for supervised training
        pdYAll=np.ones((pdXAll.shape[0])).T
        hcYAll=np.zeros((hcXAll.shape[0])).T
        yAll=np.concatenate((pdYAll,hcYAll),axis=0)
        
        
        param_grid = [
          {'C':np.logspace(0,5,25), 'gamma':np.logspace(-8,-4,25), 'degree':[1],'kernel': ['rbf','poly']},
        ]
        
        cv = StratifiedShuffleSplit(n_splits=10, test_size=0.15, random_state=42)
#         grid = GridSearchCV(SVC(),scoring = my_auc, param_grid=param_grid, cv=cv)
        grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv)
        grid.fit(pca_xAll, yAll)
        
    #         pipeline = Pipeline(
#                 [("transformer", TruncatedSVD(n_components=70)),
#                 ("classifier", xgb.XGBClassifier(scale_pos_weight=1.0, learning_rate=0.1, 
#                                 max_depth=5, n_estimators=50, min_child_weight=5))])
#         parameters_grid = {'transformer__n_components': [60, 40, 20] }
#         grid_cv = GridSearchCV(pipeline, parameters_grid, scoring = my_auc, n_jobs=-1,
#                                                         cv = StratifiedShuffleSplit(n_splits=5,test_size=0.3,random_state = 0))
#         grid_cv.fit(pca_xTrain, yTrain)

        
        filename = save_path+model+'_'+utter+'_'+rep+'Grid.pkl'
        with open(filename, 'wb') as file:
            joblib.dump(grid, filename)
                
    
    


