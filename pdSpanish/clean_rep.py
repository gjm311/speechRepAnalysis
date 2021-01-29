import sys
import os
import shutil

UTTERS=['pataka','kakaka','pakata','papapa','petaka','tatata']

def clean_frames(rep):
    data_path=PATH+'/speech/'
    dirNames=os.listdir(data_path)
    
    for dirn in dirNames:
        if dirn in UTTERS:
            frame_path=data_path+'/'+dirn+'/frames/'
            if rep=='all':
                if os.path.isdir(frame_path):
                    shutil.rmtree(frame_path)
            else: 
                if os.path.isdir(frame_path+'/'+rep+'/'):
                    shutil.rmtree(frame_path+'/'+rep+'/')

            
def clean_feats(rep):
    data_path=PATH+'/feats/'
    if rep=='all':
        if os.path.isdir(data_path):
            shutil.rmtree(data_path)
    else:
        dirNames=os.listdir(data_path)
        for dirn in dirNames:
            if dirn in UTTERS:
                feat_path=data_path+'/'+dirn+'/'
                featNames=os.listdir(feat_path)
                for feat in featNames:
                    if rep=='wvlt' and 'wvlt' in feat:
                        os.remove(feat_path+'/'+feat)
                    if rep=='narrowband' and 'narrowband' in feat:
                        os.remove(feat_path+'/'+feat)
                    if rep=='broadband' and 'broadband' in feat:
                        os.remove(feat_path+'/'+feat)
                    if rep=='spec':
                        if 'narrowband' in feat:
                            os.remove(feat_path+'/'+feat)
                        if 'broadband' in feat:
                            os.remove(feat_path+'/'+feat)

if __name__ == "__main__":
    PATH=os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv)!=3:
        print("python clean_rep.py <broadband, narrowband, spec, wvlt, all> <frames, feats, all>")
        sys.exit()   
        
    if sys.argv[1] not in ['broadband', 'narrowband','spec','wvlt','all']:
        print("python clean_rep.py <broadband, narrowband, spec, wvlt, all> <frames, feats, all>")
    else:
        rep=sys.argv[1]
        
    if sys.argv[2] not in ['frames', 'feats', 'all']:
        print("python clean_rep.py <broadband, narrowband, spec, wvlt, all> <frames, feats, all>")
        sys.exit() 
    if sys.argv[2]=='frames':
        if rep in ['narrowband','broadband']:
            print("nb/bb not options for 'frames' only <spec or wvlt>")
            sys.exit()
        clean_frames(rep)
        
    elif sys.argv[2]=='feats':
        clean_feats(rep)
    else:
        clean_frames(rep)
        clean_feats(rep)
            
    


                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
