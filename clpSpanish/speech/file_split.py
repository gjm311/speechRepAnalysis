import sys
import os
import shutil



if __name__ == "__main__":
    PATH=os.path.dirname(os.path.abspath(__file__))
    dirNames=os.listdir(PATH)
    
    for dirn in dirNames:
        if '.' not in dirn and os.path.isdir(dirn):
            curr_path=PATH+'/'+dirn+'/'
            if not os.path.exists(curr_path+'clp/'):
                os.makedirs(curr_path+'clp/')
                os.makedirs(curr_path+'hc/')
            for file in os.listdir(curr_path):
                if 'CLP' in file:
                    shutil.move(curr_path+file,curr_path+'clp/'+file)
                elif 'HC' in file:
                    shutil.move(curr_path+file,curr_path+'hc/'+file)