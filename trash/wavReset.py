import os
import sys
import toolbox.traintestsplit as tts


if  __name__=="__main__":
    PATH=os.path.dirname(os.path.abspath("__main__"))+'/'
    if len(sys.argv)!=2:
        print("python TrainCAE.py <speech path>")
        sys.exit()
    
    if sys.argv[1][0] !='/':
        sys.argv[1] = '/'+sys.argv[1]
    if sys.argv[1][-1] !='/':
        sys.argv[1] = sys.argv[1]+'/'
        
    if os.path.exists(PATH+sys.argv[1]):
        split=tts.trainTestSplit(PATH+sys.argv[1])
        split.wavReset()
    else:
        print("path invalid...")
        sys.exit()
