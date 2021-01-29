#!/bin/bash
#SBATCH --job-name=svmEval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o /cluster/%u/cl_out/%x-%j-ono-%N.out
#SBATCH -e /cluster/%u/cl_out/%x-%j-on-%N.err
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00

export WORKON_HOME=/cluster/ag61iwyb/.python_cache
#export XDG_CACHE_DIR=/cluster/ag61iwyb/.cache
#export PYTHONUSERBASE=/cluster/ag61iwyb/.python_packages
#export PATH=/opt/miniconda/bin:$PATH

pip3 install --user -r /cluster/ag61iwyb/requirements.txt

#python -c "import torch;print(torch.__version__)"
#python -c "import librosa;print(librosa.__version__)"

python3 clpSvmEval.py CAE wvlt clpSpanish/speech
python3 clpSvmEval.py CAE narrowband clpSpanish/speech
python3 clpSvmEval.py CAE broadband clpSpanish/speech
python3 clpSvmEval.py CAE mc_fuse clpSpanish/speech
python3 clpSvmEval.py CAE early_fuse2 clpSpanish/speech
python3 clpSvmEval.py CAE early_fuse3 clpSpanish/speech
