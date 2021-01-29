#!/bin/bash
#SBATCH --job-name=aeSpeech
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o /home/%u/output/%x-%j-on-%N.out
#SBATCH -e /home/%u/output/%x-%j-on-%N.err
#SBATCH --mail-type=ALL
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=24:00:00
#SBATCH --exclude=lme53
# Tell's pipenv to install the virtualenvs in the cluster folder
export WORKON_HOME==/cluster/`whoami`/.python_cache
echo "Your job is running on" $(hostname)
# Small Python packages can be installed in own home directory. Not recommended for big packages like tensorflow -> Follow instructions for pipenv below
# cluster_requirements.txt is a text file listing the required pip packages (one package per line)
pip3 install --user -r cluster_requirements.txt

python3 pdsvmEvalAgg_paramOpt.py CAE spec "./pdSpanish/speech/"
