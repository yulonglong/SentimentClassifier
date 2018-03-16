#!/bin/bash

# If you are running on a fresh computer, please run this setup files to install all the python packages

echo "WARNING: This script assumes you already have pip and java JDK installed"
echo "If not, please execute the following command:"
echo "sudo apt-get update && sudo apt-get install python-pip openjdk-8-jdk"

pip install numpy==1.13.3 --user
pip install scipy==0.19.1 --user
pip install scikit-learn==0.19.1 --user
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl --user
pip install torchvision --user
pip install nltk==3.2.2 --user
pip install pydot==1.1.0 --user
pip install h5py==2.6.0 --user
pip install matplotlib==1.5.3 --user
python -m nltk.downloader 'punkt'
