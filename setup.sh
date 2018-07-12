#!/bin/bash

# If you are running on a fresh computer, please run this setup files to install all the python packages

echo "WARNING: This script assumes you already have pip and java JDK installed"
echo "If not, please execute the following command:"
echo "sudo apt-get update && sudo apt-get install python-pip openjdk-8-jdk"

pip install numpy==1.14.5 --user
pip install scipy==1.1.0 --user
pip install scikit-learn==0.19.1 --user
pip install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl --user
pip install torchvision==0.2.1 --user
pip install nltk==3.3.0 --user
pip install pydot==1.2.4 --user
pip install h5py==2.8.0 --user
pip install matplotlib==2.2.2 --user
python -m nltk.downloader 'punkt'
