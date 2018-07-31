#!/bin/bash

# If you are running on a fresh computer, please run this setup files to install all the python packages
echo "WARNING: This script assumes you already have pip and java JDK installed"
echo "If not, please execute the following command:"
echo "sudo apt-get update && sudo apt-get install python-pip openjdk-8-jdk"


sudo pip3 install -r requirements.txt
python3 -m nltk.downloader 'punkt'
