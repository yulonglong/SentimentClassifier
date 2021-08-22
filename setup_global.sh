#!/bin/bash

# If you are running on a fresh computer, please run this setup files to install all the python packages
echo "WARNING: This script assumes you already have pip and java JDK installed"
echo "If not, please execute the following command:"
echo "sudo apt update && sudo apt install python3-pip default-jdk"


sudo pip3 install -r requirements.txt
python3 -m nltk.downloader 'punkt'
