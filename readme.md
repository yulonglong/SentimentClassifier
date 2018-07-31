Sentiment Classification of Movie Reviews
===================================
Neural Network model to classify whether a movie review is positive or negative. Movie reviews are written in English and obtained from IMDB.

Please contact me at sky@u.nus.edu if you have any questions.

The repository is publicly available at https://github.com/yulonglong/SentimentClassifier

Stable release with pdf report is in https://github.com/yulonglong/SentimentClassifier/releases/tag/v1.4

This repository has been updated with more recent version of PyTorch and other libraries, please visit tag v1.4 to replicate results reported in the pdf documentation.

**TLDR:**  
- Assuming you are on Ubuntu 16.04 with Python 3.5, open terminal  
- `./setup.sh` (Install packages) or `./setup_global.sh` (To install packages globally with sudo access)
- `cd data && ./run.sh && cd ..` (Download and Prepare dataset)
- `./run_train.sh 0 GTX1070` (Train model on GPU 0, a GTX1070)
- `./run_test` (Test model on CPU)

**Requirements and Environment:**  
- Ubuntu 16.04  
- Python 3.5.2  

**Python Library Required (requirements.txt):**  
- torch (PyTorch)  
- h5py  
- numpy  
- scikit-learn  
- scipy  
- nltk  
- pydot  
- matplotlib  

Python libraries above can be installed via `pip -r requirements.txt`.

**Python libraries setup:**

If you are running a UNIX based machine, you can run the shell script `./setup.sh` to install all the required python libraries, assuming `pip` is already installed. If you prefer to install manually, please refer to the list of libraries above.

**Dataset:**  
- IMDB Large Moview Review dataset is obtained from `http://ai.stanford.edu/~amaas/data/sentiment/`
- Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. _The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)_.  
- If you are setting up for the first time:  
	- Download and process dataset from scratch by running `run.sh` script in `data` folder  
	- i.e., execute the following command `cd data && ./run.sh`  
	- The data will be automatically downloaded and preprocessed into training, validation, and test set
	- If you do not have java development kit (JDK) installed, please install by running this command:
		- `sudo apt-get update && sudo apt-get install openjdk-8-jdk`

**To train the model**
- Execute `./run_train.sh <GPU_Number> <GPU_Name>` , e.g., `$> ./run_train.sh 0 TITANX`  
- Please make sure Nvidia CUDA is installed to be able to train the model using GPU.  
- For more details on the training arguments, refer to the sample `run_train.sh` shell script  

**Pre-trained word embeddings**
- To train your own word embeddings from the provided unsupervised movie reviews, execute `./run_word2vec`  
- To download pre-trained word embeddings such as GloVe, execute `cd word2vec/vectors && ./run.sh`

**To test the model**
- For quick testing, execute `./run_test.sh`  
- There are three arguments for testing:
	- `-v`  : vocab path, the path to the vocabulary files saved during training
	- `-m`  : model path, the path to the best model saved during training
	- `-ts` : test path, the path to the text file containing the movie review to be evaluated
- Sample command : `python test.py -v saved_model/vocab_v50000.pkl -m saved_model/best_model_weights.h5 -ts data/aclImdb/train/unsup/74_0.txt`
