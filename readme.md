Sentiment Classification of Movie Reviews
===================================
Neural Network model to classify whether a movie review is positive or negative. Movie reviews are written in English and obtained from IMDB.

The repository is publicly available at https://github.com/yulonglong/SentimentClassifier

**Requirements and Environment:**  
- Ubuntu 16.04  
- Python 2.7.12  
- GCC 5.4.0  

**Python Library Required:**  
- torch 0.2.0.post3
- h5py 2.7.1  
- numpy 1.13.3  
- scipy 0.19.1  
- nltk 3.2.1

**Dataset:**  
- IMDB Large Moview Review dataset is obtained from `http://ai.stanford.edu/~amaas/data/sentiment/`
- Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. _The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)_.  
- If you are setting up for the first time:  
	- Download and process dataset from scratch by running `run.sh` script in `data` folder  
	- i.e., execute the following command `cd data && ./run.sh`  
	- The data will be automatically downloaded and preprocessed into training, validation, and test set

**To train the model**
- Execute `./run_train.sh`  
- For more details on the training arguments, refer to the sample `run_train.sh` shell script  

**Pre-trained word embeddings**
- To train your own word embeddings from the provided unsupervised movie reviews, execute `./run_word2vec`  
- To download pre-trained word embeddings such as GloVe, execute `cd word2vec/vectors && ./run.sh`
