This folder is to contain dataset to be used in this Sentiment Classification Project
Run the shell script in current directory `run.sh`
or to do it manually, please refer to the steps below

Steps to download and process the data for training/testing:
1. Download the imdb dataset from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
2. Place the tar.gz in this current directory
3. Extract it here (i.e., we will get the following `~/SentimentClassifier/data/aclImdb/`)
4. Compile `Preprocess.java` (i.e., `javac Preprocess.java`)
5. Execute Preprocess (i.e., `java Preprocess`)

## Please note that the splitting of the training and validation dataset is in 80% to 20% ratio.
## In this case, it is 20,000 training and 5,000 validation, Feel free to change the ratio in the `Preprocess.java`, `f_percentageValid`
## And also note that the sampling is done randomly, so you might get different split set for each time the Preprocess is run
