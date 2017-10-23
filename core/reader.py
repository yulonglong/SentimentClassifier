import random
import codecs
import sys
import nltk
import logging
import glob
import numpy as np
import pickle as pk
import re # regex
import copy
import os
import math

# for multithreading
import multiprocessing
import time

logger = logging.getLogger(__name__)

###############################################################################
## MultiThread class for reading datasets
#

class ReadDatasetThread (multiprocessing.Process):
    """
    A class to multithread read_dataset method (train, dev, and test)
    Please take note of the multiprocessing implementation as join() was not used
    """
    def __init__(self, threadID, threadName, dir_path, maxlen, vocab, tokenize_text, to_lower):
        """
        Constructor to initialize necessary information to start reading dataset
        """
        multiprocessing.Process.__init__(self)

        self.threadID = threadID
        self.threadName = threadName
        
        # Read dataset arguments
        self.dir_path = dir_path
        self.maxlen = maxlen
        self.vocab = vocab
        self.tokenize_text = tokenize_text
        self.to_lower = to_lower

        # Result of read_dataset to be kept
        self.x = []
        self.y = []
        self.filename_y = []
        self.overallMaxlen = 0

        self.result_queue = multiprocessing.Queue()

    def run(self):
        """
        Main multi-threaded function to run in this class
        """
        logger.info('Thread ' + str(self.threadID) + ' : ' + self.threadName + ' - Reading dataset')
        x, y, filename_y, overallMaxlen = read_dataset(self.dir_path, self.maxlen, self.vocab, self.tokenize_text, self.to_lower, thread_id = self.threadID)
        
        self.result_queue.put(x)
        self.result_queue.put(y)
        self.result_queue.put(filename_y)
        self.result_queue.put(overallMaxlen)
        return

    def get_dataset(self):
        """
        Getter function
        Obtain the results after the thread is finished running.
        """
        x = self.result_queue.get()
        y = self.result_queue.get()
        filename_y = self.result_queue.get()
        overallMaxlen = self.result_queue.get()
        logger.info('Thread ' + str(self.threadID) + ' : Dataset is retrieved from queue successfully')
        return x, y, filename_y, overallMaxlen

###############################################################################
## END Multithread class for reading datasets
###############################################################################


###############################################################################
## FUNCTIONS for PREPROCESSING / TOKENIZING WORDS
#

def isContainDigit(string):
    return re.search("[0-9]", string)

def isAllDigit(string):
    return re.search("^[0-9]+$", string)

def cleanWord(string):
    if isAllDigit(string):
        return '<num>'
    if len(string) == 1:
        return string
    if isContainDigit(string):
        return '<num>'
    return string

def tokenize(string):
    """
    Tokenize a string (i.e., sentence(s)).
    """
    tokens = nltk.word_tokenize(string)
    index = 0 
    while index < len(tokens):
        tokens[index] = cleanWord(tokens[index])
        index += 1
    return tokens

###############################################################################
## END FUNCTIONS for PREPROCESSING / TOKENIZING WORDS
###############################################################################


def load_vocab(vocab_path):
    """
    Load vocabulary with the given path
    """
    logger.info('Loading vocabulary from: ' + vocab_path)
    with open(vocab_path, 'rb') as vocab_file:
        vocab = pk.load(vocab_file)
    return vocab

def create_vocab(dir_path, vocab_size, tokenize_text, to_lower):
    """
    Create vocabulary from training set with the specified parameters.
    Parameters are vocab_size, tokenize_text, to_lower
    """
    logger.info('Creating vocabulary from: ' + dir_path)

    total_words, unique_words = 0, 0
    word_freqs = {}

    dir_path_list = []
    dir_path_list.append(glob.glob(dir_path + "pos/*")) # path to positive files
    dir_path_list.append(glob.glob(dir_path + "neg/*")) # path to negative files

    # Reading both positive and negative data
    for i in range(len(dir_path_list)):
        len_curr = len(dir_path_list[i])
        counter_curr = 0
        for file_path in dir_path_list[i]:
            with codecs.open(file_path, mode='r', encoding='ISO-8859-1') as input_file:
                # logger.info(input_file)
                for line in input_file:
                    content = line
                    if to_lower:
                        content = content.lower()
                    if tokenize_text:
                        content = tokenize(content)
                    else:
                        content = content.split()
                    for word in content:
                        try:
                            word_freqs[word] += 1
                        except KeyError:
                            unique_words += 1
                            word_freqs[word] = 1
                        total_words += 1
            counter_curr += 1
            if counter_curr % 100 == 0:
                logger.info(str(i) + ") Completed " + str(counter_curr) + " out of " + str(len_curr))

    # Pop the <num> string because going to be added later
    if '<num>' in  word_freqs: word_freqs.pop('<num>')

    logger.info('  %i total words, %i unique words' % (total_words, unique_words))
    import operator
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    # # Print vocabulary for debugging purposes
    # with open('data_binary_offline_test/vocab_' + str(len(sorted_word_freqs)) + '.tsv', 'w') as outfile:
    #     for item in sorted_word_freqs:
    #         outfile.write(str(item[1]) + '\t' + item[0].encode('utf-8') + '\n')

    if vocab_size <= 0:
        # Choose vocab size automatically by removing all singletons
        vocab_size = 0
        for word, freq in sorted_word_freqs:
            if freq > 1: vocab_size += 1

    # Initialize the index/vocabulary of 4 most basic tokens
    vocab = {'<pad>':0, '<unk>':1, '<num>':2, '<newline>':3}
    vcb_len = len(vocab)
    index = vcb_len
    # Build vocabulary with its individual index
    for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
        vocab[word] = index
        index += 1

    return vocab

def read_dataset_folder(dir_path, maxlen, vocab, tokenize_text, to_lower, thread_id = 0, char_level=False):
    """
    Read dataset from a specified folder. 
    """

    data_x, data_y, filename_y = [], [], []
    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = -1
    total_len = 0
    num_files = 0

    # Reading data in the specified folder
    dir_path_curr = glob.glob(dir_path)
    # Traverse every file in the directory
    for file_path in dir_path_curr:
        ###################################################
        ## BEGIN READ FREE-TEXT
        #
        indices = []
        with codecs.open(file_path, mode='r', encoding='ISO-8859-1') as input_file:
            contains_ct_word = False
            for line in input_file:
                content = line
               
                if to_lower:
                    content = content.lower()
                if tokenize_text:
                    content = tokenize(content)
                else:
                    content = content.split()
                if maxlen > 0 and len(content) > maxlen:
                    continue
                
                for word in content:
                    if word in vocab:
                        indices.append(vocab[word])
                        if (word == '<num>'): num_hit += 1
                    else:
                        indices.append(vocab['<unk>'])
                        unk_hit += 1
                    total += 1
                # if this line is not a blank
                if (len(content) > 0) and ('<newline>' in vocab):
                    indices.append(vocab['<newline>'])

            data_x.append(indices)
            if ("pos/*" in dir_path):
                data_y.append(float(1))
            elif ("neg/*" in dir_path):
                data_y.append(float(0))
            else:
                raise wrongFolderNameError
            filename_y.append(input_file.name) # Keep track of the filename

            if maxlen_x < len(indices):
                maxlen_x = len(indices)
            total_len += len(indices)
            num_files += 1

    logger.info("Average length for this dataset is %.5f" % (total_len / num_files))
    return data_x, data_y, filename_y, maxlen_x, num_hit, unk_hit, total

def read_dataset(dir_path, maxlen, vocab, tokenize_text, to_lower, thread_id = 0, char_level=False):
    """
    Read dataset from a particular directory which contains positive, negative, and lab folders.
    In other words, this function is to read either 'train', 'valid', or 'test' set
    This class will call read_dataset_folder for each 'pos' and 'neg' folder
    """
    t0 = time.time()

    logger.info('Thread ' + str(thread_id) + ' : ' + 'Reading dataset from: ' + dir_path)
    if maxlen > 0:
        logger.info('Thread ' + str(thread_id) + ' : ' + '  Removing sequences with more than ' + str(maxlen) + ' words')

    # Reading positive data
    dir_path_pos = dir_path + "pos/*"
    pos_data_x, pos_data_y, pos_filename_y, pos_maxlen_x, pos_num_hit, pos_unk_hit, pos_total = read_dataset_folder(dir_path_pos, maxlen, vocab, tokenize_text, to_lower, thread_id = thread_id, char_level=False)

    # Reading negative data
    dir_path_neg = dir_path + "neg/*"
    neg_data_x, neg_data_y, neg_filename_y, neg_maxlen_x, neg_num_hit, neg_unk_hit, neg_total = read_dataset_folder(dir_path_neg, maxlen, vocab, tokenize_text, to_lower, thread_id = thread_id, char_level=False)

    # Appending array
    data_x = pos_data_x + neg_data_x
    data_y = pos_data_y + neg_data_y
    filename_y = pos_filename_y + neg_filename_y
    maxlen_x = max(pos_maxlen_x, neg_maxlen_x)

    num_hit = pos_num_hit + neg_num_hit
    unk_hit = pos_unk_hit + neg_unk_hit
    total = pos_total + neg_total

    # Capture time
    time_taken =  time.time() - t0
    time_taken_min = time_taken/60

    logger.info('Thread ' + str(thread_id) + ' : ' + '  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
    logger.info('Thread ' + str(thread_id) + ' : ' + 'Read Dataset time taken = %i sec (%.1f min)' % (time_taken, time_taken_min))
    logger.info('Thread ' + str(thread_id) + ' : ' + 'Number of positive instances         = ' + str(len(pos_data_y)))
    logger.info('Thread ' + str(thread_id) + ' : ' + 'Number of negative instances         = ' + str(len(neg_data_y)))
    logger.info('Thread ' + str(thread_id) + ' : ' + 'Number of instances                  = ' + str(len(data_y)))

    return data_x, data_y, filename_y, maxlen_x


def read_dataset_single(file_path, vocab, tokenize_text, to_lower, char_level=False):
    """
    Method to read a single movie review
    Used for testing/demo
    """
    data_x = []
    indices = []
    with codecs.open(file_path, mode='r', encoding='ISO-8859-1') as input_file:
        for line in input_file:
            content = line     
            if to_lower:
                content = content.lower()
            if tokenize_text:
                content = tokenize(content)
            else:
                content = content.split()

            for word in content:
                if word in vocab:
                    indices.append(vocab[word])
                else:
                    if isAllDigit(word):
                        indices.append(vocab['<num>'])
                    else:
                        indices.append(vocab['<unk>'])
                            # if this line is not a blank
            if (len(content) > 0) and ('<newline>' in vocab):
                indices.append(vocab['<newline>'])
        data_x.append(indices)

    return data_x

def get_data(args, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None):
    """
    Function to read dataset from a global perspective.
    Responsible to read and collate all 3 datasets: train, valid, and test.
    """
    train_path, dev_path, test_path = args.train_path, args.dev_path, args.test_path
    vocab_size = args.vocab_size
    maxlen = 0
    is_multithread = not args.no_multithread
    
    if not vocab_path:
        # If vocab path is not specified, create vocabulary from scratch
        vocab = create_vocab(train_path, vocab_size, tokenize_text, to_lower)
        if len(vocab) < vocab_size:
            logger.warning('The vocabulary includes only %i words (less than %i)' % (len(vocab), vocab_size))
        else:
            assert vocab_size == 0 or len(vocab) == vocab_size
    else:
        # If vocab path is specified, read from the file
        vocab = load_vocab(vocab_path)
        if len(vocab) != vocab_size:
            logger.warning('The vocabulary includes %i words which is different from given: %i' % (len(vocab), vocab_size))
    logger.info('  Vocab size: %i' % (len(vocab)))

    # ===== Begin Concurrent/Parallel multithreading read_dataset =====
    # Please take note of the multiprocessing implementation as join() was not used
    # but it was proven to work fine and double checked the result with no multiprocessing
    if (is_multithread):
        logger.info('Creating 3 threads for train, dev, and test read_dataset')

        # Create new threads
        trainReadDatasetThread = ReadDatasetThread(1, "Train-Thread", train_path, maxlen, vocab, tokenize_text, to_lower)
        devReadDatasetThread = ReadDatasetThread(2, "Dev-Thread", dev_path, 0, vocab, tokenize_text, to_lower)
        testReadDatasetThread = ReadDatasetThread(3, "Test-Thread", test_path, 0, vocab, tokenize_text, to_lower)

        # Start new Threads
        trainReadDatasetThread.start()
        devReadDatasetThread.start()
        testReadDatasetThread.start()

        train_x, train_y, train_filename_y, train_maxlen = trainReadDatasetThread.get_dataset()
        dev_x, dev_y, dev_filename_y, dev_maxlen = devReadDatasetThread.get_dataset()
        test_x, test_y, test_filename_y, test_maxlen = testReadDatasetThread.get_dataset()
        
        logger.info('Successfully synchronized train, dev, and test threads.. Reading dataset complete..')
    # ===== End multithreading read_dataset =====
    else:
        train_x, train_y, train_filename_y, train_maxlen, = read_dataset(train_path, maxlen, vocab, tokenize_text, to_lower)
        dev_x, dev_y, dev_filename_y, dev_maxlen, = read_dataset(dev_path, 0, vocab, tokenize_text, to_lower)
        test_x, test_y, test_filename_y, test_maxlen, = read_dataset(test_path, 0, vocab, tokenize_text, to_lower)


    overal_maxlen = max(train_maxlen, dev_maxlen, test_maxlen)
    
    return ((train_x,train_y,train_filename_y), (dev_x,dev_y,dev_filename_y), (test_x,test_y,test_filename_y), vocab, len(vocab), overal_maxlen)

def load_dataset(args):
    """
    This is the main and the most important method in this file
    This function is called by the main program and call the sub-methods such as read_dataset
    """

    # If data binary path is provided, read from the binary data
    if (args.data_binary_path):
        logger.info("Loading binary processed data...")
        with open(args.data_binary_path, 'rb') as read_processed_data_file:
            ((train_x, train_y, train_filename_y),
            (dev_x, dev_y, dev_filename_y),
            (test_x, test_y, test_filename_y),
            vocab,
            vocab_size,
            overal_maxlen) = pk.load(read_processed_data_file)
        logger.info("Loading binary processed data completed!")
    # else read and process from plain text data provided
    else:
        logger.info("Processing data...")
        # data_x is a list of lists
        ((train_x, train_y, train_filename_y),
        (dev_x, dev_y, dev_filename_y),
        (test_x, test_y, test_filename_y),
        vocab, vocab_size, overal_maxlen) = get_data(args)

        logger.info("Data processing completed!")

        ######################################################################################
        ## if the seed starts with "10", which means first iteration and gpu num 0, then save
        ## This is to prevent unecessary double dumping/saving which will take alot of disk space
        ## Remove the if statement to always dump the files.
        #
        # Dump dev file
        with open(args.out_dir_path + '/data/dev_data_v'+ str(vocab_size) + '.pkl', 'wb') as dev_data_file:
            pk.dump([dev_x, dev_y, dev_filename_y], dev_data_file)

        # Dump test file
        with open(args.out_dir_path + '/data/test_data_v'+ str(vocab_size) + '.pkl', 'wb') as test_data_file:
            pk.dump([test_x, test_y, test_filename_y], test_data_file)

        # Dump processed data
        with open(args.out_dir_path + '/data/processed_data_v'+ str(vocab_size) + '.pkl', 'wb') as processed_data_file:
            pk.dump([
                (train_x, train_y, train_filename_y),
                (dev_x, dev_y, dev_filename_y),
                (test_x, test_y, test_filename_y),
                vocab,
                vocab_size,
                overal_maxlen
            ], processed_data_file)

        # Dump vocab
        with open(args.out_dir_path + '/data/vocab_v'+ str(vocab_size) + '.pkl', 'wb') as vocab_data_file:
            pk.dump([vocab], vocab_data_file)

    with open(args.out_dir_path + '/preds/train_ref_length.txt', 'w') as instance_length_file:
        for s in train_x: instance_length_file.write('%d\n' % len(s))

    with open(args.out_dir_path + '/preds/dev_ref_length.txt', 'w') as instance_length_file:
        for s in dev_x: instance_length_file.write('%d\n' % len(s))

    with open(args.out_dir_path + '/preds/test_ref_length.txt', 'w') as instance_length_file:
        for s in test_x: instance_length_file.write('%d\n' % len(s))

    return ((train_x, train_y, train_filename_y),
        (dev_x, dev_y, dev_filename_y),
        (test_x, test_y, test_filename_y),
        vocab, vocab_size, overal_maxlen)

##############################################################################
## unrelated to Training neural network
## a separate tokenizer for text before training word2vec
#

def tokenize_dataset(file_path, output_path, to_lower=True):
    """
    Simple tokenizer for the text before training word2vec
    """
    output = open(output_path,"w")
    logger.info('Reading dataset from: ' + file_path)

    with codecs.open(file_path, mode='r', encoding='ISO-8859-1') as input_file:
        for line in input_file:
            content = line

            if to_lower:
                content = content.lower()

            content = tokenize(content)

            for word in content:
                if isContainDigit(word):
                    output.write('<num>')
                else:
                    output.write((word).encode("utf8"))
                output.write(' ')

            output.write('\n')
