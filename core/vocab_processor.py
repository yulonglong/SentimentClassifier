
import pickle as pk
import codecs
import glob
import multiprocessing
import logging
from core import text_cleaner as text_cleaner

logger = logging.getLogger(__name__)


def load_vocab(vocab_path):
    """
    Load vocabulary with the given path
    """
    logger.info('Loading vocabulary from: ' + vocab_path)
    with open(vocab_path, 'rb') as vocab_file:
        vocab = pk.load(vocab_file)
    return vocab


def create_vocab_file_list(file_list, tokenize_text, to_lower):
    """
    Create vocabulary from a specified file_list
    """

    total_words = 0
    word_freqs = {}

    # Reading from file list
    for file_path in file_list:
        with codecs.open(file_path, mode='r', encoding='ISO-8859-1') as input_file:
            # logger.info(input_file)
            for line in input_file:
                splitBrLine = line.replace("<br />", "\n").replace("<br/>", "\n").replace("<br>", "\n").split("\n")
                for subline in splitBrLine:
                    content = subline
                    if to_lower:
                        content = content.lower()
                    if tokenize_text:
                        content = text_cleaner.tokenize(content)
                    else:
                        content = content.split()
                    for word in content:
                        try:
                            word_freqs[word] += 1
                        except KeyError:
                            word_freqs[word] = 1
                        total_words += 1
           
    # Pop the <num> string because going to be added later
    if '<num>' in  word_freqs: word_freqs.pop('<num>')
    return word_freqs, total_words


class CreateVocabThread (multiprocessing.Process):
    """
    A class to multithread vocab creation
    """
    def __init__(self, threadID, file_list, tokenize_text, to_lower):
        """
        Constructor to initialize necessary information to start reading dataset
        """
        multiprocessing.Process.__init__(self)

        self.threadID = threadID
        self.file_list = file_list
        self.tokenize_text = tokenize_text
        self.to_lower = to_lower

        self.result_queue = multiprocessing.Queue()

    def run(self):
        """
        Main multi-threaded function to run in this class
        """
        logger.info('Thread ' + str(self.threadID) + ' : Reading dataset for vocab')
        word_freqs, total_words = create_vocab_file_list(self.file_list, self.tokenize_text, self.to_lower)
        
        self.result_queue.put(word_freqs)
        self.result_queue.put(total_words)
        return

    def get_dataset(self):
        """
        Getter function
        Obtain the results after the thread is finished running.
        """
        word_freqs = self.result_queue.get()
        total_words = self.result_queue.get()
        logger.info('Thread ' + str(self.threadID) + ' : Vocab dataset is retrieved from queue successfully')
        return word_freqs, total_words


###############################################################################
## MultiThread class for Creating Vocab
#

class CreateVocab(object):
    """
    A class to multithread vocab creation
    """
    def __init__(self, dir_path, vocab_size, tokenize_text, to_lower):
        """
        Constructor to initialize necessary information to start reading dataset
        """
        
        # Read dataset arguments
        self.dir_path = dir_path
        self.vocab_size = vocab_size
        self.tokenize_text = tokenize_text
        self.to_lower = to_lower
        self.file_list_collection = []
        # Leaving 2 free threads for other purposes
        self.num_cpu = multiprocessing.cpu_count() - 2
        if self.num_cpu <= 0:
            self.num_cpu = 1

        file_list_full = []
        # Reading data in the specified folder

        # Assuming there are two subfolders, pos, and neg in the directory
        dir_path_pos = glob.glob(dir_path + "pos/*")
        dir_path_neg = glob.glob(dir_path + "neg/*")
        # Traverse every file in the directory
        for file_path in dir_path_pos:
            file_list_full.append(file_path) # Keep track of the filename
        for file_path in dir_path_neg:
            file_list_full.append(file_path) # Keep track of the filename
        
        batch_size = len(file_list_full) // (self.num_cpu)
        if (len(file_list_full) % self.num_cpu > 0): batch_size += 1
        
        self.file_list_collection = [file_list_full[i:i+batch_size] for i in range(0, len(file_list_full), batch_size)]

    def read_dataset_multithread(self):
        """
        The only function to call in this class other than initializing the class
        """
        threadCollection = [None] * len(self.file_list_collection)
        for threadNum in range(len(self.file_list_collection)):
            threadCollection[threadNum] = CreateVocabThread(threadNum, self.file_list_collection[threadNum], self.tokenize_text, self.to_lower)
            threadCollection[threadNum].start()

        word_freqs = {}
        total_words, unique_words = 0, 0

        for threadNum in range(len(self.file_list_collection)):
            curr_word_freqs, curr_total_words = threadCollection[threadNum].get_dataset()
            total_words += curr_total_words
            # logger.warning(threadNum)
            # logger.warning(curr_word_freqs)
            for word, freq in curr_word_freqs.items():
                # logger.error(word)
                if word in word_freqs:
                    word_freqs[word] = word_freqs[word] + freq
                else:
                    word_freqs[word] = freq
                    unique_words += 1
                
        assert(unique_words == len(word_freqs))
        logger.info('  %i total words, %i unique words' % (total_words, unique_words))
        import operator
        sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

        # Print vocabulary for debugging purposes
        # with open('vocab_' + str(len(sorted_word_freqs)) + '.tsv', 'w') as outfile:
        #     for item in sorted_word_freqs:
        #         outfile.write(str(item[1]) + '\t' + item[0].encode('utf-8') + '\n')

        if self.vocab_size <= 0:
            # Choose vocab size automatically by removing all singletons
            self.vocab_size = 0
            for word, freq in sorted_word_freqs:
                if freq > 1: self.vocab_size += 1

        # Initialize the index/vocabulary of 4 most basic tokens
        vocab = {'<pad>':0, '<unk>':1, '<num>':2, '<newline>':3}
        vcb_len = len(vocab)
        index = vcb_len
        # Build vocabulary with its individual index
        for word, _ in sorted_word_freqs[:self.vocab_size - vcb_len]:
            vocab[word] = index
            index += 1
        
        return vocab

