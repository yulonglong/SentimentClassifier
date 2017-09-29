import argparse
import logging
import numpy as np
from time import time
import sys
import utils as U
import pickle as pk
import copy

logger = logging.getLogger(__name__)

"""
Helper.py

Contains helper functions needed by other modules
"""

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def sort_data(x, y, filename_y, lab_x=None):
    """Sort data based on the length of x"""

    test_lab_x = lab_x
    test_xy = zip(x, y, filename_y)
    if not (lab_x is None):
        test_xy = zip(x, y, filename_y, lab_x)

    # Sort tuple based on the length of the first entry in the tuple
    test_xy.sort(key=lambda t: len(t[0]))
    if not (lab_x is None):
        test_x, test_y, test_filename_y, test_lab_x = zip(*test_xy)
    else:
        test_x, test_y, test_filename_y = zip(*test_xy)

    return test_x, test_y, test_filename_y, test_lab_x

def sort_data_given_index(x, y, perm_index, lab_x=None):
    """
    Arrange data sequence given permutation index.
    The index was randomly shuffled before hand (usually for training)
    """
    assert len(x) == len(y)
    train_x = [None]* len(x)
    train_y = [None]* len(y)
    train_lab_x = None
    if not (lab_x is None): train_lab_x = [None] * len(lab_x)

    counter = 0
    for idx in perm_index:
        train_x[idx] = x[counter]
        train_y[idx] = y[counter]
        if not (lab_x is None): train_lab_x[idx] = lab_x[counter]
        counter += 1

    return train_x, train_y, train_lab_x

def split_data_into_chunks(x, y, batch_size, combine_y=True, lab_x=None):
    """
    Split data into chunks/batches (with the specified batch size) for mini-batch processing in neural network.
    """
    import keras.backend as K
    from keras.preprocessing import sequence

    test_x_chunks = list(chunks(x, batch_size))
    test_x = []
    test_x_len = 0

    test_y_chunks = list(chunks(y, batch_size))
    test_y = []
    test_y_len = 0

    test_lab_x_chunks = None
    test_lab_x = []
    test_lab_x_len = 0
    if not (lab_x is None): test_lab_x_chunks = list(chunks(lab_x, batch_size))

    assert len(test_x_chunks) == len(test_y_chunks)

    for i in range(len(test_x_chunks)):
        curr_test_x = test_x_chunks[i]
        curr_test_x = sequence.pad_sequences(curr_test_x)
        test_x.append(curr_test_x)
        test_x_len += len(curr_test_x)

        curr_test_y = test_y_chunks[i]
        curr_test_y = np.array(curr_test_y, dtype=K.floatx())
        test_y.append(curr_test_y)
        test_y_len += len(curr_test_y)

        if not (lab_x is None):
            curr_test_lab_x = test_lab_x_chunks[i]
            curr_test_lab_x = np.array(curr_test_lab_x, dtype=K.floatx())
            test_lab_x.append(curr_test_lab_x)
            test_lab_x_len += len(curr_test_lab_x)

    assert test_x_len == test_y_len
    assert test_x_len == len(y)
    if not (lab_x is None):
        assert test_x_len == test_lab_x_len

    if (combine_y):
        test_y = np.array(y, dtype=K.floatx())

    return test_x, test_y, test_lab_x

def sort_and_split_data_into_chunks(x, y, filename_y, batch_size, lab_x=None):
    """
    Sort based on length of x
    Split test data into chunks of N (batch size) and pad them per chunk/batch
    Faster processing because of localized padding
    Usually used for validation and testing where the sequence of the dataset does not matter.
    """
    test_lab_x = None
    test_x, test_y, test_filename_y, test_lab_x = sort_data(x, y, filename_y, lab_x=lab_x)

    test_x, test_y, test_lab_x = split_data_into_chunks(test_x, test_y, batch_size, lab_x=test_lab_x)
    return test_x, test_y, test_filename_y, test_lab_x

def calculate_performance(tps, fps, fns, tns):
    """
    Calculate the performance/evaluation metrics given the confusion matrix (TP, FP, FN, TN)
    The evaluation metrics are:
    Recall/Sensitivity, Precision, Specificity, F1-score, F0.5-score, F1-recall-specificity, F0.5-recall-specificity.
    """
    recall, precision, specificity, f1, f05, accuracy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # f1ss is f1 score of sensitivity(recall) and specificity
    # f05ss is f0.5 score of sensitivity(recall) and specificity
    f1ss, f05ss = 0.0, 0.0

    if (tps + fns > 0): recall = float(tps) / float(tps + fns)
    if (tps + fps > 0): precision = float(tps) / float(tps + fps)
    if (tns + fps > 0): specificity = float(tns) / float(tns + fps)
    if (recall + precision > 0): f1 = 2.0 * recall * precision / (recall + precision)
    if (recall + precision > 0): f05 = 1.25 * recall * precision / ((0.25 * precision) + recall)
    if (tps + tns + fns + fps > 0): accuracy = float(tps + tns) / float (tps + tns + fns + fps)

    if (recall + specificity > 0): f1ss = 2.0 * recall * specificity / (recall + specificity)
    if (recall + specificity > 0): f05ss = 1.25 * recall * specificity / ((0.25 * specificity) + recall)

    return (recall, precision, specificity, f1, f05, accuracy, f1ss, f05ss)


def calculate_confusion_matrix(y_gold, y_pred):
    """
    Calculate the confusion matrix 
    True Positive (TP), False Positive (FP), False Negative (FN), and True Negative (TN)
    """
    y_actual = copy.deepcopy(y_gold)
    y_hat = copy.deepcopy(y_pred)
    y_actual = y_actual.tolist()
    y_hat = y_hat.tolist()

    tps, fps, fns, tns = 0, 0, 0, 0
    y_hat_len = len(y_hat)

    for i in range(y_hat_len):
        if y_actual[i] == 1:
            if y_hat[i] == 1: tps += 1
            elif y_hat[i] == 0: fns += 1
        elif y_actual[i] == 0:
            if y_hat[i] == 0: tns += 1
            elif y_hat[i] == 1: fps += 1
    
    return (tps, fps, fns, tns)

def calculate_confusion_matrix_performance(y_gold, y_pred):
    """
    Calculate the confusion matrix and several evaluation metrics given ground truth and predicted class.
    They are:
    True Positive (TP), False Positive (FP), False Negative (FN), and True Negative (TN)
    Recall/Sensitivity, Precision, Specificity, F1-score, F0.5-score, F1-recall-specificity, F0.5-recall-specificity.
    """
    (tps, fps, fns, tns) = calculate_confusion_matrix(y_gold, y_pred)
    (recall, precision, specificity, f1, f05, accuracy, f1ss, f05ss) = calculate_performance(tps, fps, fns, tns)
    return (tps, fps, fns, tns, recall, precision, specificity, f1, f05, accuracy, f1ss, f05ss)
 
def get_binary_predictions(pred, threshold=0.5):
    """
    Convert real number predictions between 0.0 to 1.0 to binary predictions (either 0 or 1)
    Using 0.5 as its default threshold unless specified
    """
    binary_pred = copy.deepcopy(pred)
    high_indices = binary_pred >= threshold
    low_indices = binary_pred < threshold
    binary_pred[high_indices] = 1
    binary_pred[low_indices] = 0

    return binary_pred

def compute_class_weight(train_y):
    """
    Compute class weight given imbalanced training data
    Usually used in the neural network model to augment the loss function (weighted loss function)
    Favouring/giving more weights to the rare classes.
    """
    import sklearn.utils.class_weight as scikit_class_weight

    class_list = list(set(train_y))
    class_weight_value = scikit_class_weight.compute_class_weight('balanced', class_list, train_y)
    class_weight = dict()

    # Initialize all classes in the dictionary with weight 1
    curr_max = np.max(class_list)
    for i in range(curr_max):
        class_weight[i] = 1

    # Build the dictionary using the weight obtained the scikit function
    for i in range(len(class_list)):
        class_weight[class_list[i]] = class_weight_value[i]

    logger.info('Class weight dictionary: ' + str(class_weight))
    return class_weight

def get_permutation_list(args, train_y):
    """
    Produce a fixed list of permutation indices for training
    So that we dont depend on third party shuffling
    """
    if (args.shuffle_path):
        logger.info('Loading shuffle permutation list from : %s' % args.shuffle_path)
        permutation_list = np.loadtxt(args.shuffle_path, dtype=int)
        logger.info('Loading shuffle permutation list completed!')
        return permutation_list
    
    shuffle_list_filename = "shuffle_permutation_list_len" + str(len(train_y)) + "_seed" + str(args.shuffle_seed) + ".txt"

    logger.info('Creating and saving shuffle permutation list to %s' % shuffle_list_filename)
    if args.shuffle_seed > 0:
        np.random.seed(args.shuffle_seed)

    # Create permutation list with the number of 2 x epochs
    # Actually 1x number of epoch is enough but just to be safe if need for other purposes
    permutation_list = []
    for ii in range(args.epochs*2):
        p = np.random.permutation(len(train_y))
        permutation_list.append(p)

    permutation_list = np.asarray(permutation_list, dtype=int)
    np.savetxt(args.out_dir_path + "/" + shuffle_list_filename, permutation_list, fmt='%d')

    logger.info('Creating and saving shuffle permutation list completed!')
    return permutation_list

