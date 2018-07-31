"""Evaluator.py - A python class/module to calculate neural network performance."""

import numpy as np
import pickle as pk
import copy
from core import helper as helper
from time import time
from torch.autograd import Variable
import torch
import logging

logger = logging.getLogger(__name__)

####################################################################################################
## Evaluator class
#

class Evaluator(object):
    """
    Evaluator class

    The class which handles evaluation and picking the best threshold and epoch
    after the neural network model is trained.
    It is also responsible to save the model and weights for future use.
    """

    def __init__(self, out_dir, train, dev, test, no_threshold=False, batch_size_eval=256):
        """
        Constructor to initialize the Evaluator class with the necessary attributes.
        """
        self.out_dir = out_dir
        self.batch_size_eval = batch_size_eval

        self.train_x, self.train_y, self.train_filename_y = (
            train[0], train[1], train[2])
        self.dev_x, self.dev_y, self.dev_filename_y = (
            dev[0], dev[1], dev[2])
        self.test_x, self.test_y, self.test_filename_y = (
            test[0], test[1], test[2])

        # Sort data based on their length and pad them only per batch_size
        self.train_x, self.train_y, self.train_filename_y, _ = (
            helper.sort_and_split_data_into_chunks(
                self.train_x, self.train_y, self.train_filename_y,
                batch_size_eval)
        )

        self.dev_x, self.dev_y, self.dev_filename_y, _ = (
            helper.sort_and_split_data_into_chunks(
                self.dev_x, self.dev_y, self.dev_filename_y,
                batch_size_eval)
        )

        self.test_x, self.test_y, self.test_filename_y, _ = (
            helper.sort_and_split_data_into_chunks(
                self.test_x, self.test_y, self.test_filename_y,
                batch_size_eval)
        )

        self.dev_mean = self.dev_y.mean()
        self.dev_std = self.dev_y.std()
        self.test_mean = self.test_y.mean()
        self.test_std = self.test_y.std()

        self.train_y_org = self.train_y.astype('int32')
        self.dev_y_org = self.dev_y.astype('int32')
        self.test_y_org = self.test_y.astype('int32')

        self.best_dev = [-1, 9999, 9999, -1, -1, -1, -1, -1, -1, -1, -1]
        self.best_test = [-1, 9999, 9999, -1, -1, -1, -1, -1, -1, -1, -1]
        self.best_dev_epoch = -1
        self.best_dev_threshold = 0.5
        self.best_test_missed = -1
        self.best_test_missed_epoch = -1
        self.dump_ref_scores()
        self.dump_ref_filenames()
        
        self.dev_loss, self.dev_metric = 0.0, 0.0
        self.test_loss, self.test_metric = 0.0, 0.0

        self.no_threshold = no_threshold
        self.threshold = 0.5

        self.train_tps = 0
        self.train_fps = 0
        self.train_fns = 0
        self.train_tns = 0
        self.train_recall = 0.0
        self.train_precision = 0.0
        self.train_f1 = 0.0
        self.train_accuracy = 0.0

        self.dev_tps = 0
        self.dev_fps = 0
        self.dev_fns = 0
        self.dev_tns = 0
        self.dev_recall = 0.0
        self.dev_precision = 0.0
        self.dev_f1 = 0.0
        self.dev_accuracy = 0.0

        self.test_tps = 0
        self.test_fps = 0
        self.test_fns = 0
        self.test_tns = 0
        self.test_recall = 0.0
        self.test_precision = 0.0
        self.test_f1 = 0.0
        self.test_accuracy = 0.0

        self.train_pred = np.array([])
        self.dev_pred = np.array([])
        self.test_pred = np.array([])

    def dump_ref_filenames(self):
        """Dump/print the reference (ground truth) filenames to a file"""
        dev_ref_filename_file = open(self.out_dir + '/preds/dev_ref_filenames.txt', "w")
        for idx, _ in enumerate(self.dev_filename_y):
            for dev_filename in self.dev_filename_y[idx]:
                dev_ref_filename_file.write(dev_filename + '\n')
        dev_ref_filename_file.close()

        test_ref_filename_file = open(self.out_dir + '/preds/test_ref_filenames.txt', "w")
        for idx, _ in enumerate(self.test_filename_y):
            for test_filename in self.test_filename_y[idx]:
                test_ref_filename_file.write(test_filename + '\n')
        test_ref_filename_file.close()

    def dump_ref_scores(self):
        """Dump reference (ground truth) scores"""
        np.savetxt(self.out_dir + '/preds/train_ref.txt', self.train_y_org, fmt='%i')
        np.savetxt(self.out_dir + '/preds/dev_ref.txt', self.dev_y_org, fmt='%i')
        np.savetxt(self.out_dir + '/preds/test_ref.txt', self.test_y_org, fmt='%i')

    def dump_train_predictions(self, train_pred, epoch):
        """Dump predictions of the model on the training set"""
        np.savetxt(self.out_dir + '/preds/train_pred_' + str(epoch) + '.txt',
                   train_pred, fmt='%.8f')

    def dump_predictions(self, dev_pred, test_pred, threshold, epoch):
        """Dump predictions of the model on the dev and test set"""
        np.savetxt(self.out_dir + '/preds/dev_pred_' + str(epoch) + '.txt', dev_pred, fmt='%.8f')
        np.savetxt(self.out_dir + '/preds/test_pred_' + str(epoch) + '.txt', test_pred, fmt='%.8f')
        np.savetxt(self.out_dir + '/preds/threshold_' + str(epoch) + '.txt', [threshold], fmt='%.8f')

    def get_best_threshold(self):
        """
        Given the real-valued scores in the development set,
        do a linear search to find the best threshold (i.e., the best F0.5 score)
        classifying the real-valued scores as 1 or 0.

        The threshold will later be used in the test set.
        """
        t0 = time()
        
        gold_y = copy.deepcopy(self.dev_y_org)
        pred_y = copy.deepcopy(self.dev_pred)

        sorted_pred_y, sorted_gold_y = (list(t) for t in zip(*sorted(zip(pred_y, gold_y))))
        assert len(sorted_gold_y) == len(gold_y)
        assert len(sorted_pred_y) == len(gold_y)

        tps, fps, fns, tns = 0, 0, 0, 0
        recall, precision, specificity, f1, f05, accuracy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        curr_best_threshold = 0
        curr_best_f1 = 0
        prev_idx = 0

        for i in range(1, len(sorted_pred_y)):
            if sorted_pred_y[i] > sorted_pred_y[i-1]:
                possible_threshold = (sorted_pred_y[i]+sorted_pred_y[i-1])/2.0
                # Use default method for first time calculation
                if prev_idx == 0:
                    binary_pred = helper.get_binary_predictions(pred_y, threshold=possible_threshold)
                    (tps, fps, fns, tns, recall, precision, specificity, f1, f05, accuracy, f1ss, f05ss) = helper.calculate_confusion_matrix_performance(gold_y, binary_pred)
                else:
                    # Check for each affected row/y as the threshold slides (instead of doing the if statement above which is O(n))
                    for j in range(prev_idx, i):
                        if sorted_gold_y[j] == 0:
                            tns += 1
                            fps -= 1
                        elif sorted_gold_y[j] == 1:
                            fns += 1
                            tps -= 1
                    (recall, precision, specificity, f1, f05, accuracy, f1ss, f05ss) = helper.calculate_performance(tps, fps, fns, tns)
                
                prev_idx = i
                # Update current best F1
                if curr_best_f1 < f1:
                    curr_best_threshold = possible_threshold
                    curr_best_f1 = f1
            
            # Uncomment below for debugging purposes
            # logger.warning("%d ::>>> threshold: %.5f >>> TP: %5d, FP: %5d, FN: %5d, TN: %5d, F05: %.3f" % (i,possible_threshold,tps,fps,fns,tns,f05))

        threshold_time = time() - t0
        logger.info('threshold finding: %is (%.1fm)' % (threshold_time, threshold_time/60.0))
        
        return curr_best_threshold

    def evaluate(self, model, epoch):
        """
        The main (most important) function in this class

        Evaluate a trained model at a given epoch on the development and test set.
        Handles all model saving, prediction score printing, dynamic padding by length sorting, etc.
        """

        # Reset train_pred, dev_pred and test_pred
        self.train_pred = np.array([])
        self.dev_pred = np.array([])
        self.test_pred = np.array([])

        for idx, _ in enumerate(self.train_x):
            pytorch_train_x = torch.from_numpy(self.train_x[idx].astype('int64'))
            if torch.cuda.is_available():
                pytorch_train_x = pytorch_train_x.cuda()
            curr_train_pred = model(Variable(pytorch_train_x))
            curr_train_pred = curr_train_pred.cpu().data.numpy()
           
            self.train_pred = np.append(self.train_pred, curr_train_pred)

        for idx, _ in enumerate(self.dev_x):
            pytorch_dev_x = torch.from_numpy(self.dev_x[idx].astype('int64'))
            if torch.cuda.is_available():
                pytorch_dev_x = pytorch_dev_x.cuda()
            curr_dev_pred = model(Variable(pytorch_dev_x))
            curr_dev_pred = curr_dev_pred.cpu().data.numpy()
            self.dev_pred = np.append(self.dev_pred, curr_dev_pred)

        for idx, _ in enumerate(self.test_x):
            pytorch_test_x = torch.from_numpy(self.test_x[idx].astype('int64'))
            if torch.cuda.is_available():
                pytorch_test_x = pytorch_test_x.cuda()
            curr_test_pred = model(Variable(pytorch_test_x))
            curr_test_pred = curr_test_pred.cpu().data.numpy()
            self.test_pred = np.append(self.test_pred, curr_test_pred)

        if (self.no_threshold): self.threshold = 0.5
        else: self.threshold = self.get_best_threshold()

        # self.dump_train_predictions(self.train_pred)
        self.dump_predictions(self.dev_pred, self.test_pred, self.threshold, epoch)

        binary_train_pred = helper.get_binary_predictions(self.train_pred, threshold=self.threshold)
        (self.train_tps, self.train_fps, self.train_fns, self.train_tns,
            self.train_recall, self.train_precision, _,
            self.train_f1, _, self.train_accuracy,
            _, _) = helper.calculate_confusion_matrix_performance(self.train_y_org, binary_train_pred)

        binary_dev_pred = helper.get_binary_predictions(self.dev_pred, threshold=self.threshold)
        (self.dev_tps, self.dev_fps, self.dev_fns, self.dev_tns,
            self.dev_recall, self.dev_precision, _,
            self.dev_f1, _, self.dev_accuracy,
            _, _) = helper.calculate_confusion_matrix_performance(self.dev_y_org, binary_dev_pred)

        binary_test_pred = helper.get_binary_predictions(self.test_pred, threshold=self.threshold)
        (self.test_tps, self.test_fps, self.test_fns, self.test_tns,
            self.test_recall, self.test_precision, _,
            self.test_f1, _, self.test_accuracy,
            _, _) = helper.calculate_confusion_matrix_performance(self.test_y_org, binary_test_pred)

        # Choose epoch based on f1
        if (self.dev_f1 > self.best_dev[6]):
            self.best_dev = [self.dev_tps, self.dev_fps, self.dev_fns, self.dev_tns,
                            self.dev_recall, self.dev_precision,
                            self.dev_f1, self.dev_accuracy]
            self.best_test = [self.test_tps, self.test_fps, self.test_fns, self.test_tns,
                            self.test_recall, self.test_precision,
                            self.test_f1, self.test_accuracy]
            self.best_dev_epoch = epoch
            self.best_dev_threshold = self.threshold
            # Save all necessary things
            torch.save(model, self.out_dir + '/models/best_model_weights.h5')
            np.savetxt(self.out_dir + '/models/best_threshold.txt', [self.best_dev_threshold], fmt='%.8f')
            self.dump_predictions(self.dev_pred, self.test_pred, self.threshold, 9999)

        if (self.test_f1 > self.best_test_missed):
            self.best_test_missed = self.test_f1
            self.best_test_missed_epoch = epoch

    def print_info(self):
        """
        Print and return the current performance of the model.
        """

        content = "" + (
            "\r\n\r\n[TRAIN]        (%.4f) TP: %5d, FP: %5d, FN: %5d, TN: %5d, FP+FN: %5d, R: %.3f, P: %.3f, F1: %.3f, A: %.5f" % (
            self.threshold, self.train_tps, self.train_fps, self.train_fns, self.train_tns, self.train_fps + self.train_fns,
            self.train_recall, self.train_precision,
            self.train_f1, self.train_accuracy)) + (
            "\r\n\r\n[DEV]          (%.4f) TP: %5d, FP: %5d, FN: %5d, TN: %5d, FP+FN: %5d, R: %.3f, P: %.3f, F1: %.3f, A: %.5f" % (
            self.threshold, self.dev_tps, self.dev_fps, self.dev_fns, self.dev_tns, self.dev_fps + self.dev_fns,
            self.dev_recall, self.dev_precision,
            self.dev_f1, self.dev_accuracy)) + (
            "\r\n\r\n[BEST-DEV @%2d] (%.4f) TP: %5d, FP: %5d, FN: %5d, TN: %5d, FP+FN: %5d, R: %.3f, P: %.3f, F1: %.3f, A: %.5f" % (
            self.best_dev_epoch, self.best_dev_threshold, self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3], self.best_dev[1] + self.best_dev[2],
            self.best_dev[4], self.best_dev[5], self.best_dev[6], self.best_dev[7])) + (
            "\r\n\r\n[TEST]         (%.4f) TP: %5d, FP: %5d, FN: %5d, TN: %5d, FP+FN: %5d, R: %.3f, P: %.3f, F1: %.3f, A: %.5f" % (
            self.threshold, self.test_tps, self.test_fps, self.test_fns, self.test_tns, self.test_fps + self.test_fns,
            self.test_recall, self.test_precision,
            self.test_f1, self.test_accuracy)) + (
            "\r\n\r\n[BEST-TEST @%2d](%.4f) TP: %5d, FP: %5d, FN: %5d, TN: %5d, FP+FN: %5d, R: %.3f, P: %.3f, F1: %.3f, A: %.5f" % (
            self.best_dev_epoch, self.best_dev_threshold, self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3], self.best_test[1] + self.best_test[2],
            self.best_test[4], self.best_test[5], self.best_test[6], self.best_test[7]))

        logger.info(content.replace("\r\n\r\n","\r\n"))
        logger.info('------------------------------------------------------------------------')

        return content
    
    def print_final_info(self):
        """Print and return the final performance of the model"""

        content = "" + (
            "\r\n\r\n[BEST-DEV @%2d] (%.4f) TP: %5d, FP: %5d, FN: %5d, TN: %5d, FP+FN: %5d, R: %.3f, P: %.3f, F1: %.3f, A: %.5f" % (
            self.best_dev_epoch, self.best_dev_threshold, self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3], self.best_dev[1] + self.best_dev[2], 
            self.best_dev[4], self.best_dev[5], self.best_dev[6],
            self.best_dev[7])) + (
            "\r\n\r\n[BEST-TEST @%2d](%.4f) TP: %5d, FP: %5d, FN: %5d, TN: %5d, FP+FN: %5d, R: %.3f, P: %.3f, F1: %.3f, A: %.5f" % (
            self.best_dev_epoch, self.best_dev_threshold, self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3], self.best_test[1] + self.best_test[2],
            self.best_test[4], self.best_test[5], self.best_test[6],
            self.best_test[7]))

        logger.info(content.replace("\r\n\r\n","\r\n"))

        return content

