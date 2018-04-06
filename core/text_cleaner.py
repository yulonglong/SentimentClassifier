
import nltk
import re

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
