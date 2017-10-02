import argparse
import sys
import re

###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-in", "--input", dest="input_path", type=str, metavar='<str>', required=True, help="The path to the input text to be tokenized")
parser.add_argument("-out", "--output", dest="output_path", type=str, metavar='<str>', required=True, help="The path to file to write the tokenized text")
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path

import reader as reader
print("Tokenizing dataset...")
print("It will take a while...")
reader.tokenize_dataset(input_path, output_path)
