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

##############################################################################
## Tokenizer for text before training word2vec
#

def tokenize_dataset(file_path, output_path, to_lower=True):
    """
    Simple tokenizer for the text before training word2vec
    """
    from core import text_cleaner as text_cleaner
    import codecs
    
    output = open(output_path,"w")
    print('Reading dataset from: ' + file_path)

    with codecs.open(file_path, mode='r', encoding='ISO-8859-1') as input_file:
        for line in input_file:
            splitBrLine = line.replace("<br />", "\n").replace("<br/>", "\n").replace("<br>", "\n").split("\n")
            for subline in splitBrLine:
                content = subline

                if to_lower:
                    content = content.lower()

                content = text_cleaner.tokenize(content)

                for word in content:
                    if text_cleaner.isContainDigit(word):
                        output.write('<num>')
                    else:
                        output.write((word).encode("utf8"))
                    output.write(' ')

                output.write('\n')


print("Tokenizing dataset...")
print("It will take a while...")
tokenize_dataset(input_path, output_path)
