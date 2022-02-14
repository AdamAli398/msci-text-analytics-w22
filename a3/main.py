# Adam Ali (20657020)
# A3

import os
import os.path
import sys
import re
import nltk

from tqdm import tqdm
from gensim.models import Word2Vec

def read_dataset(data_dir):
    with open(os.path.join(data_dir, 'pos.txt')) as file:
        pos_lines = file.readlines()
    with open(os.path.join(data_dir, 'neg.txt')) as file:
        neg_lines = file.readlines()
    all_lines = pos_lines + neg_lines
    return list(zip(all_lines, [1] * len(pos_lines) + [0] * len(neg_lines)))

def tokenize(reviews):
    sentences = []

    for idx, line in tqdm(enumerate(reviews)):
        # Remove regex chars from each line: !"#$%&()*+/:;<=>@[//]^`{|}~\t\n
        sentence_no_punctuation = re.sub(r"!|\"|#|\$|%|&|\(|\)|\*|\+|/|:|;|<|=|>|@|\[|\\|\]|\^|`|\{|\|\}|~|\t|\n", ' ', line[0])

        # Split lines into tokenized words
        tokens = nltk.word_tokenize(sentence_no_punctuation.lower())

        # Append tokenized words into array
        sentences.append(tokens)

    return sentences

def main(data_dir):
    """
    loads the pos.txt and neg.txt, trains and saves a model using gensim Word2Vec
    """

    reviews = read_dataset(data_dir)
    # sentences = [line[0].strip().split() for line in reviews]
    sentences = tokenize(reviews)
    model = Word2Vec(sentences, min_count=1, vector_size=100, window=5, workers=4)
    model.save('a3/data/w2v.model')

if __name__ == '__main__':
    main(sys.argv[1])