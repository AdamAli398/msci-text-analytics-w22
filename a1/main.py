# Adam Ali (20657020)
# A1

import os
import sys
import json
import random
import re
import nltk

# Uncomment these lines if either library is not working upon running
# nltk.download('punkt')
# nltk.download('stopwords')

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk import word_tokenize

def read_dataset(data_path):
    with open(os.path.join(data_path, 'pos.txt')) as file:
        pos_lines = file.readlines()
    with open(os.path.join(data_path, 'neg.txt')) as file:
        neg_lines = file.readlines()
    all_lines = pos_lines + neg_lines
    return list(zip(all_lines, [1] * len(pos_lines) + [0] * len(neg_lines)))

def tokenize(all_lines):
    # Declare data variables
    vocab = {}
    vocab_ns = {}
    csv_data = ''
    csv_ns_data = ''
    train_data = ''
    train_ns_data = ''
    val_data = ''
    val_ns_data = ''
    test_data = ''
    test_ns_data = ''
    labels_data = ''

    # Declare stop_words as imported list of stop words from nltk
    stop_words = stopwords.words('english')

    # Declare training, validation & testing sizes
    total_lines = len(all_lines)
    train_size = int(0.8 * total_lines)
    val_size = int(0.1 * total_lines)

    # Iterate through all lines to tokenize each
    for idx, line in tqdm(enumerate(all_lines)):
        # Remove regex chars from each line: !"#$%&()*+/:;<=>@[//]^`{|}~\t\n
        sentence_no_punctuation = re.sub(r"!|\"|#|\$|%|&|\(|\)|\*|\+|/|:|;|<|=|>|@|\[|\\|\]|\^|`|\{|\|\}|~|\t|\n", ' ', line[0])

        # Split lines into tokenized words
        tokenized_words = word_tokenize(sentence_no_punctuation)
        # tokenized_words = sentence_no_punctuation.strip().split()
        label = line[1]

        # Make new tokenized list without stop words
        tokenized_words_ns = []
        for w in tokenized_words:
            if w.lower() not in stop_words:
                tokenized_words_ns.append(w)

        # Construct the entry to be added to the csv files
        csv_line = '{}\n'.format(','.join(tokenized_words))
        csv_data += csv_line
        csv_ns_line = '{}\n'.format(','.join(tokenized_words_ns))
        csv_ns_data += csv_ns_line
        labels_data += '{}\n'.format(label)

        # Determine if entry is added to train/val/test
        if idx < train_size:
            train_data += csv_line
            train_ns_data += csv_ns_line
        elif idx >= train_size and idx < train_size + val_size:
            val_data += csv_line
            val_ns_data += csv_ns_line
        else:
            test_data += csv_line
            test_ns_data += csv_ns_line

        # Construct vocab dictionary
        for word in tokenized_words:
            word = word.lower()
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1

        # Construct vocab dictionary without stopwords
        for word in tokenized_words_ns:
            word = word.lower()
            if word not in vocab_ns:
                vocab_ns[word] = 0
            vocab_ns[word] += 1

    # Save processed files
    save_files(vocab, vocab_ns, csv_data, csv_ns_data, train_data, train_ns_data, val_data, val_ns_data, test_data, test_ns_data, labels_data)

def save_files(vocab, vocab_ns, csv_data, csv_ns_data, train_data, train_ns_data, val_data, val_ns_data, test_data, test_ns_data, labels_data):
    # Save each file as requested
    with open('a1/data/out.csv', 'w') as file:
        file.write(csv_data)
    with open('a1/data/train.csv', 'w') as file:
        file.write(train_data)
    with open('a1/data/val.csv', 'w') as file:
        file.write(val_data)
    with open('a1/data/test.csv', 'w') as file:
        file.write(test_data)
    with open('a1/data/out_ns.csv', 'w') as file:
        file.write(csv_ns_data)
    with open('a1/data/train_ns.csv', 'w') as file:
        file.write(train_ns_data)
    with open('a1/data/val_ns.csv', 'w') as file:
        file.write(val_ns_data)
    with open('a1/data/test_ns.csv', 'w') as file:
        file.write(test_ns_data)
    with open('a1/data/labels.csv', 'w') as file:
        file.write(labels_data)
    with open('a1/data/vocab.json', 'w') as file:
        json.dump(vocab, file)
    with open('a1/data/vocab_ns.json', 'w') as file:
        json.dump(vocab_ns, file)

def main(raw_data_path):
    # Read in raw dataset
    all_lines = read_dataset(raw_data_path)

    # Shuffle lines
    random.shuffle(all_lines)

    # Tokenize the dataset
    tokenize(all_lines)

if __name__ == "__main__":
    main(sys.argv[1])