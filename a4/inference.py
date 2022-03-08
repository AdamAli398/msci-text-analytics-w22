# Adam Ali (20657020)
# A4

import sys
import json
import keras
import os
import numpy as np

from config import config
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

def word_seq(lines, word2token):
    lines = ['<sos>,' + ' '.join(line.split()[:config['max_seq_len'] - 2]) + ',<eos>' for line in lines]
    lines_seq = [text_to_word_sequence(line, filters='!\"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n') for line in lines]

    x = [tokenize(line, word2token) for line in lines_seq]
    return pad_sequences(x, maxlen=config['max_seq_len'], padding='post', truncating='post', value=1)

def tokenize(sentence, word2token):
    tokenized = []
    for w in sentence:
        w = w.lower()
        token_id = word2token.get(w)
        if token_id is None:
            tokenized.append(word2token['<unk>'])
        else:
            tokenized.append(token_id)
    return tokenized

def predict(pred):
    return 'Positive' if pred[0] > pred[1] else 'Negative'

def main(text_path, model_code):
    # load data
    with open('data/word2token.json') as f:
        word2token = json.load(f)
    with open(text_path) as f:
        sample_text = f.readlines()
    model = keras.models.load_model(
        os.path.join('data/nn_{}.model'.format(model_code)))

    lines = [line.strip() for line in sample_text]
    lines = np.array(lines)
    lines = word_seq(lines, word2token)
    preds = model.predict(lines)
    return ['{} => {}'.format(sample_text[i], predict(preds[i]))
            for i, line in enumerate(lines)]

if __name__ == '__main__':
    predictions = main(sys.argv[1], sys.argv[2])
    print('\n'.join(predictions))