# Adam Ali (20657020)
# A4

import os
import sys
import keras
import numpy as np
import tensorflow as tf
import json

from keras.models import Sequential
from keras.layers import Dropout, Flatten
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from config import config
from keras.layers import Embedding, Dense, TextVectorization

def read_csv(data_path):
    with open(data_path) as f:
        data = f.readlines()
    return data

def load_csv_data(data_dir):
    x_train = read_csv(os.path.join(data_dir, 'train.csv'))
    x_val = read_csv(os.path.join(data_dir, 'val.csv'))
    x_test = read_csv(os.path.join(data_dir, 'test.csv'))

    train_sentences = [' '.join(line.split(',')) for line in x_train]

    x_train = ['<sos>,' + ' '.join(line.split()[:config['max_seq_len'] - 2]) + ',<eos>' for line in x_train]
    x_val = ['<sos>,' + ' '.join(line.split()[:config['max_seq_len'] - 2]) + ',<eos>' for line in x_val]
    x_test = ['<sos>,' + ' '.join(line.split()[:config['max_seq_len'] - 2]) + ',<eos>' for line in x_test]

    labels = read_csv(os.path.join(data_dir, 'labels.csv'))
    labels = [int(label) for label in labels]

    y_train = labels[:len(x_train)]
    y_val = labels[len(x_train): len(x_train)+len(x_val)]
    y_test = labels[-len(x_test):]
    return x_train, x_val, x_test, y_train, y_val, y_test, train_sentences

def build_emb_mat(vocab, w2v):
    """
    Build the embedding matrix which will be used to initialize weights of
    the embedding layer in our seq2seq architecture
    """
    # we have 4 special tokens in our vocab
    token2word = {0: '<sos>', 1: '<pad>', 2: '<eos>', 3: '<unk>'}
    word2token = {'<sos>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3}
    # +4 for the four vocab tokens
    vocab_size = len(vocab) + 4
    embedding_dim = config['embedding_dim']
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # randomly initialize embeddings for the special tokens
    # you can play with different types of initializers
    embedding_matrix[0] = np.random.random((1, embedding_dim))
    embedding_matrix[1] = np.random.random((1, embedding_dim))
    embedding_matrix[2] = np.random.random((1, embedding_dim))
    embedding_matrix[3] = np.random.random((1, embedding_dim))
    for i, word in enumerate(vocab):
        # since a word in the vocab of our vectorizer is actually stored as
        # byte values, we need to decode them as strings explicitly
        if not isinstance(word, str):
            word = word.decode('utf-8')
        try:
            # again, +4 for the four special tokens in our vocab
            embedding_matrix[i + 4] = w2v.wv[word]
            # build token-id -> word dict (will be used when decoding)
            token2word[i + 4] = word
            # build word -> token-id dict (will be used when encoding)
            word2token[word] = i + 4
        except KeyError as e:
            # skip any oov words from the perspective of our trained w2v model
            continue

    # save the two dicts
    with open('a4/data/token2word.json', 'w') as f:
        json.dump(token2word, f)
    with open('a4/data/word2token.json', 'w') as f:
        json.dump(word2token, f)
    return embedding_matrix, word2token

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

def convert_csv_to_seq(lines, labels, word2token):
    # Pre-processing
    word_seq = [text_to_word_sequence(line, filters='!\"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n') for line in lines]

    X = [tokenize(line, word2token) for line in word_seq]
    X = pad_sequences(X, maxlen=config['max_seq_len'], padding='post', truncating='post', value=1)

    y = [label for label in labels]

    return X, y

def build_nn(embedding_matrix, vocab, activation_type, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"Building {activation_type} neural network model")
    # Build Sequential model by stacking neural net units
    model = Sequential()
    model.add(Embedding(
        input_dim=len(vocab) + 4,
        output_dim=config['embedding_dim'],
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
        input_length=config['max_seq_len'],
        name='word_embedding_layer'),
    )
    model.add(Flatten())
    model.add(Dense(
        config['embedding_dim'],
        activation=activation_type,
        name='hidden_layer'),
    )
    model.add(Dropout(0.23))
    model.add(Dense(
        2,
        activation='softmax',
        kernel_regularizer='l2',
        name='output_layer'),
    )
    # model.summary()

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )
    model.fit(
        X_train,
        y_train,
        batch_size=config['batch_size'],
        epochs=config['n_epochs'],
        validation_data=(X_val, y_val),
    )

    model.save('a4/data/nn_' + activation_type + '.model')

    score, acc = model.evaluate(X_test, y_test, batch_size=config['batch_size'])
    print(activation_type)
    print("Accuracy on Test Set = {0:4.3f}".format(acc))

def main(a1_data_dir):
    """
    loads the pos.txt and neg.txt, trains and saves a model using gensim Word2Vec
    """
    if os.path.isfile('a3/data/w2v.model'):
        w2v = Word2Vec.load('a3/data/w2v.model')
    else:
        return

    # Load CSV Data from A1
    X_train, X_val, X_test, y_train, y_val, y_test, train_sentences = load_csv_data(a1_data_dir)

    vectorizer = TextVectorization(max_tokens=config['max_vocab_size'],
                                   output_sequence_length=config['max_seq_len'])
    text_data = tf.data.Dataset.from_tensor_slices(train_sentences).batch(config['batch_size'])
    print('Building vocabulary')
    vectorizer.adapt(text_data)
    vocab = vectorizer.get_vocabulary()

    # Create embedding matrix
    print('Creating embedding matrix')
    embedding_matrix, word2token = build_emb_mat(vocab, w2v)

    # Convert CSV loaded data to split sequences
    X_train, y_train = convert_csv_to_seq(X_train, y_train, word2token)
    X_val, y_val = convert_csv_to_seq(X_val, y_val, word2token)
    X_test, y_test = convert_csv_to_seq(X_test, y_test, word2token)

    # Convert y sets to 2 dimensional tuple: 1->(1,0), 0->(0,1)
    y_train = [(1, 0) if y == 1 else (0, 1) for y in y_train]
    y_val = [(1, 0) if y == 1 else (0, 1) for y in y_val]
    y_test = [(1, 0) if y == 1 else (0, 1) for y in y_test]

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)

    build_nn(embedding_matrix, vocab, 'relu', X_train, y_train, X_val, y_val, X_test, y_test)
    build_nn(embedding_matrix, vocab, 'sigmoid', X_train, y_train, X_val, y_val, X_test, y_test)
    build_nn(embedding_matrix, vocab, 'tanh', X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == '__main__':
    main(sys.argv[1])
