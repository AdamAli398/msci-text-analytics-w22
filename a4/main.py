# Adam Ali (20657020)
# A4

import os
import os.path
import sys
import re
import nltk
import json
import numpy as np
import keras
import tensorflow as tf

from tqdm import tqdm
from gensim.models import Word2Vec
from config import config
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense, TextVectorization

def read_dataset(data_dir):
    with open(os.path.join(data_dir, 'pos.txt')) as file:
        pos_lines = file.readlines()
    with open(os.path.join(data_dir, 'neg.txt')) as file:
        neg_lines = file.readlines()
    all_lines = pos_lines + neg_lines
    return list(zip(all_lines, [1] * len(pos_lines) + [0] * len(neg_lines)))

def tokenize(sentence, word2token):
    tokenized = []
    for w in sentence.lower().split():
        token_id = word2token.get(w)
        if token_id is None:
            tokenized.append(word2token['<unk>'])
        else:
            tokenized.append(token_id)
    return tokenized

def get_sentences(reviews):
    sentences = []

    for idx, line in tqdm(enumerate(reviews)):
        # Remove regex chars from each line: !"#$%&()*+/:;<=>@[//]^`{|}~\t\n
        sentence_no_punctuation = re.sub(r"!|\"|#|\$|%|&|\(|\)|\*|\+|/|:|;|<|=|>|@|\[|\\|\]|\^|`|\{|\|\}|~|\t|\n", ' ', line[0])

        # Split lines into tokenized words
        tokens = nltk.word_tokenize(sentence_no_punctuation.lower())

        # Append tokenized words into array
        sentences.append(tokens)

    return sentences

def train_w2v_model(sentences):
    model = Word2Vec(sentences, min_count=1, vector_size=100, window=5, workers=4)
    model.save('a4/data/w2v.model')
    return model

def load_autoencoder_data(data_dir):
    with open(os.path.join(data_dir, 'pos.txt')) as file:
        pos_lines = file.readlines()
    with open(os.path.join(data_dir, 'neg.txt')) as file:
        neg_lines = file.readlines()
    all_lines = pos_lines + neg_lines
    return ['<sos> ' + ' '.join(line.split()[:config['max_seq_len'] - 2]) + ' <eos>'
            for line in all_lines]

def build_emb_mat(data_dir, vocab, w2v):
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

    # randomly initizlize embeddings for the special tokens
    # you can play with different types of initializers
    embedding_matrix[0] = np.random.random((1, embedding_dim))
    embedding_matrix[1] = np.random.random((1, embedding_dim))
    embedding_matrix[2] = np.random.random((1, embedding_dim))
    embedding_matrix[3] = np.random.random((1, embedding_dim))
    for i, word in enumerate(vocab):
        # since a word in the vocab of our vectorizer is actually stored as
        # byte values, we need to decode them as strings explicitly
        # TODO: check if word below needs to be decoded, already a string?
        if not isinstance(word, str):
            print("not a string")
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
    with open(os.path.join(data_dir, 'token2word.json'), 'w') as f:
        json.dump(token2word, f)
    with open(os.path.join(data_dir, 'word2token.json'), 'w') as f:
        json.dump(word2token, f)
    return embedding_matrix, word2token

def main(data_dir):
    """
    loads the pos.txt and neg.txt, trains and saves a model using gensim Word2Vec
    """
    if os.path.isfile('a3/data/w2v.model'):
        w2v = Word2Vec.load('a3/data/w2v.model')
    else:
        return

    # Load autoencoder data format
    print('Loading autoencoder data')
    x_lines = load_autoencoder_data(data_dir)
    # Uncomment next line to decrease dataset size for quick testing - also necessary when computer does not have
    # enough RAM or memory to deal with the dataset size
    x_lines = x_lines[:1000]

    vectorizer = TextVectorization(max_tokens=config['max_vocab_size'],
                                   output_sequence_length=config['max_seq_len'])
    text_data = tf.data.Dataset.from_tensor_slices(x_lines).batch(config['batch_size'])
    print('Building vocabulary')
    vectorizer.adapt(text_data)
    # NOTE: in this vocab, index 0 is reserved for padding and 1 is reserved
    # for out of vocabulary tokens
    vocab = vectorizer.get_vocabulary()

    print('Building embedding matrix')
    # This matrix will be used to initialze weights in the embedding layer
    embedding_matrix, word2token = build_emb_mat(data_dir, vocab, w2v)
    print('embedding_matrix.shape => {}'.format(embedding_matrix.shape))

    print('Building Seq2Seq model')

    # build the embedding layer to convert token sequences into embeddings
    # set trainable to True if you wish to further finetune the embeddings.
    # It will increase train time but may yield better results. Try it out
    # on a more complex task (like neural machine translation)!
    embedding_layer = Embedding(
        input_dim=len(vocab) + 4,
        output_dim=config['embedding_dim'],
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )

    # build the encoding layers
    # encoder_inputs accepts padded tokenized sequences as input,
    # which would be converted to embeddings by the embedding_layer
    # finally, the embedded sequences are fed to the encoder LSTM to get
    # encodings (or vector representation) of the input sentences
    # you can add droputs the input/embedding layers and make your model robust
    encoder_inputs = Input((None,), name='enc_inp')
    enc_embedding = embedding_layer(encoder_inputs)
    # you can choose a GRU/Dense layer as well to keep things easier
    # note that we are not using the encoder_outputs for the given generative
    # task, but you'll need it for classification
    # Also, there hidden dimension is currently equal to the embedding dimension
    _, state_h, state_c = LSTM(config['embedding_dim'],  # try a different value
                               return_state=True,
                               name='enc_lstm')(enc_embedding)
    encoder_states = [state_h, state_c]

    # build the decoding layers
    # decoder_inputs and dec_embedding serve similar purposes as in the encoding
    # layers. Note that we are using the same embedding_layer to convert
    # token sequences to embeddings while encoding and decoding.
    # In this case, we initialize the decoder using `encoder_states`
    # as its initial state (i.e. vector representation learned by the encoder).
    decoder_inputs = Input((None,), name='dec_inp')
    dec_embedding = embedding_layer(decoder_inputs)
    dec_lstm = LSTM(config['embedding_dim'],
                    return_state=True,
                    return_sequences=True,
                    name='dec_lstm')
    dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=encoder_states)
    # finally, we add a final fully connected layer which performs the
    # transformation of decoder outputs to logits vectors
    dec_dense = Dense(len(vocab) + 4, activation='softmax', name='out')
    output = dec_dense(dec_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    print(model.summary())

    # switching to a generator instead of creating large matrices can reduce memory consumption a lot.
    encoder_input_data = np.ones(
        (len(x_lines), config['max_seq_len']),
        dtype='float32')
    decoder_input_data = np.ones(
        (len(x_lines), config['max_seq_len']),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(x_lines), config['max_seq_len'], len(vocab) + 4),
        dtype='float32')

    for i, input_text in enumerate(x_lines):
        tokenized_text = tokenize(input_text, word2token)
        for j in range(len(tokenized_text)):
            encoder_input_data[i, j] = tokenized_text[j]
            decoder_input_data[i, j] = tokenized_text[j]
            decoder_target_data[i, j, tokenized_text[j]] = 1.0

    print('Training model')
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Epochs selected at 500 because after that there is minimal change in accuracy of model
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=config['batch_size'],
              epochs=500,
              validation_split=0.2)

    # Save model
    model.save('a4/data/ae.model')
if __name__ == '__main__':
    main(sys.argv[1])