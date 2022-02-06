# Adam Ali (20657020)
# A2

import os
import sys
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from enum import Enum

class Stopwords(Enum):
    PRESENT = "with Stopwords"
    ABSENT = "without Stopwords"

class Text_Feature(Enum):
    UNIGRAMS = "Unigrams"
    BIGRAMS = "Bigrams"
    UNIGRAMS_BIGRAMS = "Unigrams & Bigrams"

def read_csv(data_path):
    with open(data_path) as f:
        data = f.readlines()
    return [' '.join(line.strip().split(',')) for line in data]

def load_data(data_dir, sw):
    if sw == Stopwords.PRESENT:
        x_train = read_csv(os.path.join(data_dir, 'train.csv'))
        x_val = read_csv(os.path.join(data_dir, 'val.csv'))
        x_test = read_csv(os.path.join(data_dir, 'test.csv'))
    elif sw == Stopwords.ABSENT:
        x_train = read_csv(os.path.join(data_dir, 'train_ns.csv'))
        x_val = read_csv(os.path.join(data_dir, 'val_ns.csv'))
        x_test = read_csv(os.path.join(data_dir, 'test_ns.csv'))
    else:
        pass
    labels = read_csv(os.path.join(data_dir, 'labels.csv'))
    labels = [int(label) for label in labels]
    y_train = labels[:len(x_train)]
    y_val = labels[len(x_train): len(x_train)+len(x_val)]
    y_test = labels[-len(x_test):]
    return x_train, x_val, x_test, y_train, y_val, y_test

def train(x_train, y_train, text_feature, sw):
    print(f'Calling CountVectorizer for {text_feature.value} {sw.value}')
    if text_feature == Text_Feature.UNIGRAMS:
        count_vect = CountVectorizer(ngram_range=(1, 1))
    elif text_feature == Text_Feature.BIGRAMS:
        count_vect = CountVectorizer(ngram_range=(2, 2))
    elif text_feature == Text_Feature.UNIGRAMS_BIGRAMS:
        count_vect = CountVectorizer(ngram_range=(1, 2))
    else:
        pass

    x_train_count = count_vect.fit_transform(x_train)
    print(f'Building Tf-idf vectors for {text_feature.value} {sw.value}')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    print(f'Training MNB for {text_feature.value} {sw.value}')
    clf = MultinomialNB().fit(x_train_tfidf, y_train)
    return clf, count_vect, tfidf_transformer

def evaluate(x, y, clf, count_vect, tfidf_transformer):
    x_count = count_vect.transform(x)
    x_tfidf = tfidf_transformer.transform(x_count)
    preds = clf.predict(x_tfidf)
    return {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds),
        'recall': recall_score(y, preds),
        'f1': f1_score(y, preds),
        }

def save_files(clf, sw, text_feature, count_vect, tfidf_transformer):
    path_mnb = 'a2/data/mnb'
    path_count_vect = 'a2/data/count_vect'
    path_tfidf = 'a2/data/tfidf_transformer'
    path_text_feature = ''
    path_sw = '' if sw == Stopwords.PRESENT else "_ns"
    if text_feature == Text_Feature.UNIGRAMS:
        path_text_feature = '_uni'
    elif text_feature == Text_Feature.BIGRAMS:
        path_text_feature = '_bi'
    elif text_feature == Text_Feature.UNIGRAMS_BIGRAMS:
        path_text_feature = '_uni_bi'
    else:
        pass

    with open(path_mnb+path_text_feature+path_sw+'.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open(path_count_vect+path_text_feature+path_sw+'.pkl', 'wb') as f:
        pickle.dump(count_vect, f)
    with open(path_tfidf+path_text_feature+path_sw+'.pkl', 'wb') as f:
        pickle.dump(tfidf_transformer, f)

def print_scores(x_val, y_val, x_test, y_test, clf, count_vect, tfidf_transformer, text_feature, sw):
    scores = {}
    # validate
    print('Validating')
    scores['val'] = evaluate(x_val, y_val, clf, count_vect, tfidf_transformer)
    # test
    print('Testing')
    scores['test'] = evaluate(x_test, y_test, clf, count_vect, tfidf_transformer)
    print(f'Accuracy for {text_feature.value} test set {sw.value}:', scores['test']['accuracy'])

def main(data_dir):
    """
    loads the dataset along with labels, trains a simple MNB classifier
    and returns validation and test scores in a dictionary
    """
    # load data for present & absent stopwords
    for sw in Stopwords:
        x_train, x_val, x_test, y_train, y_val, y_test = load_data(data_dir, sw)

        # train for each text feature
        for feature in Text_Feature:
            clf, count_vect, tfidf_transformer = train(x_train, y_train, feature, sw)
            save_files(clf, sw, feature, count_vect, tfidf_transformer)
            print_scores(x_val, y_val, x_test, y_test, clf, count_vect, tfidf_transformer, feature, sw)

if __name__ == '__main__':
    main(sys.argv[1])