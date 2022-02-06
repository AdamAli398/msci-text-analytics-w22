# Adam Ali (20657020)
# A2

import sys
import pickle

def predict(line, clf, count_vect, tfidf_transformer):
    line2feature = tfidf_transformer.transform(count_vect.transform([line]))
    return 'Postive' if clf.predict(line2feature) else 'Negative'

def main(text_path, model_code):
    sliced_model_code = model_code[3:]

    # load data
    with open(text_path) as f:
        sample_text = f.readlines()
    sample_text = [line.strip() for line in sample_text]

    with open('a2/data/{}.pkl'.format(model_code), 'rb') as f:
        clf = pickle.load(f)

    with open('a2/data/count_vect{}.pkl'.format(sliced_model_code), 'rb') as f:
        count_vect = pickle.load(f)

    with open('a2/data/tfidf_transformer{}.pkl'.format(sliced_model_code), 'rb') as f:
        tfidf_transformer = pickle.load(f)

    return ['{} => {}'.format(line, predict(line, clf, count_vect, tfidf_transformer))
            for line in sample_text]

if __name__ == '__main__':
    predictions = main(sys.argv[1], sys.argv[2])
    print('\n'.join(predictions))