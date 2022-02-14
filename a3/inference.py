# Adam Ali (20657020)
# A3

import sys
import re

from gensim.models import Word2Vec
from tqdm import tqdm

def read_words(data_path):
    with open(data_path) as file:
        lines = file.readlines()
    return list(lines)

def find_similar(model, word):
    try:
        print(f"\nTop 20 Words similar to '{word}': ")
        sims = model.wv.most_similar(word, topn=20)
        rank = 1
        for s in sims:
            print(f"{rank}) {s[0]}: {s[1]}")
            rank += 1
    except:
        print(f"'{word}' not found in the model's vocabulary.")

def main(data_path):
    words = read_words(data_path)
    model = Word2Vec.load('a3/data/w2v.model')
    for idx, word in tqdm(enumerate(words)):
        word = re.sub(r"!|\"|#|\$|%|&|\(|\)|\*|\+|/|:|;|<|=|>|@|\[|\\|\]|\^|`|\{|\|\}|~|\t|\n", ' ', word)\
            .strip()\
            .lower()
        find_similar(model, word)

if __name__ == '__main__':
        main(sys.argv[1])