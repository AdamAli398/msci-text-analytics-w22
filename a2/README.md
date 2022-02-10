# Assignment 2
To run this, go to your terminal and enter the following from the root folder:

```source venv/bin/activate```

```python3 a2/main.py a1/data```

| Stopwords removed | text features    | Accuracy (test set) |
|-------------------|------------------|---------------------|
| yes               | unigrams         | 80.5225%            |
| yes               | bigrams          | 77.925%             |
| yes               | unigrams+bigrams | 82.30875%           |
| no                | unigrams         | 80.82125%           |
| no                | bigrams          | 82.42%              |
| no                | unigrams+bigrams | 83.305%             |

To run the inference.py file that allows you to predict the positive/negative nature of any sentence in a .txt file, 
run the following command:\
```python3 a2/inference.py [.txt file name] [classifier type, i.e. mnb_uni]```

Using a given example in this repo, substitute the .txt file name with ```data/raw/a2_inference_data.txt``` and use any
text feature desired as the second argument.