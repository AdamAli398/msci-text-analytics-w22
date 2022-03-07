# Assignment 2
To run this, go to your terminal and enter the following from the root folder:

```source venv/bin/activate```

```python3 a2/main.py a1/data```

| Stopwords removed | text features    | Accuracy (test set) |
|-------------------|------------------|---------------------|
| yes               | unigrams         | 79.17125%           |
| yes               | bigrams          | 73.1525%            |
| yes               | unigrams+bigrams | 80.12625%           |
| no                | unigrams         | 79.3825%            |
| no                | bigrams          | 78.5825%            |
| no                | unigrams+bigrams | 81.23625%           |

To run the inference.py file that allows you to predict the positive/negative nature of any sentence in a .txt file, 
run the following command:\
```python3 inference.py [.txt file name] [classifier type, i.e. mnb_uni]```

Using a given example in this repo, substitute the .txt file name with ```../data/raw/a2_inference_data.txt``` and use any
text feature desired as the second argument.