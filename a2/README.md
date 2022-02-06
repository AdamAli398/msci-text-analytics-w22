# Assignment 2
To run this, go to your terminal and enter the following from the root folder:

```source venv/bin/activate```

```python3 a2/main.py a1/data```

The following results are obtained from this command:

**With Stopwords**\
_Unigrams_: 0.8081\
_Bigrams_: 0.821825\
_Unigrams & Bigrams_: 0.830725

**Without Stopwords**\
_Unigrams_: 0.8054\
_Bigrams_: 0.778125\
_Unigrams & Bigrams_: 0.8226625

// TODO - ADD WRITEUP EXPLAINING RESULTS

To run the inference.py file that allows you to predict the positive/negative nature of any sentence in a .txt file, 
run the following command:\
```python3 a2/inference.py [.txt file name] [classifier type, i.e. mnb_uni]```