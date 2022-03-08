# Assignment 4
To run this, go to your terminal and enter the following from the root folder:

```source venv/bin/activate```

```python3 a4/main.py a1/data```

| Activation Functions | Classification Accuracy (test set) |
|----------------------|------------------------------------|
| ReLU                 | 75.3%                              |
| sigmoid              | 74.7%                              |
| tanh                 | 72.8%                              |

To run the inference.py file that allows you to predict the positive/negative nature of any sentence in a .txt file, 
run the following command:\
```python3 inference.py [.txt file name] [classifier type, i.e. relu]```

Using a given example in this repo, substitute the .txt file name with ```../data/raw/a2_inference_data.txt``` and use any
text feature desired as the second argument.