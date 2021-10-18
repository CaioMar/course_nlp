"""
Contains text processing utility functions.
"""
from typing import List
import pandas as pd

def tokenizer(
    dataset: pd.Series,
    char_removal: str = '"!?,;^~´[]}{\/ªº#|:.*+-', 
    to_lowercase: bool = True,
    stopwords: List[str] = []) -> pd.Series:
    
    def removal(x: str, remove_list: List):
        for i in remove_list:
            x = x.replace(i, '')
        return x

    def remove_token(x: List, token_list: List):
        for i in token_list:
            if i in x:
                x.remove(i)
        return x

    #lowercases all tokens
    if to_lowercase:
        dataset = dataset.apply(lambda x: x.lower())

    #removes punctuation
    dataset = dataset.apply(lambda x: removal(x, list(char_removal)))
   
    #tokenization
    dataset = dataset.apply(lambda x: x.split())

    #stopword removal
    dataset = dataset.apply(lambda x: remove_token(x, stopwords))

    return dataset

