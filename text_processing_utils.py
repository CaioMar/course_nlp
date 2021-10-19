"""
Contains text processing utility functions.
"""
from typing import List, Optional

from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd

class Tokenizer(BaseEstimator, TransformerMixin):
    
    def __init__(
        self, 
        document_column: str,
        char_removal: str = '"!?,;^~´[]}{)(\/ªº#|:.*+-', 
        to_lowercase: bool = True,
        stopwords: List[str] = [],
        missing_value: str = 'NA'
    ) -> None:
        self.char_removal = char_removal
        self.to_lowercase = to_lowercase
        self.stopwords = stopwords
        self.document_column = document_column
        self.missing_value = missing_value

    @staticmethod
    def removal(x: str, remove_list: List):
        for i in remove_list:
            x = x.replace(i, '')
        return x

    @staticmethod
    def remove_token(x: List, token_list: List):
        for i in token_list:
            while i in x:
                x.remove(i)
        return x

    def _get_vocabulary(self, sequences: List[List[str]]) -> List[str]:
        vocabulary = []
        for sequence in sequences:
            vocabulary += sequence
        vocabulary = list(set(vocabulary))
        vocabulary += [self.missing_value]
        return vocabulary

    def _gen_index_sequence(self, x: List[str], vocabulary: List[str]) -> List[str]:
        index_sequence = []
        for i in x:
            if i in vocabulary:
                index_sequence.append(vocabulary.index(i))
            else:
                index_sequence.append(vocabulary.index(self.missing_value))
        return index_sequence

    def _tokenization(
        self,
        X: pd.Series,       
    ) -> pd.Series:

        #lowercases all tokens
        if self.to_lowercase:
            X = X.apply(lambda x: x.lower())

        #removes punctuation
        X = X.apply(lambda x: self.removal(x, list(self.char_removal)))
    
        #tokenization
        X = X.apply(lambda x: x.split())

        #stopword removal
        X = X.apply(lambda x: self.remove_token(x, self.stopwords))

        return X
    
    def fit(self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'Tokenizer':
        dataset = X[self.document_column].copy()
        dataset = self._tokenization(dataset)
        self.vocabulary = self._get_vocabulary(dataset)
        self.vocab_size = len(self.vocabulary)
        return self
        
    def transform(self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        dataset = X[self.document_column].copy()
        dataset = self._tokenization(dataset)
        dataset = dataset.apply(lambda x: self._gen_index_sequence(x, self.vocabulary))
        return dataset.to_frame()






