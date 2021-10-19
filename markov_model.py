from typing import Union, Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd


class MarkovModelClassifier:

    def __init__(self, dependent_var: str, epsilon: float = 1.0) -> None:
        self.epsilon = epsilon
        self.dependent_var = dependent_var
        self.transition_matrices = dict()
        self.initial_state_dicts = dict()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X['target'] = y
        self.classes = y.unique()
        for class_ in self.classes:
            initial_state_dict = defaultdict(int)
            token_dict = defaultdict(int)
            transition_matrix = dict()
            for indexes in X.query('author=="%s"' %class_)[self.dependent_var].tolist():
                if len(indexes):
                    initial_state_dict[indexes[0]] += 1
                for i, index in enumerate(indexes):
                    token_dict[index] += 1
                    if index not in transition_matrix.keys():
                        transition_matrix[index] = defaultdict(int)
                    if i != len(indexes) - 1:
                        transition_matrix[index][indexes[i+1]] += 1
            total_tokens = len(token_dict.keys())
            for start_state in transition_matrix.keys():
                for end_state in transition_matrix[start_state].keys():
                    transition_matrix[start_state][end_state] = (
                        (transition_matrix[start_state][end_state] + self.epsilon)/(token_dict[start_state] + total_tokens)
                    )
            self.transition_matrices[class_] = transition_matrix
            self.initial_state_dicts[class_] = initial_state_dict

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        pass