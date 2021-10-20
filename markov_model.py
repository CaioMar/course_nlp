from typing import List

import numpy as np
import pandas as pd


class MarkovModelClassifier:

    def __init__(
            self,
            document_column: str, 
            vocabulary_size: int, 
            epsilon: float = 1.0
        ) -> None:
        self.epsilon = epsilon
        self.vocabulary_size = vocabulary_size
        self.document_column = document_column
        self.transition_matrices = dict()
        self.initial_state_dicts = dict()
        self.classes = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MarkovModelClassifier':
        dataset = X.copy()
        dataset['target'] = y
        self.classes = list(y.unique())

        number_of_sequences = y.size

        for class_ in self.classes:

            #Initizalization of Matrices
            initial_state_vector = np.zeros(self.vocabulary_size)
            token_vector = np.zeros(self.vocabulary_size)
            transition_matrix = np.zeros([self.vocabulary_size, self.vocabulary_size])

            #Counts
            for indexes in dataset.query('target=="%s"' %class_)[self.document_column].tolist():

                if len(indexes):
                    initial_state_vector[indexes[0]] += 1

                for i, index in enumerate(indexes):
                    token_vector[index] += 1
                    if i != len(indexes) - 1:
                        transition_matrix[index][indexes[i+1]] += 1

            #Normalization
            for start_state in range(self.vocabulary_size):
                initial_state_vector[start_state] = (
                    (initial_state_vector[start_state] + self.epsilon)/(number_of_sequences + self.epsilon*self.vocabulary_size)
                    )
                for end_state in range(self.vocabulary_size):
                    transition_matrix[start_state][end_state] = (
                        (transition_matrix[start_state][end_state] + self.epsilon)/(token_vector[start_state] + self.epsilon*self.vocabulary_size)
                    )
            
            #Adds Transition Matrices and Initial state distribution of class to a key in corresponding dicts
            self.transition_matrices[class_] = np.log(transition_matrix)
            self.initial_state_dicts[class_] = np.log(initial_state_vector)
        
        return self

    def _sequence_log_probability(
            self,
            sequence: List[int], 
            transition_matrix: np.ndarray, 
            initial_state_distribution: np.ndarray
        ) -> float:
        
        if not len(sequence):
            return np.log(1/self.vocabulary_size)

        log_prob = initial_state_distribution[sequence[0]]
        
        for i, index in enumerate(sequence):
            if i != len(sequence) - 1:
                log_prob += transition_matrix[index][sequence[i+1]]

        return log_prob

        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        
        output = list()
        for class_ in self.classes:
            output.append(X[self.document_column].apply(
                lambda x: self._sequence_log_probability(x, 
                    self.transition_matrices[class_], 
                    self.initial_state_dicts[class_]
                    )
                )
            )

        output = np.dstack(output).reshape(-1, len(self.classes))

        return output/output.sum(axis=1, keepdims=True)

    def predict(self, X: pd.DataFrame) -> np.ndarray:

        output = self.predict_proba(X)

        return np.argmax(output, axis=1)
