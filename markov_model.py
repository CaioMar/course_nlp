from typing import List, Optional
from collections import defaultdict

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
        self.priors = dict()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MarkovModelClassifier':
        dataset = X.copy()
        dataset['target'] = y
        self.classes = list(y.unique())
        self.priors = y.value_counts(normalize=True).to_dict()
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
            initial_state_distribution: np.ndarray,
            prior: float
        ) -> float:
        
        if not len(sequence):
            return np.log(1/self.vocabulary_size)

        log_prob = initial_state_distribution[sequence[0]]
        
        for i, index in enumerate(sequence):
            if i != len(sequence) - 1:
                log_prob += transition_matrix[index][sequence[i+1]]

        return log_prob + np.log(prior)

        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        
        output = list()
        for class_ in self.classes:
            output.append(X[self.document_column].apply(
                lambda x: self._sequence_log_probability(x, 
                    self.transition_matrices[class_], 
                    self.initial_state_dicts[class_],
                    self.priors[class_]
                    )
                )
            )

        output = np.exp(np.dstack(output).reshape(-1, len(self.classes)))

        return output/output.sum(axis=1, keepdims=True)

    def predict(self, X: pd.DataFrame) -> np.ndarray:

        output = self.predict_proba(X)

        return np.argmax(output, axis=1)


class MarkovModelGenerator:

    def __init__(
            self,
            document_column: str, 
            epsilon: float = 1.0
        ) -> None:
        """
        Class responsible for creating generative language model by fiting to
        a training set.
        """
        self.epsilon = epsilon
        self.document_column = document_column
        self.transition_tensor = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.initial_transition_matrix = defaultdict(lambda: defaultdict(int))
        self.initial_state_vector = defaultdict(int)
        self.vocabulary = []
        self.vocabulary_size = 0
        self.poem = []

    @staticmethod
    def _get_vocabulary(sequences: List[List[str]]) -> List[str]:
        vocabulary = []
        for sequence in sequences:
            vocabulary += sequence
        vocabulary = list(set(vocabulary))
        return vocabulary

    def fit(self, X: pd.DataFrame) -> 'MarkovModelGenerator':

        dataset = X.copy()

        self.vocabulary = self._get_vocabulary(dataset[self.document_column])
        self.vocabulary_size = len(self.vocabulary)

        number_of_sequences = X.shape[0]

        #Initizalization of Matrices
        token_vector = defaultdict(int)
        token_matrix = defaultdict(lambda: defaultdict(int))

        #Counts
        for tokens in dataset[self.document_column].tolist():

            if len(tokens):
                self.initial_state_vector[tokens[0]] += 1
                if len(tokens) > 1:
                    token_vector[tokens[0]] += 1
                    self.initial_transition_matrix[tokens[0]][tokens[1]] += 1
                elif len(tokens) == 1:
                    token_vector[tokens[0]] += 1
                    self.initial_transition_matrix[tokens[0]]['\n'] += 1

            for i, token in enumerate(tokens):
                if i < len(tokens) - 2:
                    token_matrix[token][tokens[i+1]] += 1
                    self.transition_tensor[token][tokens[i+1]][tokens[i+2]] += 1
                elif i == len(tokens) - 2:
                    token_matrix[token][tokens[i+1]] += 1
                    self.transition_tensor[token][tokens[i+1]]['\n'] += 1

        #Normalization
        for start_state in self.initial_state_vector.keys():
            self.initial_state_vector[start_state] = (
                (self.initial_state_vector[start_state] + self.epsilon)/(number_of_sequences + self.epsilon*self.vocabulary_size)
                )

        for start_state, fst_transition_dict in self.initial_transition_matrix.items():        
            for fst_transition in fst_transition_dict.keys():
                self.initial_transition_matrix[start_state][fst_transition] = (
                    (self.initial_transition_matrix[start_state][fst_transition] + self.epsilon)/(token_vector[start_state] + self.epsilon*self.vocabulary_size)
                )

        for start_state, fst_transition_dict in self.transition_tensor.items():        
            for fst_transition, end_state_dict in fst_transition_dict.items():               
                for end_state in end_state_dict.keys():
                    self.transition_tensor[start_state][fst_transition][end_state] = (
                        (self.transition_tensor[start_state][fst_transition][end_state] + self.epsilon)/(token_matrix[start_state][fst_transition] + self.epsilon*self.vocabulary_size)
                    )
        
        return self

    def sample(
            self,
            number_of_verses: int = 10,
        ) -> str:
        
        self.poem = []
        for _ in range(number_of_verses):
            self.poem.append([])

        for i in range(number_of_verses):
            
            while True:
                transition = dict()
                if len(self.poem[i]) == 0:
                    transition = self.initial_state_vector
                elif len(self.poem[i]) == 1:
                    transition = self.initial_transition_matrix[self.poem[i][0]]
                else:
                    transition = self.transition_tensor[self.poem[i][-2]][self.poem[i][-1]]
                
                remaining_words = list(set(self.vocabulary) - set(transition.keys()))
                a = list(transition.keys()) + remaining_words
                p_tmp = list(transition.values())
                remaining_words_prob = (1-sum(p_tmp))/len(remaining_words)
                #print(len(a), len(remaining_words), remaining_words_prob, sum(p_tmp))
                p = p_tmp + [remaining_words_prob]*len(remaining_words)
                #print(sum(p))

                new_word = np.random.choice(
                    a,
                    p=p
                )

                if new_word == '\n':
                    break
                
                self.poem[i].append(new_word)
                
            self.poem[i] = " ".join(self.poem[i])

        return "\n".join(self.poem)