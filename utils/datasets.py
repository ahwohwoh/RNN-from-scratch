import os
import numpy as np
import pandas as pd
import nltk


class Sentiment():

    def __init__(self, data_rpath='data/'):
        # download nltk tokenizer
        nltk.download('punkt')
        # load data
        self._load_data(os.path.join(data_rpath, 'corpus.csv'))
        # build dictionary from data
        dictionary_path = os.path.join(data_rpath, 'dictionary.csv')
        # if not os.path.exists(dictionary_path):
        self._build_dictionary(np.concatenate([self.x_train, self.x_test, self.x_val]), dictionary_path)
        self.dictionary = self._load_dictionary(dictionary_path)

    def _load_data(self, path, val_size=100, test_size=100):
        data = pd.read_csv(path, sep='\t', header=None)
        x = []
        y = []
        for i in range(len(data.values)):
            x.append(data.values[i][1])
            y.append(data.values[i][0])
        x = np.asarray(x)
        y = np.asarray(y, dtype=np.int32)
        # random sample train/val/test indices
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        self.x_train = x[indices[test_size+val_size:]]
        self.y_train = y[indices[test_size+val_size:]]
        self.x_val = x[indices[test_size: test_size+val_size]]
        self.y_val = y[indices[test_size: test_size+val_size]]
        self.x_test = x[indices[:test_size]]
        self.y_test = y[indices[:test_size]]
        self.num_train = self.x_train.shape[0]
        self.num_val = self.x_val.shape[0]
        self.num_test = self.x_test.shape[0]
        print('Number of training samples: {}'.format(self.num_train))
        print('Number of validation samples: {}'.format(self.num_val))
        print('Number of testing samples: {}'.format(self.num_test))

    def _build_dictionary(self, sentences, path):
        word_set = set()
        for s in sentences:
            words = nltk.word_tokenize(s.lower())
            word_set.update(words)
        dic = {}
        dic['word'] = list(word_set)
        df = pd.DataFrame(data=dic)
        df.to_csv(path, sep='\t', header=None, index=False)

    def _load_dictionary(self, path):
        data = pd.read_csv(path, sep='\t', header=None)
        dic = dict()
        for i in range(len(data.values)): # leave index 0 for ending of a sentence
            dic[data.values[i][0]] = i+1
        return dic

    def train_loader(self, batch, shuffle=True):
        pointer = 0
        while True:
            if shuffle:
                idx = np.random.choice(self.num_train, batch, replace=False)
            else:
                if pointer+batch <= self.num_train:
                    idx = np.arange(pointer, pointer+batch)
                    pointer = pointer + batch
                else:
                    pointer = 0
                    idx = np.arange(pointer, pointer+batch)
                    pointer = pointer + batch
            print("self._one_hot_encoding(self.x_train[idx])", self._one_hot_encoding(self.x_train[idx]).shape)
            print("self.y_train[idx].shape", self.y_train[idx].shape)
            yield self._one_hot_encoding(self.x_train[idx]), self.y_train[idx]

    def test_loader(self, batch):
        pointer = 0
        while pointer+batch <= self.num_test:
            idx = np.arange(pointer, pointer+batch)
            pointer = pointer + batch
            yield self._one_hot_encoding(self.x_test[idx]), self.y_test[idx]
        if pointer < self.num_test-1:
            idx = np.arange(pointer, self.num_test-pointer-1)
            pointer = self.num_test-1
            yield self._one_hot_encoding(self.x_test[idx]), self.y_test[idx]
        else:
            return None

    def val_loader(self, batch):
        pointer = 0
        while pointer+batch <= self.num_val:
            idx = np.arange(pointer, pointer+batch)
            pointer = pointer + batch
            yield self._one_hot_encoding(self.x_val[idx]), self.y_val[idx]
        if pointer < self.num_val-1:
            idx = np.arange(pointer, self.num_val-pointer-1)
            pointer = self.num_val-1
            yield self._one_hot_encoding(self.x_val[idx]), self.y_val[idx]
        else:
            return None

    def _one_hot_encoding(self, sentences, max_length=30):
        vocab_size = len(self.dictionary)
        wordvecs = [] # of shape (N, T, V)
        mask = [] # of shape (N, T)
        for s in sentences:
            words = nltk.word_tokenize(s.lower())
            tmpw = [0 for i in range(max_length)]
            tmpm = [0 for i in range(max_length)]
            for idx, w in enumerate(words):
                if idx >= max_length:
                    break
                tmpw[idx] = self.dictionary[w]
                tmpm[idx] = 1
            one_hot = [[0 for i in range(vocab_size)] for j in range(max_length)] # (T, V)
            for i in range(len(tmpw)):
                if tmpw[i]:
                    one_hot[i][tmpw[i]-1] = 1
            one_hot = np.asarray(one_hot, dtype=np.float32)
            tmpm = np.asarray(tmpm, dtype=np.bool)
            one_hot[~tmpm, :] = np.nan
            wordvecs.append(one_hot)
            mask.append(tmpm)
        return np.array(wordvecs)