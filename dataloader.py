import os

import numpy as np
import pandas as pd

from bert_preprocess import BertPreprocess
from tfidf_preprocess import TfIdfPreprocess


class Dataloader:

    def __init__(self, data_path, processed_path, preprocess_model_name, preprocess, sentence_embedding_averaged):
        '''
        preprocess_model_name can be BERT model from huggingface or tfidf
        '''
        self.data_path = data_path
        self.processed_path = processed_path
        self.force_preprocess = preprocess
        self.preprocess_model_name = preprocess_model_name
        self.sentence_embedding_averaged = sentence_embedding_averaged

        if preprocess_model_name == 'tfidf':
            self.preprocess_model = TfIdfPreprocess()
        else:
            # BERT models
            self.preprocess_model = BertPreprocess(preprocess_model_name, sentence_embedding_averaged=sentence_embedding_averaged)


    def preprocess_data(self, dataset):
        '''
        Preprocess raw data with Bert
        '''
        mapping = {'negative':0, 'neutral':1, 'positive':2}

        y = []
            
        # Generate X set
        X = self.preprocess_model.feature_extraction(dataset['sentence'])

        # Generate y set. Label is of shape e.g. [0, 0, 1] according to the mapping
        for i in range(len(dataset['label'].values)):
            y_val = np.zeros(3)
            y_val[mapping[dataset['label'].values[i]]] = 1
            y.append(y_val)


        return np.array(X), np.array(y)


    def get_data(self, year=2013):
        '''
        Get data of all sets
        Year should be in [2013, ..., 2016]
        '''
        total_data = []

        for mode in ['train', 'test', 'dev']:
            total_data.append(self.get_single_data(year, mode))

        return total_data


    def get_single_data(self, year=2013, mode='train'):
        '''
        Gets data from tsv files
        Year should be in [2013, ..., 2016]
        '''
        assert year >= 2013 and year <= 2016

        xfile = self.processed_path + 'twitter-' + str(self.preprocess_model_name.split('/')[-1]) + '-' + str(year) + '-' + str(mode) + "-X-processed"
        yfile = self.processed_path + 'twitter-' + str(self.preprocess_model_name.split('/')[-1]) + '-' + str(year) + '-' + str(mode) + "-y-processed.npy"

        xfile += "-embedding-averaged-over-outputs.npy" if self.sentence_embedding_averaged else ".npy" 

        if os.path.isfile(xfile) and not self.force_preprocess:
            X = np.load(xfile, allow_pickle=True)
            y = np.load(yfile, allow_pickle=True)
            #assert X.shape[0] == y.shape[0]
            return X, y

        names = ['ID', 'label', 'sentence']

        data = pd.read_csv(self.data_path + "twitter-" + str(year) + str(mode) + "-A.txt", sep='\t', names=names, usecols=[0,1,2])

        X, y = self.preprocess_data(data)
        np.save(xfile, X, allow_pickle=True)
        np.save(yfile, y, allow_pickle=True)

        return X, y
