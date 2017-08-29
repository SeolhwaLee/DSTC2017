from gensim.models import word2vec
import numpy as np
import json
import glob
import os
from pprint import pprint
from collections import Counter


class dataload():
    ''' Created by seol, don't use'''
    def __init__(self, path):
        try:
            print('Data load')
            pattern = os.path.join(path, '*.json')
            file_list = glob.glob(pattern)
            votes = ['O','X','T']

            training_dict = {}

            for i in file_list:
                with open(i, 'r') as fr:
                    data = json.load(fr)

                    for each in data['turns']:
                        print(each['turn-index'], each['speaker'], each['utterance'])
                        if each['speaker'] == 'U':
                            user = each['utterance']
                        if each['speaker'] == 'S':
                            sys = each['utterance']
                            breakdown_count = 0
                            correct_count = 0
                            possibly_br_count = 0
                            for anno in each['annotations']:
                                # print(anno['breakdown'])
                                if anno['breakdown'] == votes[0]:
                                    breakdown_count += 1
                                elif anno['breakdown'] == votes[1]:
                                    correct_count += 1
                                elif anno['breakdown'] == votes[2]:
                                    possibly_br_count += 1
                            count_list = [breakdown_count, correct_count, possibly_br_count]
                            if count_list.index(max(count_list)) == 0:
                                print("result :", votes[0])
                                training_dict[sys + '\t' + user] = votes[0]
                            elif count_list.index(max(count_list)) == 1:
                                print("result :", votes[1])
                                training_dict[sys + '\t' + user] = votes[1]
                            elif count_list.index(max(count_list)) == 2:
                                print("result :", votes[2])
                                training_dict[sys + '\t' + user] = votes[2]
            print(training_dict)
        except:
            print('Data load Error')


class UtteranceEmbed():
    def __init__(self, file_name, dim=300):
        self.dim = dim
        try:
            print('Loading english word2vec model')
            self.word2vec_model = word2vec.Word2Vec.load(file_name)
        except:
            print('Error while loading word2vec model')

    def embed_utterance(self, utterance):
        utterance = utterance.lower()
        word_embeddings = []
        for word in utterance.split(' '):
            if len(word):
                if "'" in word:
                    pre_word = word.split("'")[0]
                    after_word = "'" + word.split("'")[1]
                    if pre_word in self.word2vec_model:
                        word_embeddings.append(self.word2vec_model[pre_word])
                    if after_word in self.word2vec_model:
                        word_embeddings.append(self.word2vec_model[after_word])
                else:
                    if word in self.word2vec_model:
                        word_embeddings.append(self.word2vec_model[word])
        if len(word_embeddings):
            return np.mean(word_embeddings, axis=0)
        else:
            return np.zeros([self.dim], np.float32)

    def get_vector_size(self):
        return self.word2vec_model.vector_size

