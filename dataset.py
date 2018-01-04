import os
import re
import string
import operator
from itertools import chain

import numpy as np
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from captions import Captions
from vocabulary import Vocabulary


class Dataset:
    def __init__(self, labels_file, vocab_file, tags_file, captions_file, tags_count, verbose=True):
        self.tags_file = tags_file
        self.vocab_file = vocab_file
        self.labels_file = labels_file
        self.captions_file = captions_file

        self.vocab = None
        self.captions = None
        self.tags_count = tags_count
        self.verbose = verbose

    def build(self):
        self.vocab = self.__vocab()
        self.captions = self.__captions()

        if not os.path.exists(self.tags_file):
            self.__caption_to_tags()

        if not os.path.exists(self.labels_file):
            self.__generate_labels()

    def __caption_to_tags(self):
        with open(self.tags_file, 'w') as fp:
            for k, v in self.captions.items():
                if self.verbose:
                    print(k)
                tags = self.__sentences_to_tags([''.join(v)])
                fp.write(k + "\t" + ','.join(tags[0]) + "\n")

    def __generate_labels(self):
        labels = list()
        with open(self.tags_file, 'r') as fp:
            line = fp.readline()
            while line:
                _, tags = re.split(r'\t', line)
                tags = [unidecode(v.rstrip()) for v in tags.split(',')]
                tags = self.__to_one_hot_vectors(tags)
                labels.append(tags)
                line = fp.readline()
        np.save(self.labels_file, np.array(labels))

    def __to_pos(self, sentence):
        _words = list()
        sentence = self.__remove_punctuation(sentence)
        words = pos_tag(list(map(lambda w: w.lower(), word_tokenize(sentence))))
        for word, tag in words:
            if tag in self.__accepted_pos() and word not in self.__stop_words():
                _words.append(word)
        return _words

    def __sentences_to_tags(self, sentences, min_df=1, max_df=1.0):
        tags = list()
        words = [self.__to_pos(unidecode(sentence)) for sentence in sentences]
        words = list(np.unique(list(chain(*words))))
        words = [word for word in words if word in self.vocab]
        tfidf = TfidfVectorizer(vocabulary=words, min_df=min_df, max_df=max_df, norm='l2', sublinear_tf=True)
        x = tfidf.fit_transform(sentences)
        sorted_words = [v[0] for v in sorted(tfidf.vocabulary_.items(), key=operator.itemgetter(1))]
        sorted_array = np.fliplr(np.argsort(x.toarray()))
        for array in sorted_array:
            term = [sorted_words[w] for w in array[0:self.tags_count]]
            tags.append(term)
        return tags

    def __to_one_hot_vectors(self, tags):
        indices = {word: idx for idx, word in enumerate(self.vocab)}
        output = np.zeros((len(self.vocab)), dtype=np.int32)
        for tag in tags:
            output[indices[tag]] = 1
        return output

    def __remove_punctuation(self, sentence):
        punctuation_re = re.compile('[{}]'.format(re.escape(string.punctuation)))
        return punctuation_re.sub(' ', sentence)

    def __captions(self):
        caption_processor = Captions(self.captions_file)
        return caption_processor.get_all_captions()

    def __vocab(self):
        vocab = Vocabulary(self.captions_file, save_file=self.vocab_file)
        return vocab.load()['words']

    @staticmethod
    def __accepted_pos():
        return {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBG', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}

    @staticmethod
    def __stop_words():
        return stopwords.words('english')


if __name__ == "__main__":
    dataset = Dataset(config.LABELS_FILE, config.VOCABULARY_FILE,
                      config.TAGS_FILE, config.CAPTIONS_FILE, config.TAGS_COUNT)
    dataset.build()
