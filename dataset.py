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
from helpers import load_variables
from vocabulary import Captions
from vocabulary import build_dictionary
from ordered_dict import DefaultListOrderedDict

stop_words = stopwords.words('english')
punct_re = re.compile('[{}]'.format(re.escape(string.punctuation)))

if not os.path.exists(config.DICTIONARY_PATH):
    build_dictionary()

flickr_vocab = load_variables(config.DICTIONARY_PATH)['words']

captions = Captions(config.CAPTIONS_PATH)
flickr_captions = captions.get_all_captions()


def extract_pos(sentence):
    candidates = list()
    sentence = punct_re.sub(' ', sentence)
    words = word_tokenize(sentence)
    words = list(map(lambda w: w.lower(), words))
    tagged_words = pos_tag(words)
    for word, tag in tagged_words:
        if tag in config.POS_TAGS and word not in stop_words:
            candidates.append(word)
    return candidates


def extract_tfidf(sentences, min_df=1, max_df=1.0, num_terms=config.NUM_TERMS):
    terms = list()
    vocabulary = [extract_pos(unidecode(text)) for text in sentences]
    vocabulary = list(np.unique(list(chain(*vocabulary))))
    vocabulary = [word for word in vocabulary if word in flickr_vocab]
    tfidf = TfidfVectorizer(vocabulary=vocabulary, min_df=min_df, max_df=max_df)
    x = tfidf.fit_transform(sentences)
    sorted_vocabulary = [v[0] for v in sorted(tfidf.vocabulary_.items(), key=operator.itemgetter(1))]
    sorted_array = np.fliplr(np.argsort(x.toarray()))
    for array in sorted_array:
        term = [sorted_vocabulary[w] for w in array[0:num_terms]]
        terms.append(term)
    return terms


def convert_caption_to_keywords():
    with open(config.KEYWORDS_PATH, 'w') as file:
        for k, v in flickr_captions.items():
            print(k)
            keywords = extract_tfidf([''.join(v)])
            file.write(k + "\t" + ','.join(keywords[0]) + "\n")


def convert_to_multi_hot_vectors(keywords, dictionary):
    indices = {word: idx for idx, word in enumerate(dictionary)}
    output = np.zeros((len(dictionary)), dtype=np.int32)
    for keyword in keywords:
        output[indices[keyword]] = 1
    return output


def preprocess_keywords():
    labels = list()
    with open(config.KEYWORDS_PATH, 'r') as file:
        line = file.readline()
        while line:
            _, keywords = re.split(r'\t', line)
            print(keywords)
            keywords = keywords.split(',')
            keywords = [unidecode(v.rstrip()) for v in keywords]
            keywords = convert_to_multi_hot_vectors(keywords, flickr_vocab)
            labels.append(keywords)
            line = file.readline()
    np.save(config.LABELS_PATH, np.array(labels))


if not os.path.exists(config.KEYWORDS_PATH):
    convert_caption_to_keywords()

if not os.path.exists(config.LABELS_PATH):
    preprocess_keywords()
