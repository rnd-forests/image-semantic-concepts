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
from helpers import load_variables
from vocabulary import build_dictionary

stop_words = stopwords.words('english')
punctuation_re = re.compile('[{}]'.format(re.escape(string.punctuation)))

if not os.path.exists(config.DICTIONARY_PATH):
    build_dictionary()

captions = Captions(config.CAPTIONS_PATH)
flickr_captions = captions.get_all_captions()
flickr_vocab = load_variables(config.DICTIONARY_PATH)['words']


def pos(sentence):
    candidates = list()
    sentence = punctuation_re.sub(' ', sentence)
    words = list(map(lambda w: w.lower(), word_tokenize(sentence)))
    tagged_words = pos_tag(words)
    for word, tag in tagged_words:
        if tag in config.POS_TAGS and word not in stop_words:
            candidates.append(word)
    return candidates


def get_tags(sentences, min_df=1, max_df=1.0, num_terms=config.NUM_TERMS):
    tags = list()
    words = [pos(unidecode(sentence)) for sentence in sentences]
    words = list(np.unique(list(chain(*words))))
    words = [word for word in words if word in flickr_vocab]
    tfidf = TfidfVectorizer(vocabulary=words, min_df=min_df, max_df=max_df, norm='l2', sublinear_tf=True)
    x = tfidf.fit_transform(sentences)
    sorted_words = [v[0] for v in sorted(tfidf.vocabulary_.items(), key=operator.itemgetter(1))]
    sorted_array = np.fliplr(np.argsort(x.toarray()))
    for array in sorted_array:
        term = [sorted_words[w] for w in array[0:num_terms]]
        tags.append(term)
    return tags


def convert_caption_to_tags():
    with open(config.TAGS_PATH, 'w') as file:
        for k, v in flickr_captions.items():
            print(k)
            tags = get_tags([''.join(v)])
            file.write(k + "\t" + ','.join(tags[0]) + "\n")


def convert_to_multi_hot_vectors(tags, dictionary):
    indices = {word: idx for idx, word in enumerate(dictionary)}
    output = np.zeros((len(dictionary)), dtype=np.int32)
    for tag in tags:
        output[indices[tag]] = 1
    return output


def generate_labels():
    labels = list()
    print('Generating labels...')
    with open(config.TAGS_PATH, 'r') as file:
        line = file.readline()
        while line:
            _, tags = re.split(r'\t', line)
            tags = tags.split(',')
            tags = [unidecode(v.rstrip()) for v in tags]
            tags = convert_to_multi_hot_vectors(tags, flickr_vocab)
            labels.append(tags)
            line = file.readline()
    np.save(config.LABELS_PATH, np.array(labels))


if not os.path.exists(config.TAGS_PATH):
    convert_caption_to_tags()

if not os.path.exists(config.LABELS_PATH):
    generate_labels()
