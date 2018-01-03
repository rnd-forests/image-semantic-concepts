import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

CAPTIONS_PATH = os.path.join(ROOT_PATH, 'datasets/results_20130124.token')
KEYWORDS_PATH = os.path.join(ROOT_PATH, 'datasets/keywords.txt')
DICTIONARY_PATH = os.path.join(ROOT_PATH, 'datasets/dictionary.pkl')

FEATURES_PATH = os.path.join(ROOT_PATH, 'datasets/features.npz')
LABELS_PATH = os.path.join(ROOT_PATH, 'datasets/labels.npy')

TEST_IMAGES_DIR = os.path.join(ROOT_PATH, 'images')

VOCAB_SIZE = 1000

NUM_TERMS = 5
POS_TAGS = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBG', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}
