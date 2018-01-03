import os
import pickle


def save_variables(file, data, overwrite=False):
    if os.path.exists(file) and overwrite is False:
        raise RuntimeError('{:s} existed and overwriting is not allowed.'.format(file))
    with open(file, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)


def load_variables(file):
    if os.path.exists(file):
        with open(file, 'rb') as fp:
            data = pickle.load(fp)
        return data
    else:
        raise RuntimeError('{:s} does not exist.'.format(file))
