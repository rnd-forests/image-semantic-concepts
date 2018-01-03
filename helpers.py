import os
import pickle

def save_variables(pickle_file_name, var, overwrite = False):
    if os.path.exists(pickle_file_name) and overwrite == False:
        raise Exception('{:s} exists and over write is false.'.format(pickle_file_name))
    with open(pickle_file_name, 'wb') as f:
        pickle.dump(var, f, pickle.HIGHEST_PROTOCOL)

def load_variables(pickle_file_name):
    if os.path.exists(pickle_file_name):
        with open(pickle_file_name, 'rb') as f:
            d = pickle.load(f)
        return d
    else:
        raise Exception('{:s} does not exists.'.format(pickle_file_name))
