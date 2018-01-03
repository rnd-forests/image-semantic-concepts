import os
import glob

import numpy as np

from keras.layers import Input
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array

import config
from helpers import load_variables

vocab = np.array(load_variables(config.DICTIONARY_PATH)['words'])


def encode_image_vgg16(image):
    image = load_img(image, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    input_tensor = Input(shape=(224, 224, 3))
    vgg_model_base = VGG16(weights='imagenet', input_tensor=input_tensor, pooling='max')
    vgg_model = Model(inputs=vgg_model_base.input, outputs=vgg_model_base.get_layer('fc2').output)
    image_features = vgg_model.predict(image)

    return image_features


def predict(image, num_tags=config.NUM_TERMS, verbose=True):
    model = load_model('models/model.h5')
    image_features = encode_image_vgg16(image)
    out = model.predict_proba(image_features)
    out = np.array(out)
    pred = out[0]
    sorted_pred = np.argpartition(-pred, num_tags)
    indices = sorted_pred[:num_tags]
    keywords = vocab[indices]
    if verbose:
        print(keywords)
    return keywords


if __name__ == "__main__":
    print("Generating predictions...")
    images = glob.glob(os.path.join(config.TEST_IMAGES_DIR, '*.jpg'))
    for image in images:
        print(image)
        predict(image)
