import os
import glob

import numpy as np

from keras.layers import Input
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array

import config
from helpers import load_variables


class Inference:
    def __init__(self, model_path, num_tags=10):
        self.model_path = model_path
        self.num_tags = num_tags
        self.vocab = self.__vocab()
        self.vgg16 = self.__vgg16()

    def predict(self, image):
        model = load_model(self.model_path)
        image = self.__encode_image(image)
        out = model.predict_proba(image)
        out = np.array(out)
        pred = out[0]
        sorted_pred = np.argpartition(-pred, self.num_tags)
        indices = sorted_pred[:self.num_tags]
        tags = self.vocab[indices]
        return tags

    def __encode_image(self, image):
        return self.vgg16.predict(self.__process_image(image))

    @staticmethod
    def __vocab():
        return np.array(load_variables(config.VOCABULARY_FILE)['words'])

    @staticmethod
    def __vgg16():
        input_tensor = Input(shape=(224, 224, 3))
        model_base = VGG16(weights='imagenet', input_tensor=input_tensor, pooling='max')
        model = Model(inputs=model_base.input, outputs=model_base.get_layer('fc2').output)
        return model

    @staticmethod
    def __process_image(image):
        image = load_img(image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        return image


if __name__ == "__main__":
    trained_model = 'models/model.h5'
    inference = Inference(trained_model)
    images = glob.glob(os.path.join(config.TEST_IMAGES_DIR, '*.jpg'))
    for image in images:
        tags = inference.predict(image)
        print(os.path.basename(image))
        print(' '.join(tags), "\n")
