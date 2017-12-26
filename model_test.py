import config
import numpy as np

from keras.layers import Input
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array

model = load_model('models/my_model.h5')
image = load_img('test_image.jpg', target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)

input_tensor = Input(shape=(224, 224, 3))
vgg_model_base = VGG16(weights='imagenet', input_tensor=input_tensor, pooling='max')
vgg_model = Model(inputs=vgg_model_base.input, outputs=vgg_model_base.get_layer('fc2').output)
image_feature = vgg_model.predict(image)

out = model.predict_proba(image_feature)
out = np.array(out)

dictionary = np.load(config.DICTIONARY_PATH)

pred = out[0]
sorted_pred = np.argpartition(-pred, 5)
indices = sorted_pred[:5]
keywords = dictionary[indices]
print(keywords)
