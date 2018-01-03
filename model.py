import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

import config

features = np.load(config.FEATURES_PATH, mmap_mode='r')['features']
labels = np.load(config.LABELS_PATH, mmap_mode='r')

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

n_inputs = X_train.shape[1]
n_outputs = y_train.shape[1]

model = Sequential()
model.add(Dense(2048, activation='elu', kernel_initializer='he_normal', input_dim=n_inputs))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='elu', kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(n_outputs, activation='sigmoid'))

model.summary()

adam = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64, epochs=5, shuffle=True, validation_split=0.2)
model.save('models/model.h5')

scores = model.evaluate(X_test, y_test, batch_size=64, verbose=0)
print("\n", 'Test loss:', scores[0])
print('Test accuracy:', scores[1])
