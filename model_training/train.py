import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.optimizers import *
from keras.losses import *
from keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
import numpy as np
import tensorflow
import keras
import sklearn
import os
import sys
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

if len(sys.argv) != 1:
    sys.exit(1)

for i in range(0, 4):
    print('*' * 80)

"""
Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
"""

DATASET = 'RAVDESS_6'
OPT = ''
MODE = OPT != '' and DATASET + '_' + OPT or DATASET

MODEL_SAVE_PATH = 'models/model_' + MODE + '.ckpt'

# EMOTIONS = ['calm', 'calm', 'happy', 'sad',
# 'angry', 'fearful', 'disgust', 'surprised']
# OBSERVED_EMOTIONS = ['female_happy', 'female_sad', 'female_angry', 'female_fearful', 'female_calm', 'female_disgust',
# 'female_surprised', 'male_happy', 'male_sad', 'male_angry', 'male_fearful', 'male_surprised', 'male_calm', 'male_disgust']
OBSERVED_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust']
TEST_SIZE = 0.2


x = np.load('np_arrays/' + DATASET + '/x_' + DATASET + '.npy')
y = np.load('np_arrays/' + DATASET + '/y_' + DATASET + '.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=TEST_SIZE)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


if not os.path.exists('np_arrays/' + MODE):
    os.mkdir('np_arrays/' + MODE)
np.save('np_arrays/' + MODE + '/x_train_' + MODE + '.npy', x_train)
np.save('np_arrays/' + MODE + '/x_test_' + MODE + '.npy', x_test)
np.save('np_arrays/' + MODE + '/y_train_' + MODE + '.npy', y_train)
np.save('np_arrays/' + MODE + '/y_test_' + MODE + '.npy', y_test)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)


#################################################
##          OG CNN model (64.11% on RCT)        #
#################################################

# input
inputs = Input(shape=(x_train.shape[1], 1))

# First Block
model = Conv1D(256, 8, padding='same')(inputs)
model = BatchNormalization()(model)
model = Activation('relu')(model)
model = MaxPooling1D(pool_size=(2))(model)
model = Dropout(0.25)(model)

# Second Block
model = Conv1D(256, 8, padding='same')(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)
model = MaxPooling1D(pool_size=(2))(model)
model = Dropout(0.25)(model)

# Third Block
model = Conv1D(128, 8, padding='same')(model)
model = Activation('relu')(model)

# Fourth Block
model = Conv1D(128, 8, padding='same')(model)
model = Activation('relu')(model)

# Fifth Block
model = Conv1D(128, 8, padding='same')(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)
model = MaxPooling1D(pool_size=(4))(model)
model = Dropout(0.25)(model)
model = MaxPooling1D(pool_size=(4))(model)
model = Dropout(0.25)(model)

# Sixth Block
model = Conv1D(64, 8, padding='same')(model)
model = Activation('relu')(model)
model = Dropout(0.25)(model)

model = Conv1D(64, 8, padding='same')(model)
model = Activation('relu')(model)

model = Flatten()(model)
#model = Reshape((1, model.shape[1]))(model)

#model = LSTM(64, return_sequences=True)(model)
#model = LSTM(64, return_sequences=True)(model)

model = Dense(32)(model)
model = Activation('relu')(model)
model = Dense(16)(model)
model = Activation('relu')(model)
model = Dense(len(OBSERVED_EMOTIONS))(model)
model = Activation('softmax')(model)


#################################################
##          paper model (65.8% on RCT)          #
#################################################

#inputs = Input(shape=(x_train.shape[1], 1))

#model = Conv1D(64, 8, padding='same')(inputs)
#model = BatchNormalization()(model)
#model = Activation('relu')(model)
#model = Conv1D(64, 8, padding='same')(model)
#model = BatchNormalization()(model)
#model = Activation('relu')(model)

#model = MaxPooling1D(pool_size=(2))(model)
#model = Dropout(0.3)(model)

#model = Conv1D(128, 8, padding='same')(model)
#model = BatchNormalization()(model)
#model = Activation('relu')(model)
#model = Conv1D(128, 8, padding='same')(model)
#model = BatchNormalization()(model)
#model = Activation('relu')(model)

#model = MaxPooling1D(pool_size=(2))(model)
#model = Dropout(0.3)(model)

#model = Conv1D(256, 8, padding='same')(model)
#model = BatchNormalization()(model)
#model = Activation('relu')(model)
#model = Conv1D(256, 8, padding='same')(model)
#model = BatchNormalization()(model)
#model = Activation('relu')(model)
#model = Conv1D(256, 8, padding='same')(model)
#model = BatchNormalization()(model)
#model = Activation('relu')(model)
#model = Conv1D(256, 8, padding='same')(model)
#model = BatchNormalization()(model)
#model = Activation('relu')(model)
#model = Conv1D(256, 8, padding='same')(model)
#model = BatchNormalization()(model)
#model = Activation('relu')(model)

#model = Reshape((model.shape[1], model.shape[2], 1))(model)

#model = AveragePooling2D(pool_size=(2, 2))(model)

#model = Flatten()(model)


#model = Dense(32)(model)
#model = Activation('relu')(model)
#model = Dense(16)(model)
#model = Activation('relu')(model)
#model = Dense(len(OBSERVED_EMOTIONS))(model)
#model = Activation('softmax')(model)

LEARNING_RATE = 0.01
BATCH_SIZE = 16
EPOCHS = 500

loss = SparseCategoricalCrossentropy()
#optimizer = Adam(learning_rate=LEARNING_RATE)
#optimizer = RMSprop(learning_rate=LEARNING_RATE)

model = Model(inputs=inputs, outputs=model)

if not os.path.exists('models/plots_model'):
    os.makedirs('models/plots_model')
plot_model(model, to_file='models/plots_model/' + MODE + '_model.png',
           show_shapes=True, show_layer_names=True)
model.summary()
model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50)
callbacks = [checkpoint, early_stopping]

history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=callbacks
)

if not os.path.exists('figures/' + MODE):
    os.makedirs('figures/' + MODE)
plt.plot(history.epoch, history.history['accuracy'])
plt.plot(history.epoch, history.history['val_accuracy'])
plt.title(MODE + ' model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('figures/' + MODE + '/' + MODE + '_accuracy.png')
plt.show()

plt.plot(history.epoch, history.history['loss'])
plt.plot(history.epoch, history.history['val_loss'])
plt.title(MODE + ' model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('figures/' + MODE + '/' + MODE + '_loss.png')
plt.show()

model = load_model(MODEL_SAVE_PATH)
y_predict = np.argmax(model.predict(x_test), axis=1)
sklearn.metrics.accuracy_score(y_test, y_predict)
