# -*- coding: utf-8 -*-
import numpy as np
import nltk

# Open file and copy content of file
f = open('small-corpus.txt', encoding = 'utf-8')
training_data = f.read()
f.close()

# Cleaning
import re
cleaned_training_data = re.sub('[^A-Za-z]+', ' ', training_data)
cleaned_training_data = cleaned_training_data.lower()

# Tokenize training data
tokens = nltk.word_tokenize(cleaned_training_data)
unique_tokens = list(set(tokens))
unique_tokens = sorted(unique_tokens)

# Making word to int and int to word mapping
vocab_size = len(tokens)
word_to_int_dict = dict((word, index) for index, word in enumerate(unique_tokens))
int_to_word_dict = dict((index, word) for index, word in enumerate(unique_tokens))

# Making Training Data
# sequnce_length = 5 means in RNN each LSTM cell depends on 5 previous LSTM cell
tokens = np.array(tokens)
n = len(tokens)
X_train_data = []
y_train_data = []
sequence_length = 5
num_of_words = len(tokens)

for i in range(5, num_of_words):
	input_seq = tokens[i - 5 : i]
	output_seq = tokens[i]
	X_train_data.append([word_to_int_dict[word] for word in input_seq])
	y_train_data.append(word_to_int_dict[output_seq])

# Convert list to numpy arrays because keras accept numpy arrays
X_train = np.array(X_train_data)
y_train = np.array(y_train_data)

# Reshaping dataset because keras accept 3-D dataset
# Third dimension is num of predictors/identifies
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Import necessary classes to build model 
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical

# Transform numbers in proper vectors for using with model.
y_train = to_categorical(y_train)

# Building a 3 stacked LSTM Model
model = Sequential()
model.add(LSTM(512, input_shape = (X_train.shape[1], 1), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(300, activation = 'relu'))
model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
print(model.summary())

# Compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop',
              metrics=['accuracy'])


# Use early stopping on validation loss.
# Model will stop training when validation loss start increasing.
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience = 2, verbose = 1, mode = 'min',
                   restore_best_weights = True)

# Use checkpoint to save model after each epoch
from keras.callbacks import ModelCheckpoint
filepath = "weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose = 1,
                save_best_only = False, save_weights_only = False,
                mode ='min', period = 1)
callbacks_list = [checkpoint, es]

# Fitting Model on training data 
model.fit(X_train, y_train, epochs = 30, batch_size = 32,
          callbacks = callbacks_list)

# Save Model
model.save('text_prediction_model.hdf5')

#------------------------------------------------------------------------------

# Load model
from keras.models import load_model
model = load_model('text_prediction_model.hdf5')

# Summarize model.
print(model.summary())

user_input = input("Please type something of five words\n")

# Tokenize user string
tokens = nltk.word_tokenize(user_input)
tokens = np.array(tokens)
n = len(tokens)
X_test_data = []
sequence_length = 5
num_of_words = len(tokens)

# Generate Test Data
for i in range(sequence_length, num_of_words):
	input_seq = tokens[i - sequence_length : i]
	X_test_data.append([word_to_int_dict[word] for word in input_seq])

X_test = np.array(X_test_data)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Prediction
prediction = model.predict(X_test)

# Convert prediction into human readable string.
word_prediction = []
for i in range(0, len(prediction)):
    if prediction[i] == 1:
        word_prediction.append(int_to_word_dict[i])

print("Predicted Word:- ", word_prediction)

