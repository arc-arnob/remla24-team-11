import pickle
import logging
from pathlib import Path
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def build_model(voc_size, sequence_length, categories):
    model = Sequential()
    model.add(Embedding(voc_size + 1, 50))
    model.add(Conv1D(128, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 7, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(len(categories)-1, activation='sigmoid'))

    return model

def train_model(model, x_train, y_train, x_val, y_val, batch_size, epochs, loss_function, optimizer):
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     shuffle=True,
                     validation_data=(x_val, y_val))
    return hist

def save_model(model, filepath):
    model.save(filepath)

if __name__ == '__main__':
    # Dependency
    with open('models/char_index.pkl', 'rb') as f:
        char_index = pickle.load(f)
    params = {'loss_function': 'binary_crossentropy',
              'optimizer': 'adam',
              'sequence_length': 200,
              'batch_train': 5000,
              'batch_test': 5000,
              'categories': ['phishing', 'legitimate'],
              'char_index': char_index,
              'epoch': 1,
              'embedding_dimension': 50,
              'dataset_dir': "../data/raw"}

    # Assuming char_index is defined elsewhere
    voc_size = len(params['char_index'].keys())

    model = build_model(voc_size, params['sequence_length'], params['categories'])
    
    # Dependency
    x_train = np.load('data/processed/x_train.npy')
    y_train = np.load('data/processed/y_train.npy')

    x_val = np.load('data/processed/x_val.npy')
    y_val = np.load('data/processed/y_val.npy')

    hist = train_model(model, x_train, y_train, x_val, y_val,
                       params['batch_train'], params['epoch'],
                       params['loss_function'], params['optimizer'])

    # Save the trained model
    model_output_filepath = 'models/model.keras'
    save_model(model, model_output_filepath)

    
