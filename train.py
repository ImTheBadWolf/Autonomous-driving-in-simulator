import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Dropout, Dense, Flatten
from tensorflow.keras.layers import Conv2D

from utils import generate_batch, load_config
import os

config = load_config("config.json")
INPUT_SHAPE = (config["imageHeight"], config["imageWidth"], 3)

def load_data():
    """Load data for training."""
    dataframe = pd.read_csv(os.path.join(os.getcwd(), config["learningDataDir"], 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    input_data = dataframe[['center', 'left', 'right']].values
    output_data = dataframe['steering'].values
    input_data_train, input_data_valid, output_data_train, output_data_valid = train_test_split(input_data, output_data, test_size=config["testFraction"], random_state=0)

    return input_data_train, input_data_valid, output_data_train, output_data_valid


def build_model():
    """
    Builds the ML model.

    NVIDIA model used as baseline
    """

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(config["keepProbability"]))
    model.add(Flatten())
    model.add(Dense(120, activation='elu'))
    model.add(Dense(60, activation='elu'))
    model.add(Dense(30, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, input_data_train, input_data_valid, output_data_train, output_data_valid):
    """Train the model.

    Train model based on input and output data.
    @param input_data_train Input data for training
    @param input_data_valid Input data for validation
    @param output_data_train Output data for training
    @param output_data_valid Output data for validation
    """
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 mode='auto')
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=config["learningRate"]))
    model.fit(x=generate_batch(config["learningDataDir"], input_data_train, output_data_train, config["batch"], True),
                        epochs=config["epochs"],
                        steps_per_epoch=config["samples"],
                        max_queue_size=1,
                        validation_data=generate_batch(config["learningDataDir"], input_data_valid, output_data_valid, config["batch"], False),
                        validation_steps=len(input_data_valid),
                        callbacks=[checkpoint],
                        verbose=1)

if __name__ == '__main__':
    data = load_data()
    model = build_model()
    train_model(model, *data)