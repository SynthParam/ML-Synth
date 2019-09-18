import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import SGD, Adam
from os.path import isfile
from multi_gpu import *
import numpy as np
import h5py


# This is a VGG-style network that I made by 'dumbing down' @keunwoochoi's compact_cnn code
# I have not attempted much optimization, however it *is* fairly understandable
def MyCNN_Keras2(X_shape, nb_classes, nb_layers=4):
    # Inputs:
    #    X_shape = [ # spectrograms per batch, # audio channels, # spectrogram freq bins, # spectrogram time bins ]
    #    nb_classes = number of output n_classes
    #    nb_layers = number of conv-pooling sets in the CNN
    from keras import backend as K
    K.set_image_data_format('channels_last')                   # SHH changed on 3/1/2018 b/c tensorflow prefers channels_last

    nb_filters = 32  # number of convolutional filters = "feature maps"
    kernel_size = (3, 3)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for max pooling
    cl_dropout = 0.5    # conv. layer dropout
    dl_dropout = 0.6    # dense layer dropout

    print(" MyCNN_Keras2: X_shape = ",X_shape,", channels = ",X_shape[3])
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size, padding='same', input_shape=input_shape, name="Input"))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Activation('relu'))        # Leave this relu & BN here.  ELU is not good here (my experience)
    model.add(BatchNormalization(axis=1))

    for layer in range(nb_layers-1):   # add more layers than just the first
        model.add(Conv2D(nb_filters, kernel_size, padding='same'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Activation('elu'))
        model.add(Dropout(cl_dropout))
        #model.add(BatchNormalization(axis=1))  # ELU authors reccommend no BatchNorm. I confirm.

    model.add(Flatten())
    model.add(Dense(128))            # 128 is 'arbitrary' for now
    #model.add(Activation('relu'))   # relu (no BN) works ok here, however ELU works a bit better...
    model.add(Activation('elu'))
    model.add(Dropout(dl_dropout))
    model.add(Dense(nb_classes, activation="linear"))
    return model

def abs_per_diff(y_true, y_pred):
    '''
    Metric to calculate absolute percentage difference
    '''
    diff = y_pred-y_true
    percentDiff = (diff / y_true) * 100
    return np.abs(percentDiff)


def setup_model(X, param_names, nb_layers=4, try_checkpoint=True, weights_file="weights.hdf5", quiet=False, missing_weights_fatal=False):
    '''
    Main routine for setting up the model, taken from panotti repo
    '''

    serial_model = MyCNN_Keras2(X.shape, nb_classes=len(param_names), nb_layers=nb_layers)

    # Initialize weights using checkpoint if it exists.
    if (try_checkpoint):
        print("Looking for previous weights...")
        if ( isfile(weights_file) ):
            print ('Weights file detected. Loading from ',weights_file)
            loaded_model = load_model(weights_file)   # strip any previous parallel part, to be added back in later
            serial_model.set_weights( loaded_model.get_weights() )   # assign weights based on checkpoint
        else:
            if (missing_weights_fatal):
                print("Need weights file to continue.  Aborting")
                assert(not missing_weights_fatal)
            else:
                print('No weights file detected, so starting from scratch.')
    

    opt = 'adadelta' # Adam(lr = 0.00001)  # So far, adadelta seems to work the best of things I've tried
    metrics=[abs_per_diff]
    loss = "mean_absolute_percentage_error"

    serial_model.compile(loss=loss, optimizer=opt, metrics=metrics)

    # Multi-GPU "parallel" capability
    gpu_count = get_available_gpus()
    if (gpu_count >= 2):
        print(" Parallel run on",gpu_count,"GPUs")
        model = make_parallel(serial_model, gpu_count=gpu_count)
        model.compile(loss=loss, optimizer=opt, metrics=metrics)
    else:
        model = serial_model

    if (not quiet):
        print("Summary of serial model (duplicated across",gpu_count,"GPUs):")
        serial_model.summary()  # print out the model layers

    return model, serial_model

def save_model_ext(model, filepath, overwrite=True, class_names=None):
    save_model(model, filepath, overwrite)
    if class_names is not None:
        f = h5py.File(filepath, mode='a')
        f.attrs['class_names'] = np.array(class_names, dtype='S')  # have to encode it somehow
        f.close()



