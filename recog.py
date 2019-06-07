import os
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Conv2D, GlobalAveragePooling2D,BatchNormalization, Add
import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt
from random import randint
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc
from scipy.fftpack import fft
import librosa
import librosa.display

import wave
import pyaudio

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-l', '--learn', action='store_true', help='Learn')
    parser.add_argument('-p', '--predict', action='store_true', help='Predict')
    parser.add_argument('-d', '--datadir', type=str, default='data', help='Data dir to learn/predict')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batcn size')

    args = parser.parse_args()

    setting_json = 'setting.json'
    if os.path.exists(setting_json):
        with open(setting_json , 'r') as f:
            options = json.load(f)
    else:
        print(setting_json + ' does not exist')
        exit()

    if args.predict == True:
        predict(args, options)
    else:
        learn(args, options)

def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp

# display wave in heatmap
def show_melsp(melsp):
    librosa.display.specshow(melsp)
    plt.colorbar()
    plt.show()

def show_wave(x):
    plt.plot(x)
    plt.show()

def learn(args, options):
    epochs = args.epochs
    batch_size = args.batch_size
    
    chunk_length = options["chunk_length"]
    chunk_num = options["chunk_num"]
    modes = options['modes']
    X = []
    Y = []
    mode_num = len(modes)
        
    for mode_idx, mode in enumerate(modes):
        files = glob.glob(os.path.join(options["data_dir"], mode, "*.wav"))

        for file in files:
            wr = wave.open(file, "r")
            data = wr.readframes(wr.getnframes())
            wr.close()
            
            x = np.frombuffer(data, dtype="int16") / float((2^15))
            wav_length = len(x)

            for i in range(chunk_num):
                pos = randint(0, wav_length - chunk_length)
                x_ =  x[pos:pos+chunk_length]

                # x_mfcc = mfcc(x[pos:pos+chunk_length], samplerate=options["rate"])
                # X.append(x_mfcc[0])
                # X.append(x[pos:pos+chunk_length])

                # x_ = abs(fft(x[pos:pos+chunk_length]))
                # X.append(x_[0:512])

                melsp = calculate_melsp(x_)
                X.append(melsp)

                # print("wave size:{0}\nmelsp size:{1}".format(x.shape, melsp.shape))
                # show_wave(x)
                # show_melsp(melsp)

                # plt.figure(figsize=(15,3))
                # plt.plot(x)
                # plt.show()

                Y.append(mode_idx)



    X = np.array(X).astype('float32')

    Y = tf.keras.utils.to_categorical(Y, mode_num)

    X = np.reshape(X, (X.shape[0],X.shape[1],X.shape[2],1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)

    def cba(inputs, filters, kernel_size, strides):
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x


    inputs = Input(shape=(X_train.shape[1:]))

    x_1 = cba(inputs, filters=32, kernel_size=(1,8), strides=(1,2))
    x_1 = cba(x_1, filters=32, kernel_size=(8,1), strides=(2,1))
    x_1 = cba(x_1, filters=64, kernel_size=(1,8), strides=(1,2))
    x_1 = cba(x_1, filters=64, kernel_size=(8,1), strides=(2,1))

    x_2 = cba(inputs, filters=32, kernel_size=(1,16), strides=(1,2))
    x_2 = cba(x_2, filters=32, kernel_size=(16,1), strides=(2,1))
    x_2 = cba(x_2, filters=64, kernel_size=(1,16), strides=(1,2))
    x_2 = cba(x_2, filters=64, kernel_size=(16,1), strides=(2,1))

    x_3 = cba(inputs, filters=32, kernel_size=(1,32), strides=(1,2))
    x_3 = cba(x_3, filters=32, kernel_size=(32,1), strides=(2,1))
    x_3 = cba(x_3, filters=64, kernel_size=(1,32), strides=(1,2))
    x_3 = cba(x_3, filters=64, kernel_size=(32,1), strides=(2,1))

    x_4 = cba(inputs, filters=32, kernel_size=(1,64), strides=(1,2))
    x_4 = cba(x_4, filters=32, kernel_size=(64,1), strides=(2,1))
    x_4 = cba(x_4, filters=64, kernel_size=(1,64), strides=(1,2))
    x_4 = cba(x_4, filters=64, kernel_size=(64,1), strides=(2,1))

    x = Add()([x_1, x_2, x_3, x_4])

    x = cba(x, filters=128, kernel_size=(1,16), strides=(1,2))
    x = cba(x, filters=128, kernel_size=(16,1), strides=(2,1))

    x = GlobalAveragePooling2D()(x)
    x = Dense(mode_num)(x)
    x = Activation("softmax")(x)

    model = Model(inputs, x)
    model.summary()

    # Let's train the model using Adam with amsgrad
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.summary()

    log_filepath = "./logs/"
    os.makedirs(log_filepath, exist_ok=True)
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_filepath, 
    histogram_freq=1, write_graph=True, write_images=True)

    # history = model.fit(X, Y, batch_size=batch_size, epochs=epochs, 
    #     validation_split=0.1, shuffle=True, callbacks=[tb_cb])
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, 
        validation_data=(X_test, Y_test), callbacks=[tb_cb])

    os.makedirs(options["model_dir"], exist_ok=True)
    model.save(os.path.join(options["model_dir"], 'save.h5'))


if __name__ is "__main__":
    main()