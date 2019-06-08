import os
import json
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Conv2D, GlobalAveragePooling2D,BatchNormalization, Add
import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt
from random import randint
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import wave
import pyaudio

import datetime as dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learn', action='store_true', help='Learn')
    parser.add_argument('-r', '--recode', action='store_true', help='Recode')
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
    elif args.recode == True:
        recode(args, options)
    elif args.learn == True:
        learn(args, options)


def openAudio(channels, rate, frames_per_buffer):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=frames_per_buffer)
    return audio, stream


def closeAudio(audio, stream):
    stream.stop_stream()
    stream.close()
    audio.terminate()


def recode(args, options):
    channels = options["channels"]
    rate = options["rate"]
    chunk = options["chunk"]
    seconds = options["seconds"]

    mode = input("Enter mode: ")
    if mode=="":
        mode = "test"
    os.makedirs(os.path.join(options["data_dir"], mode), exist_ok=True)

    file_idx = 0
    try:
        while 1:
            print("*** waiting trigger")

            audio, stream = openAudio(channels, rate, chunk)

            frames = []
            trigger = False

            while 1:
                data = stream.read(chunk)
                np_data = np.frombuffer(data, dtype="int16")
                level = int(np_data.max()/(2**8))
                bar = "="*level + " "*(100-level)
                th = int(options["threshold"]/(2**8))
                bar = bar[:th] + "|" + bar[1+th:]
                print("\r" + bar, end="")
                if np.any(np_data>options["threshold"]):
                    frames.append(data)
                    break
                
            print()
            print("*** triggered")

            for i in range(1, int(rate / chunk * seconds)):
                data = stream.read(chunk)
                frames.append(data)
            frames = b''.join(frames)

            closeAudio(audio, stream)

            x = np.frombuffer(frames, dtype="int16")

            print("*** recoding done")


            filename = "{:03}.wav".format(file_idx)

            wf = wave.open(os.path.join(options["data_dir"], mode, filename), 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(frames)
            wf.close()

            print("*** saved to "+filename)
            print()


            file_idx += 1
    except KeyboardInterrupt:
        print("*** done")

def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp

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

                melsp = calculate_melsp(x_)
                X.append(melsp)
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
    x = Dropout(0.5)(x)
    x = Activation("softmax")(x)

    model = Model(inputs, x)
    opt = tf.keras.optimizers.Adam(lr=0.00001, decay=1e-6, amsgrad=True)
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

def predict(args, options):
    model = load_model(os.path.join(options["model_dir"], 'save.h5'))

    X = []
    Y = []
    mode_num = len(options['modes'])
    
    result_img_path = os.path.join('results', dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(result_img_path, exist_ok=True)

    channels = options["channels"]
    rate = options["rate"]
    chunk = options["chunk"]

    chunk_length = options["chunk_length"]
    chunk_num = options["chunk_num"]
    modes = options['modes']

    while 1:
        frames = []

        audio, stream = openAudio(channels, rate, chunk)

        for i in range(0, int(chunk_length / chunk)):
            data = stream.read(chunk)
            frames.append(data)
        frames = b''.join(frames)

        closeAudio(audio, stream)

        x = np.frombuffer(frames, dtype="int16") / float((2^15))

        melsp = calculate_melsp(x)
        X = np.array([melsp])

        X = np.reshape(X, (X.shape[0],X.shape[1],X.shape[2],1))

        predict = model.predict(X)

        print(options['modes'][np.argmax(predict)], predict)

if __name__ == "__main__":
    main()
