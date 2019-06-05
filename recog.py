import os
import json
import tensorflow as tf
import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt
from random import randint
from sklearn.model_selection import train_test_split

import wave
import pyaudio

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-l', '--learn', action='store_true', help='Learn')
    parser.add_argument('-p', '--predict', action='store_true', help='Predict')
    parser.add_argument('-d', '--datadir', type=str, default='data', help='Data dir to learn/predict')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batcn size')

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
                X.append(x[pos:pos+chunk_length])
                Y.append(mode_idx)

    X = np.array(X).astype('float32')
    # plt.figure(figsize=(15,3))
    # plt.plot(X.T)
    # plt.show()

    Y = tf.keras.utils.to_categorical(Y, mode_num)

    X = np.reshape(X, (X.shape[0],X.shape[1],1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)

    model = tf.keras.models.Sequential()
       
    model.add(tf.keras.layers.Conv1D(64, 8, activation='relu', input_shape=X[0].shape))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(32, 8, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(32, 8, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(mode_num, activation='softmax'))
    #########

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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