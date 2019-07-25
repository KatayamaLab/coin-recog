import os
import json
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Conv2D, GlobalAveragePooling2D,BatchNormalization, Add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import wave
import pyaudio
import cv2



def main():
    setting_json = 'setting.json'
    if os.path.exists(setting_json):
        with open(setting_json , 'r') as f:
            options = json.load(f)
    else:
        print(setting_json + ' does not exist')
        exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learn', action='store_true', help='Learn')
    parser.add_argument('-r', '--recode', action='store_true', help='Recode')
    parser.add_argument('-p', '--predict', action='store_true', help='Predict')
    parser.add_argument('-d', '--datadir', type=str, default='data', help='Data dir to learn/predict')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batcn size')
    parser.add_argument('-t', '--threshold', type=int, default=options['threshold'], help='threshold')
    parser.add_argument('-o', '--onebyone', action='store_true', help='eachcoin')
    args = parser.parse_args()

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
    frames_per_buffer = options["frames_per_buffer"]
    seconds = options["seconds"]

    if args.onebyone:
        modes = options["recode_modes"]
    else:
        mode = input("Enter mode [test]: ")
        if mode=="":
            mode = "test"
        modes = [mode]

    file_idx = 0
    try:
        while 1:
            for mode in modes:
                os.makedirs(os.path.join(options["data_dir"], mode), exist_ok=True)

                threshold = input("Input trigger level[{}]: ".format(options["threshold"]))
                if threshold == "":
                    threshold = options["threshold"]
                else:
                    threshold = int(threshold)

                print("*** waiting trigger")

                audio, stream = openAudio(channels, rate, frames_per_buffer)

                frames = []
                trigger = False

                while 1:
                    data = stream.read(frames_per_buffer)
                    np_data = np.frombuffer(data, dtype="int16")
                    level = int(np_data.max()/(2**8))
                    bar = "="*level + " "*(100-level)
                    th = int(threshold/(2**8))
                    bar = bar[:th] + "|" + bar[1+th:]
                    print("\r" + bar, end="")
                    if np.any(np_data > threshold):
                        frames.append(data)
                        break
                    
                print()
                print("*** triggered")

                for i in range(1, int(rate / frames_per_buffer * seconds)):
                    data = stream.read(frames_per_buffer)
                    frames.append(data)
                frames = b''.join(frames)

                closeAudio(audio, stream)

                x = np.frombuffer(frames, dtype="int16")

                print("*** recoding done")

                filepath = os.path.join(options["data_dir"], mode, "{:03}.wav".format(file_idx))

                wf = wave.open(filepath, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(rate)
                wf.writeframes(frames)
                wf.close()

                print("*** saved to "+filepath)
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

def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))

def shift_sound(x, rate=2):
    return np.roll(x, rate)

def learn(args, options):
    epochs = args.epochs
    batch_size = args.batch_size
    
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
            
            x = np.frombuffer(data, dtype="int16") / float((2**15))

            for i in range(chunk_num):
                x_ = add_white_noise(x, rate=np.random.rand()*0.05)
                x_ = shift_sound(x_, np.random.randint(len(x)))
                melsp = calculate_melsp(x_)
                X.append(melsp)
                Y.append(mode_idx)

    X = np.array(X).astype('float32')
    Y = tf.keras.utils.to_categorical(Y, mode_num)
    X = np.reshape(X, (X.shape[0],X.shape[1],X.shape[2],1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=options["test_size"])

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
    # x = Dropout(0.5)(x)
    x = Activation("softmax")(x)

    model = Model(inputs, x)
    opt = tf.keras.optimizers.Adam(lr=0.00001, decay=1e-6, amsgrad=True)
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    model.summary()

    os.makedirs(options["log_dir"], exist_ok=True)
    os.makedirs(options["model_dir"], exist_ok=True)

    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=options["log_dir"], 
        histogram_freq=1, write_graph=True, write_images=True)


    # es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    chkpt = os.path.join(options["model_dir"], 'save.h5')
    cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, 
        validation_data=(X_test, Y_test), callbacks=[tb_cb, cp_cb])

    model.save(chkpt)

def predict(args, options):
    modes = options['modes']
    mode_num = len(modes)
    
    channels = options["channels"]
    rate = options["rate"]
    seconds = options["seconds"]
    frames_per_buffer = options["frames_per_buffer"]

    plt.style.use('dark_background')
    fig = plt.figure()
    ax = fig.add_subplot(212)
    labels = np.arange(mode_num)
    bar_data = ax.bar(labels, np.zeros(len(modes)))
    ax.set_ylim([0, 1])
    ax.set_ylabel('Probability')

    ax_im = fig.add_subplot(211)
    imgfiles = ['1.jpeg', '5.jpeg', '10.jpeg', '50.jpeg', '100.jpeg', '500.jpeg', '0.jpeg']
    img = []
    for imgfile in imgfiles:
        img.append(cv2.cvtColor(cv2.imread('./image/'+imgfile), cv2.COLOR_BGR2RGB))
    img_data = ax_im.imshow(img[6])

    model = load_model(os.path.join(options["model_dir"], 'save.h5'))

    while 1:
        frames = []

        audio, stream = openAudio(channels, rate, frames_per_buffer)

        for i in range(0, int(rate / frames_per_buffer * seconds)):
            data = stream.read(frames_per_buffer)
            frames.append(data)
        frames = b''.join(frames)

        closeAudio(audio, stream)

        x = np.frombuffer(frames, dtype="int16") / float((2**15))

        melsp = calculate_melsp(x)
        X = np.array([melsp])
        X = np.reshape(X, (X.shape[0],X.shape[1],X.shape[2],1))

        predict = model.predict(X)

        print(options['modes'][np.argmax(predict)], predict)

        for i in range(len(predict[0])):
            bar_data[i].set_height(predict[0][i])
        idx = predict.argmax()
        if predict[0][idx]>0.5:
            img_data.set_array(img[idx])
        else:
            img_data.set_array(img[6])

        plt.xticks(labels, modes)
        plt.pause(0.05)

if __name__ == "__main__":
    main()
