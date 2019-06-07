import sys
import pyaudio
import wave
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# read optios
setting_json = 'setting.json'
if os.path.exists(setting_json):
    with open(setting_json , 'r') as f:
        options = json.load(f)
else:
    print(setting_json + ' does not exist')
    exit()

channels = options["channels"]
rate = options["rate"]
chunk = options["chunk"]
seconds = options["seconds"]
byte_format=pyaudio.paInt16


mode = input("Enter mode: ")
if mode=="":
    mode = "test"
os.makedirs(os.path.join(options["data_dir"], mode), exist_ok=True)

file_idx = 0
try:
    while 1:
        print("*** waiting trigger")

        audio = pyaudio.PyAudio()
        stream = audio.open(format=byte_format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

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

        for i in range(0, int(rate / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)
        frames = b''.join(frames)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        x = np.frombuffer(frames, dtype="int16")

        print("*** recoding done")

        # plt.figure(figsize=(15,3))
        # plt.plot(x)
        # plt.show()

        filename = "{:03}.wav".format(file_idx)

        wf = wave.open(os.path.join(options["data_dir"], mode, filename), 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(byte_format))
        wf.setframerate(rate)
        wf.writeframes(frames)
        wf.close()

        print("*** saved to "+filename)
        print()


        file_idx += 1
except KeyboardInterrupt:
    print("*** done")
