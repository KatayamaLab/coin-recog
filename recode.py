import pyaudio
import wave
import os
import json
import numpy as np
import matplotlib.pyplot as plt

setting_json = 'setting.json'
if os.path.exists(setting_json):
    with open(setting_json , 'r') as f:
        options = json.load(f)
else:
    print(setting_json + ' does not exist')
    exit()

CHUNK = 1024
FORMAT = pyaudio.paInt16 # int16型
CHANNELS = 1             # ステレオ
RATE = 48000             # 441.kHz
RECORD_SECONDS = 5       # 5秒録音
WAVE_OUTPUT_FILENAME = "output.wav"

data_dir = "wav_data"

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("*** waiting trigger")

frames = []

trigger = False
while trigger == False:
    data = stream.read(CHUNK)
    a = np.frombuffer(data, dtype="int16")
    if np.any(a>5000):
        print("*** triggered")
        trigger = True

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
frames = b''.join(frames)

x = np.frombuffer(frames, dtype="int16")

print("*** recoding done")

plt.figure(figsize=(15,3))
plt.plot(x)
plt.show()

stream.stop_stream()
stream.close()
audio.terminate()

filename = input("Enter file name: ")
if not filename == '':
    os.makedirs(data_dir, exist_ok=True)

    wf = wave.open(os.path.join(options["data_dir"], filename), 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(frames)
    wf.close()
    