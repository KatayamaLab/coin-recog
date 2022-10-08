# coin-recog

## Requirements
```
$ sudo apt-get install portaudio19-dev  # for linux (Debian)
$ brew install portaudio # for MacOS
$ pip install -r requirements.txt
```

for M1 mac

```
brew list libsndfile  
mkdir -p path-to-python/site-packages/_soundfile_data/ 
cp /opt/homebrew/Cellar/libsndfile/1.1.0_1/lib/libsndfile.dylib path-to-python/site-packages/_soundfile_data/ 
```

```
brew install portaudio --HEAD
vim ~/.pydistutils.cfg
```

Write following text to `.pydistutils.conf`

```
[build_ext]
include_dirs=/opt/homebrew/opt/portaudio/include/
library_dirs=/opt/homebrew/opt/portaudio/lib/
```

```
pip intall pyaudio
```


## How to use

Recode sound of each coin
```
$ python recog.py -r
```
Threshold to start recording is asked every time. The default value is 5000.
`-o` option can be used to recode each coin.

Train the NN model and save trained model
```
$ python recog.py -l
```
`--epoch` option specifies the training steps. 

Predict sound of coin.
```
$ python recog.py -p
```


## Reference
- https://qiita.com/cvusk/items/61cdbce80785eaf28349