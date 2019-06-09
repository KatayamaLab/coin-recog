# coin-recog

## Requirements
```
$ sudo apt-get install portaudio19-dev  # for linux (Debian)
$ brew install portaudio # for MacOS
$ pip install -r requirements.txt
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