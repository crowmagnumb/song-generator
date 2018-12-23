# song-generator

## Install

Because tensorflow has no compatible version with pip for python3.7 I had to do the following as found [here](https://github.com/tensorflow/tensorflow/issues/20444)

```
pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
```

But I at first got ...

```
Could not install packages due to an EnvironmentError: [Errno 13] Permission denied: '/usr/local/bin/tensorboard'
Consider using the `--user` option or check the permissions.
```

... and so I had to run `sudo chown ken:admin /usr/local/bin/*` and then rerun the tensorflow install above.

## Training

    python3 rnn_train.py -g "train-data/mountain-goats/*.txt"

## Running

    python3 rnn_play.py
