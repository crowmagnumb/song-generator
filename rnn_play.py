# encoding: UTF-8
# Copyright 2017 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
import my_txtutils
import sys

# these must match what was saved !
ALPHASIZE = my_txtutils.ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512

output = "generated_output.txt"

# model = "checkpoints/rnn_train_1533066245-0"
# # model = "checkpoints/rnn_train_1545494331-0"
# model = "checkpoints/rnn_train_1545498688-2580000"
model = sys.argv[1]

# model = "checkpoints_shakespeare/rnn_train_1495455686-0"
# model = "checkpoints_shakespeare/rnn_train_1495440473-102000000"
# author = "checkpoints_shakespeare/rnn_train_1495440473-102000000"
author = model

# Orignally set for shakespeare
# num = 10000

# For mountain goats I calculated the mean and std. dev. of characters per song.
# So randomly generating the length based on gaussian distro
# Adding half the mean line length also because I'm going to strip off the list line that is generated
# because it abruptly ends mid-word and thus is unintelligable. And I figure it will probably be on avg
# half-way to a full line before being truncated.
stdev = 266
mean = 756
meanLineLength = 29
num = stdev * int(np.random.randn(1, 1)[0][0]) + mean + int(meanLineLength / 2)

startchar = "\n"

with tf.Session() as sess:
    graph = tf.train.import_meta_graph(model + '.meta')
    graph.restore(sess, author)

    x = my_txtutils.convert_from_alphabet(ord(startchar))
    x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

    # initial values
    y = x
    h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
    chars = []
    for i in range(num):
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

        # If sampling is be done from the topn most likely characters, the generated text
        # is more credible and more "english". If topn is not set, it defaults to the full
        # distribution (ALPHASIZE)

        # Recommended: topn = 10 for intermediate checkpoints, topn=2 or 3 for fully trained checkpoints

        c = my_txtutils.sample_from_probabilities(yo, topn=2)
        y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        c = chr(my_txtutils.convert_to_alphabet(c))
        chars.append(c)

    text = "".join(chars)
    text = text[:text.rindex("\n")]  # Strips off last unfinished line
    print(text)

    file = open(output, "w")
    file.write(text)
    file.close()
