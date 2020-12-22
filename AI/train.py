import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import load_model
from model import model

# text dataset path
path_to_file = 'raplyrics.txt'

# loading text dataset
text = open(path_to_file, 'r').read()

# removing non ascii characters
text = re.sub(r'[^\x00-\x7f]',r'', text) 

# removing escape characters
text = text.replace('\x10', ' ') 
text = text.replace('\x14', ' ') 
text = text.replace('\x01', ' ') 
text = text.replace('\x1c', ' ') 
text = text.replace('\x13', ' ') 
text = text.replace('\x12', ' ') 
text = text.replace('\x7f', ' ') 
text = text.replace('\x0f', ' ') 
text = text.replace('\x02', ' ') 
text = text.replace('\x0e', ' ') 

# constants variables
vocab = sorted(set(string.printable)) # variety of characters
vocab_size = len(vocab) # num of items in the vocab
embed_dim = 64 # embeding dim size
rnn_neurons = 1026 # number of neurans of out rnn units

batch_size = 128 # batch_size
buffer_size = 10000 # buffer_size
epochs = 30 # num of epochs to train 
seq_len = 120 # seq_len

char_to_ind = {u:i for i, u in enumerate(vocab)}
ind_to_char = np.array(vocab)
encoded_text = np.array([char_to_ind[c] for c in text])

 
total_num_seq = len(text)//(seq_len+1)

char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

sequences = char_dataset.batch(seq_len+1, drop_remainder=True)

def create_seq_targets(seq):
    input_txt = seq[:-1]
    target_txt = seq[1:]
    return input_txt, target_txt


dataset = sequences.map(create_seq_targets)

dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

model = model(
  vocab_size = vocab_size,
  embed_dim=embed_dim,
  batch_size=batch_size)

print(model.summary())

for i in range(epochs):
  print(f'{i}/{epochs}')
  model.fit(dataset,epochs=1)
  if i % 3 ==0:
    model.save(f'model-{i}-epochs.h5') 
    print('model has been saved')

model.save(f'model-{epochs}-epochs.h5') 
