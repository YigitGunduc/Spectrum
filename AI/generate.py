import string
import time as t
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU
from model import model

# constants variables
vocab = sorted(set(string.printable))
vocab_size = len(vocab)
embed_dim = 64
rnn_neurons = 1026

char_to_ind = {u:i for i, u in enumerate(vocab)}
ind_to_char = np.array(vocab)

model = model(vocab_size, embed_dim, batch_size=1)

model.load_weights('model-21-epochs.h5')

model.build(tf.TensorShape([1, None]))

model.summary()

def generate_text(model, start_seed,gen_size=100,temp=1.0):
  num_generate = gen_size
  input_eval = [char_to_ind[s] for s in start_seed]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
  temperature = temp
  model.reset_states()

  for i in range(num_generate):
      predictions = model(input_eval)
      predictions = tf.squeeze(predictions, 0)
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(ind_to_char[predicted_id])
  return (start_seed + ''.join(text_generated))

t1 = t.time()
print(generate_text(model,"We don't like to do too much explaining",gen_size=1000))
t2 = t.time()
print(f'time taken {t2 - t1} (ms)')