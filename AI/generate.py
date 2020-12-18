import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from   tensorflow.keras.models import Sequential
from   tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU
from   tensorflow.keras.losses import sparse_categorical_crossentropy
from   tensorflow.keras.models import load_model


# constants variables
vocab = sorted(set(string.printable))
vocab_size = len(vocab)
embed_dim = 64
rnn_neurons = 1026
# constants variables

char_to_ind = {u:i for i, u in enumerate(vocab)}
ind_to_char = np.array(vocab)


def sparse_cat_loss(y_true,y_pred):
  return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim,batch_input_shape=[batch_size, None]))
    model.add(GRU(rnn_neurons,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))
    # Final Dense Layer to Predict
    model.add(Dense(vocab_size))
    model.compile(optimizer='adam', loss=sparse_cat_loss) 
    return model

model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)

model.load_weights('shakespeare_gen.h5')

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

import time as t
t1 = t.time()
print('--------------------------------------------------')
print(generate_text(model,"We don't like to do too much explaining",gen_size=1000))
t2 = t.time()
print('--------------------------------------------------')
print(t2 - t1)
print('--------------------------------------------------')
print('done')