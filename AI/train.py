import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from   tensorflow.keras.models import Sequential
from   tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU
from   tensorflow.keras.losses import sparse_categorical_crossentropy
from   tensorflow.keras.models import load_model


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
epochs = 5 # num of epochs to train 
seq_len = 120 # seq_len

char_to_ind = {u:i for i, u in enumerate(vocab)}
ind_to_char = np.array(vocab)
encoded_text = np.array([char_to_ind[c] for c in text])


total_num_seq = len(text)//(seq_len+1)

char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

for i in char_dataset.take(500):
     print(ind_to_char[i.numpy()])

sequences = char_dataset.batch(seq_len+1, drop_remainder=True)

def create_seq_targets(seq):
    input_txt = seq[:-1]
    target_txt = seq[1:]
    return input_txt, target_txt


dataset = sequences.map(create_seq_targets)


for input_txt, target_txt in  dataset.take(1):
    print(input_txt.numpy())
    print(''.join(ind_to_char[input_txt.numpy()]))
    print('\n')
    print(target_txt.numpy())
    print(''.join(ind_to_char[target_txt.numpy()]))

dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)


def sparse_cat_loss(y_true,y_pred):
  return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim,batch_input_shape=[batch_size, None]))
    model.add(GRU(512,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))
    model.add(GRU(256,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))
    model.add(GRU(64,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))
    # Final Dense Layer to Predict
    model.add(Dense(vocab_size))
    model.compile(optimizer='adam', loss=sparse_cat_loss) 
    return model

model = create_model(
  vocab_size = vocab_size,
  embed_dim=embed_dim,
  rnn_neurons=rnn_neurons,
  batch_size=batch_size)


model.summary()

for i in range(epochs):
  model.fit(dataset,epochs=1)
  if i % 10 == 0:
    model.save(f'model-{i}epochs.h5')
model.save('shakespeare_gen.h5') 

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

print(generate_text(model,"We don't like to do too much explaining",gen_size=1000))
