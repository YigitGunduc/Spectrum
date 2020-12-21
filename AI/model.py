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

def sparse_cat_loss(y_true,y_pred):
  return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def model(vocab_size, embed_dim, batch_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim,batch_input_shape=[batch_size, None]))
    model.add(GRU(512,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform',dropout=0.2,))
    model.add(GRU(256,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform',dropout=0.2,))
    model.add(GRU(64,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform',dropout=0.1,))
    # Final Dense Layer to Predict
    model.add(Dense(vocab_size))
    model.compile(optimizer='adam', loss=sparse_cat_loss) 
    return model