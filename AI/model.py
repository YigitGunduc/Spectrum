import os
import string
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

vocab = sorted(set(string.printable))
char_to_ind = {u: i for i, u in enumerate(vocab)}             
ind_to_char = np.array(vocab)


class Generator(object):

    def __init__(self, rnn_neurons=256, embed_dim=64, dropout=0.3, num_layers=2, learning_rate=1e-4):
        self.model = None
        self.vocab = sorted(set(string.printable))
        self.vocab_size = len(self.vocab)
        self.hparams = {'rnn_neurons' : rnn_neurons, 
                        'embed_dim' : embed_dim,
                        'learning_rate' : learning_rate,
                        'dropout' : dropout,
                        'num_layers' : num_layers}

    def _createModel(self, batch_size):
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.hparams['embed_dim'],batch_input_shape=[batch_size, None]))
        for _ in range(self.hparams['num_layers']):
            model.add(GRU(self.hparams['rnn_neurons'] ,return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform', dropout=self.hparams['dropout']))
        model.add(Dense(self.vocab_size))
        opt = tf.keras.optimizers.Adam(learning_rate=self.hparams['learning_rate'])
        model.compile(optimizer=opt, loss=self._sparse_cat_loss)

        self.model = model

    def _sparse_cat_loss(self, y_true, y_pred):
        return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

    def load_weights(self, weight_file_path):
        '''
        
        Constructs the model and loads the weights 
        Parameters:
                weight_file_path (str): Path to weights location 
        Returns:
                None
        '''
        if os.path.exists(weight_file_path):
            self._createModel(batch_size = 1)
            self.model.load_weights(weight_file_path)
            self.model.build(tf.TensorShape([1, None]))
        else:
            raise FileNotFoundError

    def train(self, data, epochs=1, verbose=1, save_at=5, cuda=False):
        '''

        Trains the model for a given number of epochs
        Parameters:
                epochs (int) : number of epochs to train on
                verbose (bool) : to print loss and epoch number of not to
                save_at (int) : to save at ever n th epoch
        Returns:
                None
        '''
        self._createModel(batch_size = 128)
        
        if cuda:
            device_name = tf.test.gpu_device_name()
            if device_name != '/device:GPU:0':
                raise SystemError('GPU device not found')
            print('Found GPU at: {}'.format(device_name))
            
            with tf.device('/device:GPU:0'):
                for epoch in range(1, epochs + 1):
                    print(f'Epoch {epoch}/{epochs}')
                    self.model.fit(data, epochs=1, verbose=verbose)

                    rnnNeurons=self.hparams['rnn_neurons']
                    if (epoch + 1) % save_at == 0:
                        self.model.save(f'model-{epoch}-epochs-{rnnNeurons}-neurons.h5')

        else:
            for epoch in range(1, epochs + 1):
                print(f'Epoch {epoch}/{epochs}')
                self.model.fit(data, epochs=1, verbose=verbose)

                rnnNeurons=self.hparams['rnn_neurons']
                if (epoch + 1) % save_at == 0:
                    self.model.save(f'model-{epoch}-epochs-{rnnNeurons}-neurons.h5')

    def predict(self, start_seed, gen_size=100, temp=random.uniform(0, 1)):
        '''

        Generates further texts according to the seed text
        Parameters:
                start_seed (str) : seed that model will use to generate further texts
                gen_size (int) : number of characters to generate 700 - 1000 are the most ideal ones
        Returns:
                None
        '''
        if self.model is None:
            raise ValueError('Model Object cannot be NoneType')

        self.model.save_weights('model_weights.h5')
        self.load_weights('model_weights.h5')
        os.remove('model_weights.h5')

        num_generate = gen_size
        input_eval = [char_to_ind[s] for s in start_seed]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        temperature = temp
        self.model.reset_states()

        for _ in range(num_generate):
            predictions = self.model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(ind_to_char[predicted_id])
        return (start_seed + ''.join(text_generated))
    
    def hyperparams(self):
        print('Hyper Parameters')
        print('+--------------------------+')
        for key, value in self.hparams.items():
            print("|{: <13} | {: >10}|".format(key, value))
        print('+--------------------------+')

    def summary(self):
        self.model.summary()

    @property
    def __doc__(self):
        return '''

            Generator object can construct the model,
            save the weights, load the weights train the model,
            and make predictions

            ---------------------------------------------------

            Trainging example :
            model = Generator() # creating an instance of model
            model.train(dataset, epochs = 5, verbose=1, save_at=1) # training the model
            ----------------------------------------------------

            Continue training from a saved weights file :
            model = Generator() # creating an instance of model

            model.load_weights('model-3-epochs.h5', mode = 'training') # loading the weights
            model.train(dataset, epochs = 5, verbose=1, save_at=1) # training the model
            -----------------------------------------------------

            Preditction example :
            model = Generator() # creating an instance of model
            model.load_weights('model-10-epochs.h5') # loading the weights

            print(model.predict('hello')) # making prediction and printing
            -----------------------------------------------------
            '''
