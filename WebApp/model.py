import os
import string
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import load_model


vocab = sorted(set(string.printable)) # variety of characters
vocab_size = len(vocab) # num of items in the vocab          |
embed_dim = 64 # embeding dim size                           
rnn_neurons = 1026 # number of neurans of out rnn units      
                                                             
batch_size = 128 # batch_size                                
buffer_size = 10000 # buffer_size                            
epochs = 30 # num of epochs to train                         
seq_len = 120 # seq_len                                      
                                                 
char_to_ind = {u:i for i, u in enumerate(vocab)}             
ind_to_char = np.array(vocab)
encoded_text = np.array([char_to_ind[c] for c in text])

class Generator():

    def __init__(self):
        self.model = None
        self.vocab = sorted(set(string.printable))
        self.vocab_size = len(self.vocab)
        self.embed_dim = 64

    def _createModel(self, batch_size):
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.embed_dim,batch_input_shape=[batch_size, None]))
        model.add(GRU(256,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform',dropout=0.1))
        model.add(Dense(self.vocab_size))
        model.compile(optimizer='adam', loss=self._sparse_cat_loss) 
        
        self.model = model

    def _sparse_cat_loss(self, y_true, y_pred):
        return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

    def load_weights(self, weight_file_path,mode = 'predictions'):
        '''
        
        Constructs the model and loads the weights 

        Parameters:
                weight_file_path (str): Path to weights location 

        Returns:
                None
        '''
        if os.path.exists(weight_file_path):
            if mode == 'predictions':
                self._createModel(batch_size = 1)
                self.model.load_weights(weight_file_path)
                self.model.build(tf.TensorShape([1, None]))
            elif mode == 'training':
                self._createModel(batch_size = 128)
                self.model.load_weights(weight_file_path)
                self.model.build(tf.TensorShape([1, None]))
    
    def train(self, data, epochs=1, verbose=1, save_at=5):
        '''
        
        Trains the model for a given number of epochs 

        Parameters:
                epochs (int) : number of epochs to train on
                verbose (bool) : to print loss and epoch number of not to
                save_at (int) : to save at ever 5 th or every 7 th epoch
        Returns:
                None
        '''
        self._createModel(batch_size = 128)
        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}/{epochs}')
            self.model.fit(data, epochs=1,verbose=verbose)
            
            if epoch % save_at == 0:
                self.model.save(f'model-{epoch}-epochs.h5')

    def predict(self, start_seed, gen_size=100, temp=1.0):
        '''
        
        Generates further texts according to the seed text

        Parameters:
                start_seed (str) : seed that model will use to generate further texts
                gen_size (int) : number of characters to generate 700 - 1000 are the most ideal ones
        Returns:
                None
        '''
        num_generate = gen_size
        input_eval = [char_to_ind[s] for s in start_seed]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        temperature = temp
        self.model.reset_states()

        for i in range(num_generate):
          predictions = self.model(input_eval)
          predictions = tf.squeeze(predictions, 0)
          predictions = predictions / temperature
          predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
          input_eval = tf.expand_dims([predicted_id], 0)
          text_generated.append(ind_to_char[predicted_id])
        return (start_seed + ''.join(text_generated))

    @property
    def __doc__(self):
        return     '''
            Generator object can construct the model,
            save the weights, load the weights train the model, 
            and make predictions 
            
            ---------------------------------------------------
            
            Trainging example : 

            model = Generator() # creating an instance of model

            model.train(dataset,epochs = 5, verbose=1, save_at=1) # training the model
            ----------------------------------------------------
            
            Continue training from a saved weights file : 

            model = Generator() # creating an instance of model
            
            model.load_weights('model-3-epochs.h5', mode = 'training') # loading the weights

            model.train(dataset,epochs = 5, verbose=1, save_at=1) # training the model
            -----------------------------------------------------

            Preditction example :

            model = Generator() # creating an instance of model

            model.load_weights('model-10-epochs.h5') # loading the weights
            
            print(model.predict('hello')) # making prediction and printing
            -----------------------------------------------------
            '''