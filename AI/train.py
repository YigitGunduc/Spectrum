import re
import string
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

path_to_file = '../data/raplyrics.txt' # text dataset path

text = open(path_to_file, 'r').read() # loading text dataset

text = re.sub(r'[^\x00-\x7f]', r'', text) # removing non ascii characters

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
# ------------------------------------------------------------
vocab = sorted(set(string.printable)) # variety of characters
vocab_size = len(vocab) # num of items in the vocab          |                      
batch_size = 128 # batch_size                                
buffer_size = 10000 # buffer_size                       
seq_len = 120 # seq_len                                      
                                                 
char_to_ind = {u: i for i, u in enumerate(vocab)}             
ind_to_char = np.array(vocab)
encoded_text = np.array([char_to_ind[c] for c in text])
# ------------------------------------------------------------


# preprocessing the data 
# ------------------------------------------------------------
total_num_seq = len(text)//(seq_len+1)

char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

sequences = char_dataset.batch(seq_len+1, drop_remainder=True)

def create_seq_targets(seq):
    input_txt = seq[:-1]
    target_txt = seq[1:]
    return input_txt, target_txt

dataset = sequences.map(create_seq_targets)

dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
# ------------------------------------------------------------

if __name__ == '__main__':
    from model import Generator
    
    parser = argparse.ArgumentParser(description="training Rap lyrics generator")
    parser.add_argument('--epochs', type=int, default=100, help='epoch size')
    parser.add_argument('--save_at', type=int, default=5, help='to save at ever n th epoch')
    parser.add_argument('--verbose', type=int, default=1, required=False, help='to print loss and epoch number of not to')

    args = parser.parse_args()

    # Training the model
    # ------------------------------------------------------------
    model = Generator() # creating an instance of model

    # training the model
    model.train(dataset, epochs=args.epochs, verbose=args.verbose, save_at=args.save_at)
    # ------------------------------------------------------------

