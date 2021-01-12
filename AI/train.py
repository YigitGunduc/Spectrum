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
    parser.add_argument('--rnn_neurons', type=int, default=256, help='rnn neurons in every single layer')
    parser.add_argument('--embed_dim', type=int, default=64, help='dims for embedding layer')
    parser.add_argument('--dropout', type=float, default=0.3, required=False, help='keep prob for dropout')
    parser.add_argument('--num_layers', type=int, default=2, help='number of rnn layers')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning_rate')
    parser.add_argument('--cuda', type=bool, default=False, help='if true uses GPU acceleration')

    args = parser.parse_args()

    # Training the model
    # ------------------------------------------------------------
    model = Generator(rnn_neurons=args.rnn_neurons, embed_dim=args.embed_dim, dropout=args.dropout, num_layers=args.num_layers, learning_rate=args.learning_rate) # creating an instance of model

    # training the model
    model.train(dataset, epochs=args.epochs, verbose=args.verbose, save_at=args.save_at,cuda=args.cuda)
    # ------------------------------------------------------------
    pred = model.predict('hello')
    print(pred)