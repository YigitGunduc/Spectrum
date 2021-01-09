import argparse 
from model import Generator

parser = argparse.ArgumentParser(description="training Rap lyrics generator")
parser.add_argument('--seed', type=str, default=None, help='seed text')

args = parser.parse_args()

model = Generator()  # initializing the model

model.load_weights('../models/model-5-epochs-256-neurons.h5')  # loading weights

print('=======================Generated Text=======================')

generatedText = model.predict(start_seed=args.seed, gen_size=1000)  # making preditions

print(generatedText)
