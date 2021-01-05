import sys 
from model import Generator

seed = sys.argv[1]

model = Generator()  # initializing the model

model.load_weights('../models/model-5-epochs-256-neurons.h5')  # loading weights

print('=======================Generated Text=======================')

generatedText = model.predict(start_seed=seed, gen_size=1000)  # making preditions

print(generatedText)
