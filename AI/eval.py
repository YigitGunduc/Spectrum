import sys 
from model import Generator

# creating a new instance of model
model = Generator()

# loading weights
model.load_weights('model-1-epochs-256-neurons.h5')

# making preditions
generatedText = model.predict(start_seed=sys.argv[1], gen_size=1000, temp=1.0)

print(generatedText)