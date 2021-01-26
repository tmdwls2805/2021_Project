from model import *
from train import *

char2idx = prepro_configs['char2idx']
idx2char = prepro_configs['idx2char']

text = ""
test_index_inputs, _ = enc_processing([text], char2idx)
outputs = model.inference(test_index_inputs)

print(' '.join([idx2char[str(o)] for o in outputs]))