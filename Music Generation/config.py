from preprocess import SEQUENCE_LENGTH
import torch
import json


OUTPUT_DIM = 47         # output vocab size
EMBEDDING_DIM = 128     # 
LR = 1e-3               # learning rate
NUM_LAYERS = 2          # number of lstm layers
EPOCHS = 120            #
BATCH_SIZE = 128        # 
DROPOUT = 0.2           # drop out probability
HIDDEN_SIZE = 32        # number of hidden units for LSTM
NUM_STEPS = 500         # number of steps for music generation
TEMPERATURE = 0.9       # temperature for sampling

MODEL_PATH = 'ClassicalModel.pt'          # 
SEED = '67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _'  # seed going into melody generation
CRITERION = torch.nn.CrossEntropyLoss()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('mapping.json','r') as fp:
      NOTES_MAPPING = json.load(fp)
fp.close()