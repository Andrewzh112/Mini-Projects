from preprocess import SEQUENCE_LENGTH
import torch
import json


OUTPUT_DIM = 47         # output vocab size
EMBEDDING_DIM = 128     #
LR = 1e-3               # learning rate
NUM_LAYERS = 2          # number of lstm/transformer layers
NHEAD = 4               # number of attention heads
NUM_CHANNELS = 200      # number of channels for cnn
KERNELS = [3, 5, 7, 9]  # list of kernels for cnn
EPOCHS = 50             #
BATCH_SIZE = 64         #
DROPOUT = 0.3           # drop out probability
HIDDEN_SIZE = 32        # number of hidden units
NUM_STEPS = 500         # max number of steps for music generation
TEMPERATURE = 1.0       # temperature for sampling
MODEL_TYPE = 'Transformer'      # Options: CNN/LSTM/Transformer


MODEL_PATH = 'ClassicalModel_test.pt'          #
OUTPUT_PATH = 'generated_songs/'               #
SEED = '67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _'
TOPK = 10
CRITERION = torch.nn.CrossEntropyLoss()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('mapping.json', 'r') as fp:
    NOTES_MAPPING = json.load(fp)
fp.close()
