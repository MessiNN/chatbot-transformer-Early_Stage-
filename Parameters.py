import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


import random
import re
import os
import unicodedata
import itertools
import pandas as pd

from torch.nn.utils.rnn import pad_sequence


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

save_dir = os.path.join("/content/drive/MyDrive", "data", "save")

PAD_token = 0
SOS_token = 1
EOS_token = 2

MAX_LENGTH = 20
MIN_COUNT = 6

model_name = 'Cleopatra_model'

encoder_n_layers = 4
decoder_n_layers = 6
embedding_size = 744
head_num = 8

if embedding_size % head_num != 0:
    raise ValueError("embedding_size / head_num must result in an integer")

dropout = 0.2
batch_size = 20
decoder_learning_ratio = 0.80
learning_rate = 0.001
vocab_size = libra.num_words
task = "train"

start_model = "no"
loadFilename = None if start_model == "no" else "/content/drive/MyDrive/data/save/Cleopatra_model/512-8_1368/1000_checkpoint.tar"

model = Transformer(embedding_size, head_num, batch_size, dropout, learning_rate, decoder_learning_ratio,
                 encoder_n_layers, decoder_n_layers, vocab_size, libra, task, loadFilename)

model = model.to(device)

clip = 50.0
n_iteration = 10000
print_every = 100
save_every = 1000

if task == "train":
    trainIters(model_name, libra, pairs, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, loadFilename, vocab_size, model)

if task == "test":
    evaluateInput(model, libra, embedding_size)
