import json
import random
import numpy as np
import  os
import torch
def loadvocab(path):
    vocab = {}
    with open(path,encoding='utf-8') as file:
        for l in file.readlines():
            vocab[len(vocab)] = l.strip()
    return vocab

def loaddata(path):
    jsonlist = []
    with open(path,encoding='utf-8') as file:
        for line in file.readlines():
            jsonlist.append(json.loads(line))
    return jsonlist

def datasplit(data,num_train):
    train = data[:num_train]
    dev = data[num_train:]
    return train,dev

def set_seed(seed = 427):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
