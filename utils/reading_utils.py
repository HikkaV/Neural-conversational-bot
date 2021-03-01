import pandas as pd
import gensim
import numpy as np


def read_chameleons(path, columns=None):
    with open(path, 'rb') as f:
        res = f.readlines()
    values = []
    for line in res:
        try:
            vals = line.decode().split(" +++$+++ ")
        except:
            vals = line.decode('cp1251').split(" +++$+++ ")
        values.append([i.strip() for i in vals])
    values = pd.DataFrame(data=values, columns=columns)
    return values


def load_emb_from_disk(path):
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

    return model


def load_glove(file):
    print("Loading Glove Model")
    f = open(file, 'r')
    glove_embeddings = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        glove_embeddings[word] = wordEmbedding
    print(len(glove_embeddings), " words loaded!")
    return glove_embeddings
