import re
import os
import time
from nltk import word_tokenize
import pickle
import numpy as np
from os import listdir
from csv import DictReader
from collections import Counter
from os.path import isfile, join
from multiprocessing import Pool
import scipy.sparse as sp
from gensim.corpora import Dictionary

# ---------- CONSTANTS ----------

import argparse
parser = argparse.ArgumentParser(description='Calculate cooccurence matrices for given timeslices.')
parser.add_argument('input_path', type=str, help='Input directory for tokens.')
parser.add_argument('dictionary', type=str, help='Input dictionary file path')
parser.add_argument('output_path', type=str, help='Output directory to store matrices.')

args = parser.parse_args()

# read in the freq counter
cooccurPath = args.output_path
tokensPath = args.input_path

print("------Starting Cooccur------")
dct = Dictionary.load(args.dictionary).token2id
print("Reading in the directory of token files")
rTime = time.time()
tokenFiles = [f for f in listdir(tokensPath) if isfile(join(tokensPath, f))]
print("Read. Elapsed time: {}".format(time.time() - rTime))

# ---------- HELPER FUNCTIONS ------------

# efficient sum of sparse matrices
def sum_sparse(m):
    x = np.zeros(m[0].shape)
    for a in m:
        ri = np.repeat(np.arange(a.shape[0]),np.diff(a.indptr))
        x[ri,a.indices] += a.data
    return x

# ---------- EXECUTOR -----------

# SET PARAMETERS
L = 15 # window size, or L in the paper
P = 7 # number of processors

'''
Logic: each row is a window, each column is a word. So we are vectorizing
each window.
X.T * X = [ window x word ]^T [ window x word ] 
= [ word x window ] [ window x word ]
Say there are V words and W windows
Then X.T * X \in [V x V]
and V[i,j] = \sum_(w in W) (word_i in window w)(word_j in window w)

What are the diagonals? 
https://stackoverflow.com/questions/42814452/co-occurrence-matrix-from-list-of-words-in-python/42814963
'''

def cooccur(tokens):
    # creates word and window pairs to pass to cooccur
    wordTarget = tokens[L:len(tokens) - L]
    windowTarget = [tokens[i - L : i + L + 1] for i in range(L, len(tokens) - L)]
    pairs = list(zip(wordTarget, windowTarget))

    parallelTime = time.time()

    rows, cols, vals = [], [], []
    for vword in dct.keys():
        index = dct[vword]
        rows.append(index)
        cols.append(index)
        vals.append(0)
        
    for word, window in pairs:
        for coword in window:
            if dct.get(word) is not None and dct.get(coword) is not None:
                rows.append(dct[word])
                cols.append(dct[coword])
                vals.append(1)
                
    X = sp.csr_matrix((vals, (rows, cols)))
    return X

def parseYear(filename):
    year = filename.split('-')[0]
    year = re.sub('\D', '', year)
    return int(year)
# --------- WRITE TOKENS FOR FILES ---------
print("Starting cleaning")

batchCount = 0
running_batch_time = 0
last_time = time.time()
interval = 20
print("creating batches in intervals of {} years".format(interval))

fileYearPairs = [(t, parseYear(t)) for t in tokenFiles if parseYear(t) > 1400]

sortedFileYearPairs = sorted(fileYearPairs, key = lambda p: p[1])

yearMin = min([fyp[1] for fyp in fileYearPairs])
yearMax = max([fyp[1] for fyp in fileYearPairs])
print("min: {}, max: {}".format(yearMin, yearMax))

batches = []
batch = []
batchYear = yearMin
for pair in sortedFileYearPairs:
    filename = pair[0]
    year = pair[1]
    if year - batchYear < interval:
        batch.append(filename)
    else:
        batches.append(batch)
        batchYear = year
        batch = []
        batch.append(filename)
batches.append(batch)
        
print("starting tokens batching")
print([len(b) for b in batches])

# need to batch by year
timeElapsed = 0
    
lastTime = time.time()
for batchIndex in range(1, len(batches)+1):
    batch = batches[batchIndex-1]
    
    batchCooccur = sp.csr_matrix((len(dct),len(dct)))
    for filename in batch:
        with open(tokensPath + filename, 'rb') as fp:
            tokens = pickle.load(fp)
            batchCooccur += cooccur(tokens)
    
#     batchCooccur = np.dot(preCooccur.T, preCooccur)
    
    with open("{}cooccurBatch{}.p".format(cooccurPath, batchIndex), "wb") as fp:
        pickle.dump(batchCooccur, fp)

    batchTime = time.time() - lastTime
    timeElapsed += batchTime
    ETA = (timeElapsed/batchIndex) * (len(batches) - batchIndex)
    ETAstring = "{}:{}:{}".format( int(ETA / 3600), int( (ETA % 3600) / 60 ), int(ETA % 60))

    print("Batch {} of {} | Batch time: {} | ETA: {}".format(batchIndex, len(batches), batchTime, ETAstring))
    lastTime = time.time()
