import torch
from scipy.sparse import csr_matrix
import numpy as np
import time
import math
import pickle
from os import listdir
from os.path import isfile, join
import argparse

# --------- CONSTANTS ---------

parser = argparse.ArgumentParser(description='Calculate PPMI matrices for given corpus.')
parser.add_argument('input_path', type=str, help='Input directory for cooccur matrices.')
parser.add_argument('output_path', type=str, help='Output directory to store ppmi matrices.')

args = parser.parse_args()

cooccurPath = args.input_path
ppmiPath = args.output_path

def numpyToPPMI(X, f):
    # clear the diagonals and replace with total counts
    print("Formatting cooccur matrix...")
    a = time.time()
    V = torch.from_numpy(X.todense()).float()
    V = torch.mul(V, torch.ones_like(V) - torch.diag(torch.ones(f)))
    S = torch.mv(V, torch.ones(f))
    V.add_(torch.diag(S))
    print("Formatted. Time elapsed: {}".format( time.time() - a))
    print(V[0:3, 0:3])

    print("Calculating logXij...")
    a = time.time()
    logXij = torch.log1p(V)
    print("Calculated. Time elapsed: {}".format(time.time() - a))
    print(logXij[0:3, 0:3])

    print("Calculating logXiXj...")
    a = time.time()
    logXiXj = torch.log1p(S.repeat(f,1)) + torch.log1p(S.repeat(f,1)).t()
    print("Calculated. Time elapsed: {}".format(time.time() - a))
    print(logXiXj[0:3, 0:3])
    
    print("Calculating PPMI...")
    a = time.time()
    trace = math.log(torch.sum(S))
    PPMI = torch.max(logXij + trace - logXiXj, torch.zeros(f,f))
    print("Calculated. Time elapsed: {}".format(time.time() - a))
    print(PPMI[0:3,0:3])
    return PPMI

def using_tocoo_izip(x, name):
    cx = x.tocoo()    
    with open(name, "a+") as fp:
        fp.write("word,context,pmi\n")
        for i,j,v in zip(cx.row, cx.col, cx.data):
            fp.write("{},{},{}\n".format(i,j,v))

# ----------------- Execute PPMI ------------------------

print("======= Starting PPMI =======")
s = time.time()
a = time.time()

# for loop
cooccurFiles = [file for file in listdir(cooccurPath) if isfile(join(cooccurPath, file))]

for file in cooccurFiles:
    batchNo = file.split('h')[1].split('.')[0]
    print("Batch number {}".format(batchNo))
    with open( cooccurPath + file, 'rb') as fp:
        X = pickle.load(fp)

    print("Performing PPMI...")
    a = time.time()
    f = X.shape[0]
    PPMI = numpyToPPMI(X,f)
    print("Finished PPMI. Time elapsed: {}".format(time.time() - a))

    # takes 4077 seconds, or 1h08m
    print("Saving torch tensor...")
    a = time.time()
    torch.save(PPMI, "{}PPMI_{}.t".format(ppmiPath, batchNo))
    print("Written. Time elapsed: {}".format(time.time() - a))


