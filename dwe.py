#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:10:42 2016

"""

# main script for time CD 
# trainfile has lines of the form
# tok1,tok2,pmi

import os
import numpy as np
import util
import pickle
import time
import torch
from gensim.corpora import Dictionary

import argparse

# ---------- HELPERS -------------
def print_params(r,lam,tau,gam,emph,ITERS):
    print('rank = {}'.format(r))
    print('frob regularizer = {}'.format(lam))
    print('time regularizer = {}'.format(tau))
    print('symmetry regularizer = {}'.format(gam))
    print('emphasize param   = {}'.format(emph))
    print('total iterations = {}'.format(ITERS))

# ------------ MAIN --------------

parser = argparse.ArgumentParser(description='Align embeddings via DWE.')

parser.add_argument('dictionary', type=str, help='Filepath for input dictionary.')
parser.add_argument('model', type=str, help='Filepath for baseline static embeddings generated from FastText. Should be a .vec file.')
parser.add_argument('trainhead', type=str, help='Directory containing PPMI matrices.')
parser.add_argument('savehead', type=str, help='Directory to store model outputs.')

# optional arguments
parser.add_argument('-rank', type=int, default=100, help='Rank (no. of dimensions) of the baseline static embeddings.')
parser.add_argument('-iters', type=int, default=14, help='total iterations')
parser.add_argument('-lam', type=float, default=10., help='frob regularizer')
parser.add_argument('-tau', type=float, default=50., help='time regularizer')
parser.add_argument('-gam', type=float, default=100., help='symmetry regularizer')
parser.add_argument('-emph', type=float, default=1., help='emphasize parameter')

args = parser.parse_args()

# make save directory if it doesn't exist
if not os.path.exists(args.savehead):
    print("{} does not exist, creating directory".format(args.savehead))
    os.mkdir(args.savehead)

# PARAMETERS
print("======== DW2V ========")

print("Loading files...")
a = time.time()

dct = Dictionary.load(args.dictionary)

print('Extracting vectors...')
embeddingsFilePath = args.model
embeddings, wpe = util.reconcile_embeddings(dct, embeddingsFilePath)
vocabularySize = len(wpe)
with open(args.savehead + 'wpe.p','wb') as fp:
    pickle.dump(wpe, fp)

print("Sanity checks")
for t in wpe[:5]:
    print(t)
    print(t[0]) # prints word
    print(dct[t[1]]) # maps w_p to word

print('Extracted.')

print('Creating mask...')
mask = util.torch_getmask(len(dct), [t[1] for t in wpe])
print('Created.')

# how to autocalculate?

number_of_slices = len(os.listdir(args.trainhead))
T = range(1, number_of_slices + 1) # total number of time points (20/range(27) for ngram/nyt)
cuda = True

trainhead = args.trainhead # location of training data
savehead = args.savehead

print("Loaded. Time elapsed: {}".format(time.time() - a))
    
ITERS = args.iters
lam = args.lam
gam = args.gam
tau = args.tau
D   = args.rank
emph = args.emph

savefile = savehead+'L'+str(lam)+'T'+str(tau)+'G'+str(gam)+'A'+str(emph)

print('Starting training with following parameters:')
print_params(D,lam,tau,gam,emph,ITERS)
print('{} words, {} time points.'.format(vocabularySize,len(T)))

print("======== Starting DW2V ========")

print("Loading static embeddings...")
a = time.time()
Ulist = embeddings.repeat(len(T),1,1) # [t x V x D]
Vlist = embeddings.repeat(len(T),1,1) # [t x V x D]
print("Loaded. Time elapsed: {}".format(time.time() - a))
print("Ulist shape: {}".format(Ulist.shape))

print('Creating batches (but only one batch now)...')
a = time.time()
batches = [range(vocabularySize)]
print("Batches created. Time elapsed: {}".format(time.time() - a))
print("Number of batches: {}".format(len(batches)))

print("+++++++ Starting epochs +++++++")
for epoch in range(ITERS):  
    print("Epoch {} of {}".format(epoch, ITERS))
    epochTime = time.time()

    print("Trying to load U,V tensors...")
    try:
        Ulist = torch.load("{}ngU_iter{}.t".format(savefile,epoch))
        Vlist = torch.load("{}ngV_iter{}.t".format(savefile,epoch))
        print('Epoch {} loaded succesfully.'.format(epoch))
        continue
    except(IOError):
        pass

    loss = 0
    # shuffle times
    if epoch == 0: times = T
    else: times = np.random.permutation(T)

    for t in range(len(times)):   # select a time
        print('Epoch {}, time {}'.format(epoch, t+1))
        torchSlice = trainhead + 'PPMI_' + str(t+1) + '.t'

        print("Subselecting PMI...")
        a = time.time()

        # problem is here. it's moving the words down by one each time.
        pmi = torch.masked_select(util.torch_getmat(torchSlice), mask).view(vocabularySize, -1)
        print("PMI created. Time elapsed: {}".format(time.time() - a))
        print("PMI shape: {}".format(pmi.shape))
        print(pmi[0:3,0:3])

        for batchIndex in range(len(batches)): # select a mini batch
            print('Batch {} out of {}'.format(batchIndex + 1, len(batches)))
            batch = batches[batchIndex]

            ## UPDATE V
            # if it's the first batch
            if t==0:
                vp = torch.zeros(len(batch),D) # [batchSize x D]
                up = torch.zeros(len(batch),D) # [batchSize x D]
                iflag = True
            else:
                vp = Vlist[t-1,:,:] # previous time, [batchSize x D]
                up = Ulist[t-1,:,:] # previous time, [batchSize x D]
                iflag = False

            print("vp shape: {}".format(vp.shape))
            print(vp[0:3,0:3])

            print("up shape: {}".format(up.shape))
            print(up[0:3,0:3])

            # if it's the last batch
            if t==len(T)-1:
                vn = torch.zeros(len(batch),D) # [batchSize x D]
                un = torch.zeros(len(batch),D) # [batchSize x D]
                iflag = True
            else:
                vn = Vlist[t+1,:,:] # [batchSize x D]
                un = Ulist[t+1,:,:] # [batchSize x D]
                iflag = False

            print("vn shape: {}".format(vn.shape))
            print(vn[0:3,0:3])

            print("un shape: {}".format(un.shape))
            print(un[0:3,0:3])

            print("Updating U, V...")
            a = time.time()
            Vlist[t,:,:] = util.torch_update(Ulist[t,:,:],emph*pmi,vp,vn,lam,tau,gam,batch,iflag) # [batchSize x D]
            Ulist[t,:,:] = util.torch_update(Vlist[t,:,:],emph*pmi,up,un,lam,tau,gam,batch,iflag) # [batchSize x D]
            print("Updated. Time elapsed: {}".format(time.time() - a))

    print("+++++++ Epoch {} finished. +++++++".format(epoch))
    print("Time elapsed: {}".format(time.time() - epochTime))

    print("Saving...")
    torch.save(Ulist, "{}ngU_iter{}.t".format(savefile,epoch))
    torch.save(Vlist, "{}ngV_iter{}.t".format(savefile,epoch))
    print("Saved.")
