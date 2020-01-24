import numpy as np
import scipy.io as sio
import copy
import pandas as pd

'''
- U: the embeddings of all words at time t [batchSize x D]
- Y: pmi values [batchSize x V]
- Vml: previous time's embeddings [batchSize x D]
- Vpl: next time's embeddings [batchSize x D]
- lam: scalar
- tau: scalar
- gam: scalar
- batch: [batchSize x 1] list of word indices 
- iflag: bool
'''
import torch
import pickle
'''
Ax = B
'''
def torch_lstsq(A,B):
    U, S, V = torch.svd(A)
    S_inv = (1./S).view(1,S.size(0))
    VS = V*S_inv # inverse of diagonal is just reciprocal of diagonal
    UtB = torch.mm(U.permute(1,0), B)
    betas_svd = torch.mm(VS, UtB) # []
    return betas_svd

def torch_update(U,Y,Vm1,Vp1,lam,tau,gam,batch,iflag):
    
    UtU = torch.mm(U.t(),U) # [D x V] x [V x D]
    D = UtU.shape[0]    # D
    if iflag:   
        A = UtU + (lam + 2*tau + gam)*torch.eye(D)
        # [D x D] = [D x D] + [D x D]
    else:       
        A = UtU + (lam + tau + gam)*torch.eye(D)
        # [D x D] = [D x D] + [D x D]
       
    Uty = torch.mm(U.t(),Y) # rxb
    # [D x batchSize] = [D x V] x [V x batchSize]
    Ub = U[batch,:].t()  # [batchSize x D]^T = [D x batchSize]
    B = Uty + gam*Ub + tau*(Vm1.t()+Vp1.t())  # [D x batchSize] = [D x batchSize] + [D x batchSize] + [D x batchSize] + [D x batchSize]
    
    '''
    numpy.linalg.lstsq(a, b, rcond=-1)[source]
    Return the least-squares solution to a linear matrix equation.
    Solves the equation a x = b by computing a vector 
    x that minimizes the Euclidean 2-norm || b - a x ||^2. 
    The equation may be under-, well-, or over- determined 
    (i.e., the number of linearly independent rows of a can 
    be less than, equal to, or greater than its number of 
    linearly independent columns). If a is square and of 
    full rank, then x (but for round-off error) is the 
    exact solution of the equation.
    
    finding x to minimize || B - Ax ||^2
    
    B - Ax = [D x batchSize] - [D x D][D x batchSize]
    '''
    Vhat = torch_lstsq(A,B) # [] = lstsq([D x D], [D x batchSize])
    '''finding x to minimize || B - Ax ||^2
    B - Ax = [D x batchSize] - [D x D][D x batchSize]'''
    # Vhat [D x batchSize] (20936, 50)
    return Vhat.t() # [batchSize x D] (Vhat is like a group of things, and the first element is the lstsq)

'''
import_static_init(T) - loads static word embeddings from a .mat file.
The dimensions of .mat are [V x D], where V vocab size and D dimensions.
'''
# go from embID (out of 31k) to ppmiID (out of 36k)
with open("./vep_embIDToPPMIID.p", "rb") as fp:
    embIDToPpmiID = pickle.load(fp)
    
# go from ppmiID (out of 36k) to embID (out of 31k)
with open("./vep_ppmiIDToEmbID.p", "rb") as fp:
    ppmiIDToEmbID = pickle.load(fp)

# staticEmbeddings[ppmiIDToEmbID[wordToID[word]]][0] = wordToID[word]

def torch_getmask(large, availableWords):
    print("\tlarge: {}, availableWords: {}".format(large, len(availableWords)))
    maskTemp = torch.zeros(large, large)
    maskTemp[availableWords, :] = 1
    mask = torch.mm(maskTemp, maskTemp.t()).ge(1)        
    return mask

def reconcile_embeddings(dct, embeddingsFilePath):
    # read the csv
    print("--- reconciling embeddings ---")
    emb = pd.read_csv(embeddingsFilePath, 
                  sep=' ', 
                  skiprows=[0], 
                  header=None) 
    
    # set the column names to be the word, and create the w_p column
    emb.columns = ['word', *list(range(100)), 'w_p'] # 690619 rows × 102 columns
    allWords = [w[1] for w in dct.items()]
    rectEmb = emb[emb.word.isin(allWords)]
    rectEmb['w_p'] = list(map(lambda w: dct.token2id[w], rectEmb['word'].values.tolist()))
    rectEmb = rectEmb.sort_values(by=['w_p']) # doesn't sort in place lmao
    embeddings = torch.from_numpy(rectEmb.as_matrix([*list(range(100))])).float()
    vocabulary = rectEmb['word'].values.tolist()
    w_pList = rectEmb['w_p'].values.tolist()
    
    return embeddings, list(zip(vocabulary, w_pList, range(len(w_pList))))

def torch_import_static_init(X,words,T):
    # emb is a numpy and needs to be a tensor
    df = pd.read_csv('./vep.vec', 
                  sep=' ', 
                  skiprows=[0], 
                  header=None) 
    
    Ulist = emb.repeat(len(T),1,1)
    Vlist = emb.repeat(len(T),1,1)
    print(Ulist.shape)
    return Ulist, Vlist
    
def torch_initvars(vocab_size,T,rank):
    # dictionary will store the variables U and V. tuple (t,i) indexes time t and word index i
    emb = torch.rand(vocab_size, rank)
    Ulist = emb.repeat(len(T),1,1)
    Vlist = emb.repeat(len(T),1,1)
    return Ulist, Vlist
    
    
def torch_getmat(file):
    model = torch.load(file)
    return model

def getbatches(vocab, b):
    batchinds = []
    current = 0
    while current < vocab:
        inds = range(current, min(current + b, vocab))
        current = min(current + b, vocab)
        batchinds.append(inds)
    return batchinds

#   THE FOLLOWING FUNCTION TAKES A WORD ID AND RETURNS CLOSEST WORDS BY COSINE DISTANCE
from sklearn.metrics.pairwise import cosine_similarity
def getclosest(wid,U):
    C = []
    for t in range(len(U)):
        temp = U[t]
        K = cosine_similarity(temp[wid,:],temp)
        mxinds = np.argsort(-K)
        mxinds = mxinds[0:10]
        C.append(mxinds)
    return C
        
# THE FOLLOWING FUNCTIONS COMPUTES THE REGULARIZER SCORES GIVEN U AND V ENTRIES
def compute_symscore(U,V):
    return np.linalg.norm(U-V)**2

def compute_smoothscore(U,Um1,Up1):
    X = np.linalg.norm(U-Up1)**2 + np.linalg.norm(U-Um1)**2
    return X
    
