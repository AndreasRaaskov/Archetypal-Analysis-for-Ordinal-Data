"""
Implementation of Archetypal analysis using PyThorch
"""

import torch
from scipy.sparse import csr_matrix
import numpy as np
from py_pcha_master.py_pcha.furthest_sum import furthest_sum


def AA(data,K, conv_crit=1E-6,maxiter=500):

    #Convert data to tensor
    X=torch.tensor(data.values,dtype=torch.float)

    N, M = X.size()
    try:
        i=furthest_sum(data.values, K, [int(np.ceil(M * np.random.rand()))]) #Select 3 random stating point  TODO implement furthest sum algoritme instead
    except IndexError:
        class InitializationException(Exception): pass
        raise InitializationException("Initialization does not converge. Too few examples in dataset.")



    #Initialize C
    C=torch._sparse_csr_tensor(torch.tensor(range(K+1)),torch.tensor(i),torch.ones(K),(K,M)).to_dense().transpose(0,1)
    C.requires_grad=True

    XC=torch.matmul(X, C)

    #Initilice S
    S=torch.rand(K,M).softmax(dim=0)
    S.requires_grad=True

    optimiser=torch.optim.SGD((C,S),lr=0.001)

    SSE=torch.linalg.norm(X-torch.matmul(XC,S)) #Calculate TODO ask morten about the correct matrix norm

    #SSE.backward()

    iter_=0

    SSE_old=torch.tensor(np.inf)
    #SSE=torch.tensor(0)
    miniter=10

    while (SSE_old >= SSE*(1+conv_crit) and iter_ < maxiter) or iter_<miniter:

        iter_+=1

        SSE_old=SSE

        optimiser.zero_grad()



        SSE.backward()

        optimiser.step()

        C=C.detach()
        C=C.clip(0)
        C=torch.matmul(C,torch.diag(1/(C.sum(dim=0)+np.finfo(float).eps)))
        C.requires_grad = True


        S=S.detach()
        S=S.clip(0)
        S=torch.matmul(S,torch.diag((1/S.sum(dim=0)+np.finfo(float).eps)))
        S.requires_grad = True






        XC = torch.matmul(X, C)
        SSE = torch.linalg.norm(X-torch.matmul(XC, S))

        print(SSE)

    return C






