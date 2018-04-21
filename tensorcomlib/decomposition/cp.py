import  numpy as np
from  tensorcomlib.base import unfold
from  tensorcomlib import tensor
from  tensorcomlib.MatrixSVD import SVD

def cp(X, maxiter=1000, rank=None, init='svd', random_seed = None, eps = 1e-11, tol=1e-10):

    dims = X.ndims()
    modelist = list(range(dims))
    shape = X.shape

    if init == 'svd':
        factors = []
        for mode in range(dims):
            U,_,_ = SVD.PartialSvd(unfold(X,mode),rank)
            factors.append(U[:, :rank])

    elif init == 'random':
        seed = np.random.RandomState
        factors = [seed.random_sample((shape[i],rank)) for i in range(dims)]

    error = []
    for iteration in range(maxiter):
        for mode in range(dims):
            pass

