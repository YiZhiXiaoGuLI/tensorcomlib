import numpy as np
from tensorcomlib import tensor as tl
from functools import reduce

#Tensor Model-n Unfold and Fold
def unfold(ten,mode):

    shp = ten.shape
    ndim = ten.ndims()

    order = []
    order.extend([mode])
    order.extend(range(0, mode))
    order.extend(range(mode + 1, ndim))

    data = ten.data
    newdata = np.transpose(data, order)
    newdata = newdata.reshape(shp[mode], int(reduce(lambda x, y: x * y, shp) / shp[mode]))

    return newdata


def fold(G, shape, mode):

    return tl.tensor(X)

#Tensor Times Matrix
def tensor_times_mat(X,mat,mode):
    shp = X.shape
    ndim = X.ndims()

    order = []
    order.extend([mode])
    order.extend(range(0, mode))
    order.extend(range(mode + 1, ndim))

    newdata = np.dot(mat,unfold(X,mode).data)
    p = mat.shape[0]

    newshp = [p]
    newshp.extend(shp[0:mode])
    newshp.extend(shp[mode+1:ndim])

    T = newdata.reshape(newshp)

    T = np.transpose(T,[order.index(i) for i in range(len(order))])
    return tl.tensor(T)

#Tensor 
def tensor_multi_times_mat(X,matlist,modelist,transpose):
    res = X
    for i,(mat,mode) in enumerate(zip(matlist,modelist)):
        if transpose:
            res = tensor_times_mat(res,mat.T,mode)
        else:
            res = tensor_times_mat(res, mat, mode)
    return res

def tensor_times_vec(X,vec,mode):
    ndim = X.ndims
    return tl.tensor(T)

def tensor_times_tensor(X1,X2):
    pass


def tensor2vec(X):
    return X.data.reshape(X.size(),order='F')

def vec2tensor():
    pass

def khatri_rao():
    pass

def kronecker(ten1,ten2):
    res = np.kron(ten1.data,ten2.data)
    return tl.tensor(res,res.shape)

def einstein():
    pass

def tenzeros(shp):
    data = np.ndarray(shp)
    data.fill(0)
    return tl.tensor(data,shp)

def tenones(shp):
    data = np.ndarray(shp)
    data.fill(1)
    return tl.tensor(data,shp)

def tenrands(shp):
    data = np.random.random(shp)
    return tl.tensor(data,shp)

def teninner(X1,X2):
    if(X1.shape != X2.shape):
        raise ValueError("All the tensor's shape must be same!")
    res = (X1.data) * (X2.data)
    return tl.tensor(res,X1.shape)

def tenouter(X1,X2):
    return tl.tensor(np.tensordot(X1.data, X2.data, axes=0))

def tennorm(X):
    return np.sqrt(np.sum(X.data**2))

def tensor_contraction(X1,X2):
    return tl.tensor(np.tensordot(X1.data,X2.data,axes=2))


# Tensor Addition and Subtraction
def tensor_add(X1,X2):
    return tl.tensor(X1.data+X2.data)

def tensor_sub(X1,X2):
    return tl.tensor(X1.data-X2.data)