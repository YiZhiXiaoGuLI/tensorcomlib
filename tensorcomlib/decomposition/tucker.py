import numpy as np
from  tensorcomlib import base
from  tensorcomlib import tensor
from sklearn.utils.extmath import randomized_svd



#hosvd
def hosvd(X):
    U = [None for _ in range(X.ndims())]
    dims = X.ndims()
    S = X
    for d in range(dims):
        C = base.unfold(X,d)
        U1,S1,V1 = np.linalg.svd(C,full_matrices=False)
        S = base.tensor_times_mat(S, U1.T,d)
        U[d] = U1
    core  = S
    return U,core


#randomized_hosvd
def randomized_hosvd(X):
    U = [None for _ in range(X.ndims())]
    dims = X.ndims()
    S = X
    for d in range(dims):
        C = base.unfold(X,d)
        U1, S1, V1 = randomized_svd(C, n_components=3, n_oversamples=10, n_iter='auto',
                     power_iteration_normalizer='auto', transpose='auto',
                     flip_sign=True, random_state=42)
        S = base.tensor_times_mat(S, U1.T, d)
        U[d] = U1
    core = S
    return U, core

def H(A):
    return np.transpose(np.conjugate(A))

def TruncatedSvd(X,eps_svd = None):

    U,S,V = np.linalg.svd(X,full_matrices=False)
    N1,N2 = X.shape
    r = min(N1,N2)
    for i in range(r):
        if sum(S[i:r]) <= eps_svd:
            r =i
            break
    U = U[:,:r].copy()
    S = S[:r].copy()
    V = V[:r,:].copy()

    return U,H(V),r



#TruncatedHosvd
def TruncatedHosvd(X,eps):
    U = [None for _ in range(X.ndims())]
    dims = X.ndims()
    S = X
    R = [None for _ in range(X.ndims())]
    for d in range(dims):
        C = base.unfold(X,d)
        eps_svd = eps**2*base.tennorm(X)**2/dims
        U1,S1,r = TruncatedSvd(C,eps_svd=eps_svd)
        R[d] = r
        U[d] = U1
        S = base.tensor_times_mat(S,U[d].T,d)
    return U,S,R,eps_svd



def PartialSvd(X,n):
    U, S, V = np.linalg.svd(X, full_matrices=True)
    U= U[:,:n]
    return U,S,V

def PartialHosvd(X,ranks):
    U = [None for _ in range(X.ndims())]
    dims = X.ndims()
    S = X
    for d in range(dims):
        C = base.unfold(X, d)
        U1,_,_= PartialSvd(C,ranks[d])
        U[d] = U1
        S = base.tensor_times_mat(S, U[d].T, d)
    return U, S

#hooi
def hooi(X,maxiter=1000,init='svd',eps = 1e-11,tol=1e-10):

    dims = X.ndims()
    modelist = list(range(dims))

    if init == 'svd':
        U,core,ranks,eps_svd = TruncatedHosvd(X,eps= eps)
        print('TruncatedHosvd Ranks:\t'+str(ranks))
        data = base.tensor_multi_times_mat(core, U, modelist=modelist, transpose=False)
        errorsvd = base.tennorm(base.tensor_sub(data, X))
        print(errorsvd)
        print(data.data.reshape(1024)[1:10])
        print(X.data.reshape(1024)[1:10])
    else:
        U,core = randomized_hosvd(X)

    error_old = 0
    normx = base.tennorm(X)
    S1 = X

    for iteration in range(maxiter):
        Uk = [None for _ in range(dims)]
        for i in range(dims):
            U1 = U.copy()
            U1.pop(i)
            L = list(range(dims))
            L.pop(i)
            Y = base.tensor_multi_times_mat(X,U1,modelist=L,transpose=True)
            C = base.unfold(Y,i)
            Uk[i],_,_ = PartialSvd(C,ranks[i])

        core = base.tensor_multi_times_mat(X,Uk,list(range(dims)),transpose=True)
        U = Uk
        S2 = base.tensor_multi_times_mat(core,Uk,list(range(dims)),transpose=False)
        # error = base.tennorm(base.tensor_sub(S2,S1))
        # S1 = S2
        error = base.tennorm(base.tensor_sub(X, S2))

        if error<tol:
            break

    print('iteration:' + str(iteration) + '\t\t' + 'error:' + str(error))
    # error = base.tennorm(base.tensor_sub(X, S1))
    print(error)
    return  U,core
