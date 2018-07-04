import numpy as np
from  tensorcomlib import base
from  tensorcomlib import tensor
from  sklearn.utils.extmath import randomized_svd
from  tensorcomlib.MatrixSVD import SVD
from matplotlib.pylab import plt
import time

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

#TruncatedHosvd
def TruncatedHosvd(X,eps):
    U = [None for _ in range(X.ndims())]
    dims = X.ndims()
    S = X
    R = [None for _ in range(X.ndims())]
    for d in range(dims):
        C = base.unfold(X,d)
        U1,S1,r = SVD.TruncatedSvd(C,eps_svd=eps)
        R[d] = r
        U[d] = U1
        S = base.tensor_times_mat(S,U[d].T,d)
    return U,S,R,eps

#PartialHosvd
def PartialHosvd(X,ranks):
    U = [None for _ in range(X.ndims())]
    dims = X.ndims()
    S = X
    for d in range(dims):
        C = base.unfold(X, d)
        U1,_,_= SVD.PartialSvd(C,ranks[d])
        U[d] = U1
        S = base.tensor_times_mat(S, U[d].T, d)
    return U, S

#hooi
def hooi(X,maxiter=1000,init='svd',eps = 1e-11,tol=1e-10, plot = True):

    time0 = time.time()

    dims = X.ndims()
    modelist = list(range(dims))

    if init == 'svd':
        U,core,ranks,eps_svd = TruncatedHosvd(X,eps= eps)
        print('TruncatedHosvd Ranks:\t'+str(ranks))
        data = base.tensor_multi_times_mat(core, U, modelist=modelist, transpose=False)
        errorsvd = base.tennorm(base.tensor_sub(data, X))
        print('---------------------------->>>>>>')
        print('TruncatedHosvd Init:')
        print("Original tensor:",X.data.reshape(1024)[1:10])
        print("TruncatedHosvd tensor:",data.data.reshape(1024)[1:10])
        print("Truncated error:",errorsvd)

    else:
        U,core = randomized_hosvd(X)

    error_X = []
    error_iter = []

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
            Uk[i],_,_ = SVD.PartialSvd(C,ranks[i])

        core = base.tensor_multi_times_mat(X,Uk,list(range(dims)),transpose=True)
        U = Uk
        S2 = base.tensor_multi_times_mat(core,Uk,list(range(dims)),transpose=False)
        error0 = base.tennorm(base.tensor_sub(S2,S1))
        S1 = S2
        error_iter.append(error0)
        error1 = base.tennorm(base.tensor_sub(X, S2))
        error_X.append(error1)

        if error0<tol:
            print('---------------------------->>>>>>')
            print('HOOI:')
            print('Iteration:' + str(iteration) + '\t\t' + 'Error_iter:' + str(error0)+'\t\t'+'Error_X:' + str(error1))
            print("Cost time:",time.time()-time0)
            break

    if plot:
        plt.plot(error_X)
        plt.title('The norm difference between the reduction tensor and the original tensor')
        plt.xlabel('Iteration')
        plt.ylabel('Norm difference')
        plt.show()
        plt.plot(error_iter)
        plt.title('The difference between the norm of restoring tensors in two consecutive iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Norm difference')
        plt.show()

    return  U,core
