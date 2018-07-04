import numpy as np

def H(A):
    return np.transpose(np.conjugate(A))

#TruncatedSvd
def TruncatedSvd(X,eps_svd = None):


    U,S,V = np.linalg.svd(X,full_matrices=False)

    N1,N2 = X.shape
    r = min(N1,N2)
    for i in range(r):
        if sum(S[i:r])/sum(S) <= eps_svd:
            r =i
            break
    U = U[:,:r].copy()
    S = S[:r].copy()
    V = V[:r,:].copy()

    return U,H(V),r

#PartialSvd
def PartialSvd(X,n):
    U, S, V = np.linalg.svd(X, full_matrices=True)
    U= U[:,:n]
    return U,S,V
