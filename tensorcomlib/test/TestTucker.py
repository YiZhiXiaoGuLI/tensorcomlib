import  numpy as  np
from tensorcomlib.decomposition import tucker as tk
from  tensorcomlib import  base
from  tensorcomlib import  tensor
import  tensorly


def caleps(X):
    ndim = data.ndims()
    normx = base.tennorm(X)
    print("eps max:"+str(ndim/normx))

def TestHooI(data):
    U, core = tk.hooi(data, eps= 0.12, tol=1e-10, plot=True)

    ndim = data.ndims()
    modelist = list(range(ndim))
    data1 = base.tensor_multi_times_mat(core, U, modelist=modelist, transpose=False)

    print(data.data.reshape(1024)[1:10])
    print(data1.data.reshape(1024)[1:10])



if __name__ == '__main__':

    B = np.round(np.random.randint(1, 10, (8, 16, 8)))
    data = tensor.tensor(B)
    caleps(data)
    TestHooI(data)
