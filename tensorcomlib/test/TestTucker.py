import  numpy as  np
from tensorcomlib.decomposition import tucker as tk
from  tensorcomlib import  base
from  tensorcomlib import  tensor


def TestHooI(data):

    #eps: range(0,1)
    U, core = tk.hooi(data, eps=0.08, tol=1e-10, plot=True)

    ndim = data.ndims()
    modelist = list(range(ndim))
    data1 = base.tensor_multi_times_mat(core, U, modelist=modelist, transpose=False)

    print("Original tensor:",data.data.reshape(1024)[1:10])
    print("Tucker tensor:",data1.data.reshape(1024)[1:10])



if __name__ == '__main__':

    np.random.seed(5)
    B = np.round(np.random.randint(1, 10, (8, 16, 8)))
    print(B.shape)
    data = tensor.tensor(B)
    TestHooI(data)