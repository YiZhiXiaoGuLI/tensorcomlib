import numpy as np
from tensorcomlib import base
from tensorcomlib import tensor

def TestSubAdd():
    ten1 = tensor.tensor(np.arange(10))
    ten2 = tensor.tensor(np.arange(10))
    ten = base.tensor_sub(ten1,ten2)
    print(ten.data)
    ten = base.tensor_add(ten1, ten2)
    print(ten.data)

def TestNorm():
    ten1 = tensor.tensor(np.arange(3))
    print(ten1.data)
    print(base.tennorm(ten1)**2)

def TestUnfoldAndFold():
    ten1 = tensor.tensor(np.arange(24).reshape(3,4,2))
    U = [base.unfold(ten1, i) for i in range(3)]
    print([U[i].shape for i in range(3)])
    ten = [base.fold(U[i],ten1.shape,i) for i in range(3)]
    print([ten[i].shape for i in range(3)])

def Test


if __name__ == '__main__':

    TestNorm() # F Norm
    TestSubAdd() #Add and Sub
    TestUnfoldAndFold() #Test Unfold And Fold

