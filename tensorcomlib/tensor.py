import numpy as np

#Tensor Class

class tensor(object):

	data = None
	shape = None
	dtype = None

	def __init__(self, data = None, shape = None):

		if shape == None:
			shape = data.shape
		super(tensor, self).__init__()
		self.data = np.array(data)
		self.shape = tuple(shape)
		self.dtype = data.dtype

	def size(self):
		ret = 1
		for i in range(0,len(self.shape)):
			ret = ret * self.shape[i]
		return  ret

	def copy(self):
		return tensor(self.data.copy(),self.shape)

	def dimsize(self, ind):
		return self.shape[ind]

	def ndims(self):
		return len(self.shape)

	def tendtype(self):
		return self.dtype