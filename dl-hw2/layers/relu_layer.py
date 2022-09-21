""" ReLU激活层 """

import numpy as np

class ReLULayer():
	def __init__(self):
		"""
		ReLU激活函数: relu(x) = max(x, 0)
		"""
		self.trainable = False # 没有可训练的参数
	
	@staticmethod
	def relu(x):
		return max(x, 0)

	def d_relu(x):
		return 0 if x<0 else 1

	def forward(self, Input):

		############################################################################
	    # TODO: 
		# 对输入应用ReLU激活函数并返回结果
	    ############################################################################
		self.Input = Input
		f = ReLULayer.relu
		vfunc = np.vectorize(f)
		output_M = vfunc(Input)
		return output_M
	

	def backward(self, delta):

		############################################################################
	    # TODO: 
		# 根据delta计算梯度
	    ############################################################################
		# delta , shape(batchsize, N_l), 因为是激活层， N_l = N_l-1
		d_f = ReLULayer.d_relu
		vfunc = np.vectorize(d_f)
		df_v = vfunc(self.Input) # shape(batchsize, N_l-1)
		delta_back = np.multiply(delta, df_v)
		return delta_back