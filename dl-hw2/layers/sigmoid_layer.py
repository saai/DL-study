""" Sigmoid Layer """

import numpy as np
import math

class SigmoidLayer():
	def __init__(self):
		"""
		Sigmoid激活函数: f(x) = 1/(1+exp(-x))
		"""
		self.trainable = False
	
	@staticmethod
	def sigmoid(x):
		return 1 / (1 + math.exp(-x))

	def forward(self, Input):

		############################################################################
	    # TODO: 
		# 对输入应用Sigmoid激活函数并返回结果
	    ############################################################################
		# input shape(batch_size, N_l-1)
		self.Input = Input
		f = SigmoidLayer.sigmoid
		vfunc = np.vectorize(f)
		output_M = vfunc(Input)
		return output_M

	def backward(self, delta):

		############################################################################
	    # TODO: 
		# 根据delta计算梯度
	    ############################################################################
		# delta , shape(batchsize, N_l), 因为是激活层， N_l = N_l-1
		f = SigmoidLayer.sigmoid
		vfunc = np.vectorize(f)
		v1 = vfunc(self.Input)
		v2 = np.subtract(1, v1)
		df_v = np.multiply(v1,v2) # shape(batchsize, N_l-1)
		delta_back = np.multiply(delta, df_v)
		return delta_back 

