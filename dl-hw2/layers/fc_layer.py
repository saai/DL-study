""" 全连接层 """

import numpy as np

class FCLayer():
	def __init__(self, num_input, num_output, actFunction='relu', trainable=True):
		"""
		对输入进行线性变换: y = Wx + b
		参数简介:
			num_input: 输入大小
			num_output: 输出大小
			actFunction: 激活函数类型(无需修改)
			trainable: 是否具有可训练的参数
		"""
		self.num_input = num_input
		self.num_output = num_output
		self.trainable = trainable
		self.actFunction = actFunction
		assert actFunction in ['relu', 'sigmoid']

		self.XavierInit()

		self.grad_W = np.zeros((num_input, num_output)) # shape(N_l-1, N_l) 
		self.grad_b = np.zeros((1, num_output)) # shape(1, N_l) 


	def forward(self, Input):
		############################################################################
	    # TODO: 
		# 对输入计算Wx+b并返回结果.
	    ############################################################################
		W_T = self.W.transpose() # shape(N_l, N_l-1) 
		b_T = self.b.transpose() # shape (N_l, 1)
		# print('W_T shape', W_T.shape)
		# print('b_T shape', b_T.shape)
		self.Input = Input  # shape(batch_size, N_l-1)
		# print('Input shape', Input.shape)
		Input_T = Input.transpose() # shape(N_l-1, batch_size)
		a = np.dot(W_T,Input_T) # shape 
		out_M = a+b_T # shape (batchsize, N_l)
		out_M = out_M.transpose()
		return out_M 
		
	def backward(self, delta):
		# 输入的delta由下一层计算得到
		############################################################################
	    # TODO: 
		# 根据delta计算梯度
	    ############################################################################
		# delta shape(batch_size, N_l)
		# 更新 grad_W, grad_b
		N = delta.shape[0] # 最后一层时是batch_size
		N_l = delta.shape[1] # 下一层的维度 N_l, 也就是当前层的输出维度
		N_pre = self.Input.shape[1] # N_l-1, 也就是当前层的输入维度
		delta_T = delta.transpose() # shape (N_l, batch_size)
		# 每一列与Input的一行相乘，形成一个矩阵
		for n in range(N):
			# 当前层局部敏感度，由下一层计算的来的
			delta_n = delta[n].reshape(N_l, 1) # shape(N_l, 1)
			# 前一层的输出，即当前层的输入的一个元素
			y_pre = self.Input[n].reshape(1, N_pre) # shape(1, N_l-1)
			self.grad_W += np.dot(delta_n, y_pre).transpose() #  shape (N_l-1, N_l)
			self.grad_b += delta_n.transpose() # shape(1, N_l)
		self.grad_W = self.grad_W/N # shape (N_l-1, N_l)
		self.grad_b = self.grad_b/N # shape(1, N_l)	
		# 带入当前层输出 y 
		delta_back = np.dot(self.W, delta_T)  # 返回给 l-1 层的delta， 最终shape (N_l-1, batch_size )
		delta_back = delta_back.transpose() # shape (batch_size, N_l-1)
		
		# 返回前一层需要的delta
		return delta_back


	def XavierInit(self):
		# 初始化，无需了解.
		raw_std = (2 / (self.num_input + self.num_output))**0.5
		if 'relu' == self.actFunction:
			init_std = raw_std * (2**0.5)
		elif 'sigmoid' == self.actFunction:
			init_std = raw_std
		else:
			init_std = raw_std # * 4

		self.W = np.random.normal(0, init_std, (self.num_input, self.num_output)) # shape(N_l-1, N_l) 
		self.b = np.random.normal(0, init_std, (1, self.num_output)) # shape (1, N_l)
