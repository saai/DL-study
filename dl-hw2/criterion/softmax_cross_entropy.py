""" Softmax交叉熵损失层 """

import numpy as np

# 为了防止分母为零，必要时可在分母加上一个极小项EPS
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')

	def forward(self, logit, gt):
		"""
	      输入: (minibatch)
	      - logit: 最后一个全连接层的输出结果, 尺寸(batch_size, 10)
	      - gt: 真实标签, 尺寸(batch_size, 10)
	    """

		############################################################################
	    # TODO: 
		# 在minibatch内计算平均准确率和损失，分别保存在self.accu和self.loss里(将在solver.py里自动使用)
		# 只需要返回self.loss
	    ############################################################################
		self.logit = logit
		self.gt = gt
		batch_size = logit.shape[0]
		# 以下这段代码在层次很深的时候会出现问题，还是用分母加一个 EPS 值的方式来优化softmax 计算中分母为0的情况
		# logit_max_M = np.max(logit, axis=1).reshape(batch_size,1) # 求logit每行最大的值
		# print('logit_max_M shape', logit_max_M.shape)
		# # 求softmax矩阵， 用避免下溢的方法，原矩阵每个都减去此行最大值
		# logit_1 = np.subtract(self.logit, logit_max_M)  #logit修正值, 根据广播机制，会在logit每行的值都减去该行的最大值logit_max。
		# print('logit_1 shape', logit_1.shape)
		# logit_exp = np.exp(logit_1) # exp 值
		# logit_denom = np.sum(logit_exp, axis=1).reshape(batch_size, 1) # 每行和, 列对齐
		# print ('logit denom shape', logit_denom.shape)
		logit_exp = np.exp(logit)
		logit_denom = np.sum(logit_exp, axis = 1).reshape(batch_size, 1)
		softmax_M = np.divide(logit_exp, logit_denom+EPS) # logit_denom 自动广播，使每一个exp后的值都除以该行exp和值
		self.softmax_M = softmax_M
		loss_M = - np.sum(np.multiply(np.log(softmax_M),gt), axis = 1)  # 每一行是一个样本的loss，shape(batch_size, 1)
		self.loss = np.mean(loss_M) # shape(1)
		pred = np.argmax(softmax_M, axis=1) #预测最大概率的类, shape(batch_size,)
		y_M = np.zeros(softmax_M.shape, dtype=int) # shape(batch_size,10)
		for i in range(0, batch_size):
			y_M[i][pred[i]] = 1 # 将第i个元素(行)的k类（列）设置为1
		self.acc = 1.0 * np.sum(np.multiply(y_M, gt))/(batch_size) #判断对了类标的个数占总比例
		return self.loss

	def backward(self):

		############################################################################
	    # TODO: 
		# 计算并返回梯度(与logit具有同样的尺寸)
	    ############################################################################
		delta = self.softmax_M - self.gt # 返回的是 l-1 层需要的delta
		return delta
