""" 欧式距离损失层 """

import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = 0.

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
		sub_M = np.subtract(logit,gt)
		self.sub_M = sub_M # 损失层的delta
		row_sum_M = np.sum(np.sqrt(np.multiply(sub_M, sub_M)), axis = 1),  # sum of rows
		self.loss = 0.5 * np.mean(row_sum_M) # euclidean loss
		pred = np.argmax(logit, axis=1) #预测最大概率的类, shape(batch_size,)
		batch_size = logit.shape[0]
		y_M = np.zeros(logit.shape, dtype=int) # shape(batch_size,10)
		for i in range(0, batch_size):
			y_M[i][pred[i]] = 1 # 将第i个元素(行)的k类（列）设置为1
		self.acc = 1.0 * np.sum(np.multiply(y_M, gt))/(batch_size) #判断对了类标的个数占总比例		
		return self.loss

	def backward(self):

		############################################################################
	    # TODO: 
		# 计算并返回梯度(与logit具有同样的尺寸)
	    ############################################################################
		# 返回的是当前层的 delta
		return self.sub_M
