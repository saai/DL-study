from cProfile import label
import numpy as np
# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLoss(object):

    def __init__(self, num_input, num_output, trainable=True):
        """
        Apply a linear transformation to the incoming data: y = Wx + b
        Args:
            num_input: size of each input sample
            num_output: size of each output sample
            trainable: whether if this layer is trainable
        """

        self.num_input = num_input
        self.num_output = num_output
        self.trainable = trainable
        self.XavierInit()

    def forward(self, Input, labels):
        """
          Inputs: (minibatch)
          - Input: (batch_size, 784)
          - labels: the ground truth label, shape (batch_size, )
        """
        N = Input.shape[0] # batch_size
        self.N = N
        self.Input = Input
        ############################################################################
        # TODO: Put your code here
        # Apply linear transformation (WX+b) to Input, and then
        # calculate the average accuracy and loss over the minibatch
        # Return the loss and acc, which will be used in solver.py
        # Hint: Maybe you need to save some arrays for gradient computing.

        ############################################################################
        # print('Input shape', Input.shape)
        # print('labels shape', labels.shape)
        # print('smaple label value', labels[:10])
        K = 10 # 只有10个类
        X = Input.transpose() # shape(m, N)
        W_T = self.W.transpose() # shape(K,m)
        b_T = self.b.transpose() # shape(K,1)
        prob_M = np.dot(W_T,X) + b_T # shape(K,N)
        self.prob_M = prob_M
        # print('prob_M shape:', prob_M.shape)
        # tM =  # 第labels-1 的index 为非0， 其他都是0，shape (K,N)
        self.t_M = np.zeros([K, N], dtype=int) # shape(K,N)
        for i in range(0, N):
            self.t_M[labels[i]][i] = 1 # 将第i个元素(列)的k类（行）设置为1
        nom_M =  np.sum(np.multiply(np.exp(prob_M),self.t_M),axis=0) # 分子的exp 矩阵, shape(1,N)
        denom_M = np.sum(np.exp(prob_M), axis=0) # 以列相加 shape(1,N)
        # print('denom_M shape:', denom_M.shape)
        pred = np.argmax(prob_M, axis=0) #预测最大概率的类
        y_M = np.zeros([K, N], dtype=int) # shape(K,N)
        for i in range(0, N):
            y_M[pred[i]][i] = 1 # 将第i个元素(列)的k类（行）设置为1
        # 计算loss每一列非0行的prob = log(exp（每一列非0行值prob_M）/exp（每一列求和)）
        # 得到N个值，求平均，取负，就是loss
        loss = -1* np.mean(np.log(np.divide(nom_M, denom_M)))
        acc = 1.0 * np.sum(np.multiply(y_M, self.t_M))/N #判断对了类标的个数占总比例
        # print("loss:", loss, "acc:", acc)
        return loss, acc

    def gradient_computing(self):
        ############################################################################
        # TODO: Put your code here
        # Calculate the gradient of W and b.
        ############################################################################
        prob_diff_M = self.prob_M - self.t_M # shape(K, N)
        grad_W = np.zeros([self.num_output, self.num_input]) # shape(K, m)
        grad_b = np.zeros([self.num_output, 1]) # shape(K, 1)
        for n in range(self.N):
            f_n = prob_diff_M[:,n].reshape(self.num_output,1) # shape(K,1)
            t_n = self.Input[n].reshape(1, self.num_input) #  shape(1,m)
            grad_W += np.dot(f_n, t_n)#  shape(K, 1) dot* shape(1,m) = shape(K, m)
            grad_b += prob_diff_M[:,n].reshape(self.num_output, 1) # shape (K, 1)
        grad_W = grad_W/self.N # shape(K, m)
        grad_b = grad_b/self.N # shape(K, 1)
        self.grad_W = grad_W.transpose() # shape(m, K)
        self.grad_b = grad_b.transpose() # shape(1, K)

    def XavierInit(self):
        """
        Initialize the weigths
        """
        raw_std = (2 / (self.num_input + self.num_output))**0.5
        init_std = raw_std * (2**0.5)
        self.W = np.random.normal(0, init_std, (self.num_input, self.num_output)) # shape(m, K)
        self.b = np.random.normal(0, init_std, (1, self.num_output)) # shape(1, K)

