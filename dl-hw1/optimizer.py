import numpy as np

class SGD(object):
    def __init__(self, model, learning_rate, momentum=0.0):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.last_v_W = np.zeros([self.model.num_input, self.model.num_output], dtype=int) # shape(m, K)
        self.last_v_b = np.zeros([1, self.model.num_output], dtype=int) # shape(1, K)

    def step(self):
        """One updating step, update weights"""
        ############################################################################
        # TODO: Put your code here
        # Calculate diff_W and diff_b using layer.grad_W and layer.grad_b.
        # You need to add momentum to this.

        # Weight update with momentum
        

        # # Weight update without momentum
        # layer.W += -self.learning_rate * layer.grad_W
        # layer.b += -self.learning_rate * layer.grad_b

        ############################################################################
        layer = self.model
        if layer.trainable:
            # Weight update with momentum
            v_W = self.momentum * self.last_v_W + self.learning_rate * layer.grad_W
            v_b = self.momentum * self.last_v_b + self.learning_rate * layer.grad_b 
            layer.W += - v_W 
            layer.b += - v_b
            self.last_v_W = v_W # 更新
            self.last_v_b = v_b # 更新
            # # Weight update without momentum
            # layer.W += -self.learning_rate * layer.grad_W
            # layer.b += -self.learning_rate * layer.grad_b