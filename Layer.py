import numpy as np

class ConvolutionalLayer():
    def __init__(self, weights_shape, pad="SAME", stride=1, activation_function=None):
        Fi = np.prod(weights_shape[:3])
        self.shape = weights_shape
        self.weights = np.random.uniform(-2.4/Fi, 2.4/Fi, weights_shape)
        self.biases = np.random.uniform(-2.4/Fi, 2.4/Fi, weights_shape[-1]).reshape(1, 1, 1, -1)
        self.activation_function = activation_function
        self.pad = pad
        self.stride = stride

    def forward_propagation(self, inputs):
        self.inputs = inputs
        if self.pad == "SAME":
            inputs = np.pad(inputs, (((self.shape[0]-1)//2, self.shape[0]//2), ((self.shape[1]-1)//2, self.shape[1]//2), (0, 0)), "constant")

        outputs = np.zeros([(inputs.shape[0]-self.shape[0])//self.stride + 1, (inputs.shape[1]-self.shape[1])//self.stride + 1, self.shape[-1]])
        for h in range(outputs.shape[0]):
            for w in range(outputs.shape[1]):
                for f in range(outputs.shape[2]):
                    outputs[h, w, f] = np.tensordot(inputs[h*self.stride:h*self.stride+self.shape[0], w*self.stride:w*self.stride+self.shape[1], :] * self.weights[..., f]) + self.bias[f]

        if self.activation_function == "SIGMOID":
            outputs = 1/(1+np.exp(outputs))
        elif self.activation_function == "SQUASHING":
            outputs = 1.7159 * np.tanh(2/3 * outputs)
            
        return outputs

    def backward_propagation(self, d_outputs, learning_rate):
        if self.activation_function == "SIGMOID":
            d_outputs = (1-d_outputs) * d_outputs
        elif self.activation_function == "SQUASHING":
            d_outputs = 1.7159 * 2/3 / np.power(np.cosh(2/3 * d_outputs), 2)
        
        d_inputs = np.zeros(self.inputs.shape)
        d_weights = np.zeros(self.weights.shape)
        d_biases = np.zeros(self.biases.shape)
        if self.pad == "SAME":
            inputs = np.pad(self.inputs, (((self.shape[0]-1)//2, self.shape[0]//2), ((self.shape[1]-1)//2, self.shape[1]//2),(0, 0)), "constant")
            d_inputs = np.pad(d_inputs, (((self.shape[0]-1)//2, self.shape[0]//2), ((self.shape[1]-1)//2, self.shape[1]//2),(0, 0)), "constant")
        for h in range(d_outputs.shape[0]):
            for w in range(d_outputs.shape[1]):
                d_inputs[h*self.stride:h*self.stride+self.shape[0], w*self.stride:w*self.stride+self.shape[1], :] += d_outputs[h, w, :] * self.weights
                d_weights += d_outputs[h, w, :] * inputs[h*self.stride:h*self.stride+self.shape[0], w*self.stride:w*self.stride+self.shape[1], :]
                d_biases += d_outputs[h, w, :]
                
        if self.pad == "SAME":
            d_inputs = d_inputs[(self.size[0]-1)//2:self.d_inputs.shape[0]-self.size[0]//2+1, (self.size[1]-1)//2:self.d_inputs.shape[1]-self.size[1]//2+1]
        #update
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_inputs

class PoolingLayer():
    def __init__(self, inputs_shape, size=(2,2), stride=2, mode="MAX"):
        Fi = inputs_shape[-1]
        self.weights = np.random.uniform(-2.4 / Fi, 2.4 / Fi, inputs_shape[-1]).reshape(1, 1, inputs_shape[-1])
        self.biases = np.random.uniform(-2.4 / Fi, 2.4 / Fi, (1, 1, inputs_shape[-1]))
        self.size = size
        self.stride = stride
        self.mode = mode
    def forward_propagation(self, inputs):
        self.inputs = inputs
        outputs = np.zeros([(inputs.shape[0]-self.size[0])//self.stride + 1, (inputs.shape[1]-self.size[1])//self.stride + 1, self.inputs.shape[2]])
        for h in range(inputs.shape[0]):
            for w in range(inputs.shape[1]):
                if self.mode == "MAX":
                    outputs[h, w, :] = np.max(inputs[h*self.stride:h*self.stride + self.size[0], w*self.stride:w*self.stride + self.size[1], :], axis=(1, 2))
                elif self.mode == "AVERAGE":
                    outputs[h, w, :] = self.weights * np.average(inputs[h*self.stride:h*self.stride + self.size[0], w*self.stride:w*self.stride + self.size[1], :], axis=(1, 2)) + self.biases
        return outputs
    def backward_propagation(self, d_outputs, learning_rate):
        d_inputs = np.zeros(self.inputs.shape)
        d_weights = np.zeros(self.weights.shape)
        d_biases = np.zeros(self.biases.shape)
        for h in range(d_outputs.shape[0]):
            for w in range(d_outputs.shape[1]):
                w_interval = range(w * self.stride, w*self.stride+self.size[0])
                h_interval = range(h*self.stride, h*self.stride+self.size[1])
                if self.mode == "MAX":
                    weights = self.inputs[h_interval, w_interval, :] == np.max(self.inputs[h_interval, w_interval, :], axis=(0, 1))
                    d_inputs[h_interval, w_interval, :] += np.repeat(np.repeat(d_outputs[h, w, :] * weights, 2, axis=0), 2, axis=1)
                elif self.mode == "AVERAGE":
                    d_inputs[h_interval, w_interval, :] += self.weights * np.repeat(np.repeat(d_outputs[h, w, :], 2, axis=0), 2, axis=1) / self.size[0] / self.size[1]
                    d_weights += d_inputs[h_interval, w_interval, :] * d_outputs[h, w, :] / self.size[0] / self.size[1]
                    d_biases += d_outputs[h, w, :]

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_inputs

class FullyConnectedLayer():
    def __init__(self, inputs, size, activation_function=None):
        self.weights = np.random.uniform(-1, 1, [inputs.shape[-1], size])
        self.biases = np.random.uniform(-1, 1, size)
        self.activation_function = activation_function
    def forward_propagation(self, inputs):
        self.inputs = inputs
        outputs = np.matmul(inputs, self.weights) + self.biases
        if self.activation_function == "SIGMOID":
            outputs = 1/(1+np.exp(outputs))
    def backward_propagation(self, d_outputs, learning_rate):
        if self.activation_function == "SIGMOID":
            d_outputs = (1-d_outputs) * d_outputs
        d_inputs = np.matmul(d_outputs, self.weights.T)
        d_weights = np.matmul(self.inputs.T, d_outputs)
        d_biases = d_outputs

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_inputs

