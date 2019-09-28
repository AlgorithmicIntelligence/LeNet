import Layer
import RBF_BITMAP
import numpy as np

C3_combination = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 0], [5, 0, 1],
                  [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 0], [4, 5, 0, 1], [5, 0, 1, 2], [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5],
                  [0, 1, 2, 3, 4, 5]]


class Lenet5():
    def __init__(self):
        self.C1 = Layer.ConvolutionalLayer([5, 5, 1, 6], pad="VALID", activation_function="SQUASHING")
        self.S2 = Layer.PoolingLayer([2, 2, 6], mode="AVERAGE")
        self.C3 = Layer.ConvolutionalCombinationLayer([5, 5, 16], C3_combination, pad="VALID", activation_function="SQUASHING")
        self.S4 = Layer.PoolingLayer([2, 2, 16], mode="AVERAGE")
        self.C5 = Layer.ConvolutionalLayer([5, 5, 16, 120], pad="VALID", activation_function="SQUASHING")
        self.F6 = Layer.FullyConnectedLayer([120, 84], activation_function="SQUASHING")
        self.RBF = Layer.RBFLayer(RBF_BITMAP.rbf_bitmap())

    def forward_propagation(self, inputs, labels):
        c1_output = self.C1.forward_propagation(inputs)
        s2_output = self.S2.forward_propagation(c1_output)
        c3_output = self.C3.forward_propagation(s2_output)
        s4_output = self.S4.forward_propagation(c3_output)
        c5_output = self.C5.forward_propagation(s4_output)
        c5_output = np.squeeze(c5_output, axis=(1, 2))
        f6_output = self.F6.forward_propagation(c5_output)
        loss, outputs = self.RBF.forward_propagation(f6_output, labels)
        return loss, outputs

    def backward_propagation(self):
        d_inputs_rbf = self.RBF.backward_propagation()
        d_inputs_f6 = self.F6.backward_propagation(d_inputs_rbf)
        d_inputs_f6 = d_inputs_f6.reshape([d_inputs_f6.shape[0], 1, 1, d_inputs_f6.shape[-1]])
        d_inputs_c5 = self.C5.backward_propagation(d_inputs_f6)
        d_inputs_s4 = self.S4.backward_propagation(d_inputs_c5)
        d_inputs_c3 = self.C3.backward_propagation(d_inputs_s4)
        d_inputs_s2 = self.S2.backward_propagation(d_inputs_c3)
        self.C1.backward_propagation(d_inputs_s2)
    
    def SDLM(self, learning_rate):
        d2_inputs_rbf = self.RBF.SDLM()
        d2_inputs_f6 = self.F6.SDLM(d2_inputs_rbf, learning_rate)
        d2_inputs_f6 = d2_inputs_f6.reshape([d2_inputs_f6.shape[0], 1 , 1, d2_inputs_f6.shape[-1]])
        d2_inputs_c5 = self.C5.SDLM(d2_inputs_f6, learning_rate)
        d2_inputs_s4 = self.S4.SDLM(d2_inputs_c5, learning_rate)
        d2_inputs_c3 = self.C3.SDLM(d2_inputs_s4, learning_rate)
        d2_inputs_s2 = self.S2.SDLM(d2_inputs_c3, learning_rate)
        self.C1.SDLM(d2_inputs_s2, learning_rate)        

