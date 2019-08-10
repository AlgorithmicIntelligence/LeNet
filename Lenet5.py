import Layer
import RBF_BITMAP

C3_combination = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 0], [5, 0, 1],
                  [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 0], [4, 5, 0, 1], [5, 0, 1, 2], [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5],
                  [0, 1, 2, 3, 4, 5]]


class Lenet5():
    def __init__(self):
        self.C1 = Layer.ConvolutionalLayer([5, 5, 3, 6], pad="VALID")
        self.S2 = Layer.PoolingLayer([2, 2, 6], mode="AVERAGE")
        self.C3 = Layer.ConvolutionalCombinationLayer([5, 5, 16], C3_combination)
        self.S4 = Layer.PoolingLayer([2, 2, 16], mode="AVERAGE")
        self.C5 = Layer.ConvolutionalLayer([5, 5, 16, 120], pad="VALID")
        self.F6 = Layer.FullyConnectedLayer([120, 84])
        self.RBF = Layer.RBFLayer(RBF_BITMAP.rbf_bitmap())

    def forward_propagation(self, inputs):
        c1_output = self.C1.forward_propagation(inputs)
        s2_output = self.S2.forward_propagation(c1_output)
        c3_output = self.C3.forward_propagation(s2_output)
        s4_output = self.S4.forward_propagation(c3_output)
        c5_output = self.C5.forward_propagation(s4_output)
        f6_output = self.F6.forward_propagation(c5_output)
        output = self.RBF.forward_propagation(f6_output)

    def backward_propagation(self, learning_rate):
        d_inputs_rbf = self.RBF.backward_propagation()
        d_inputs_f6 = self.F6.backward_propagation(d_inputs_rbf, learning_rate)
        d_inputs_c5 = self.C5.backward_propagation(d_inputs_f6, learning_rate)
        d_inputs_s4 = self.S4.backward_propagation(d_inputs_c5, learning_rate)
        d_inputs_c3 = self.C3.backward_propagation(d_inputs_s4, learning_rate)
        d_inputs_s2 = self.S2.backward_propagation(d_inputs_c3, learning_rate)
        d_inputs_c1 = self.C1.backward_propagation(d_inputs_s2, learning_rate)

