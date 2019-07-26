import Layer

class Lenet5():
    def __init__(self, inputs):
        self.C1 = Layer.ConvolutionalLayer(inputs,(5, 5), 6, "VALID")
        self.S2 = Layer.PoolingLayer(sel)
