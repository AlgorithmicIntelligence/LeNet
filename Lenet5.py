import Layer

class Lenet5():
    def __init__(self, inputs):
<<<<<<< HEAD
        self.C1 = Layer.ConvolutionalLayer(inputs,(5, 5), 6, "VALID")
=======
        self.C1 = Layer.ConvolutionalLayer([5, 5, 3, 6], "VALID")
>>>>>>> 830f7a00ebea4ab7cc18561aa6edba3bb5586f88
        self.S2 = Layer.PoolingLayer(self)
