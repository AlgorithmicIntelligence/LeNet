import numpy as np
import cv2
import Layer

def loadMNIST():
    int_type = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * int_type.itemsize

    data = np.fromfile('MNIST/train-images-idx3-ubyte', dtype='ubyte')
    magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), int_type)

    train_data = data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width, height])
    train_labels = np.fromfile('MNIST/train-labels-idx1-ubyte', dtype='ubyte')[2 * int_type.itemsize:]

    data = np.fromfile('MNIST/t10k-images-idx3-ubyte', dtype='ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), int_type)

    test_data = data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width, height])
    test_labels = np.fromfile('MNIST/t10k-labels-idx1-ubyte', dtype='ubyte')[2 * int_type.itemsize:]

    return train_data, train_labels, test_data, test_labels

#data preprocessing
train_data, train_labels, test_data, test_labels = loadMNIST()

train_data  = np.pad(train_data, ((0, ), (2, ), (2, ) ), "constant")
test_data  = np.pad(test_data, ((0, ), (2, ), (2, ) ), "constant")

normalize_min, normalize_max = -0.1, 1.175
train_data = train_data/255*(normalize_max-normalize_min) + normalize_min
test_data = test_data/255*(normalize_max-normalize_min) + normalize_min


