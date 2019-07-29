import numpy as np
import cv2
import Layer
import MNIST

#data preprocessing
train_data, train_labels, test_data, test_labels = MNIST.load()

train_data  = np.pad(train_data, ((0, ), (2, ), (2, ) ), "constant")
test_data  = np.pad(test_data, ((0, ), (2, ), (2, ) ), "constant")

normalize_min, normalize_max = -0.1, 1.175
train_data = train_data/255*(normalize_max-normalize_min) + normalize_min
test_data = test_data/255*(normalize_max-normalize_min) + normalize_min


