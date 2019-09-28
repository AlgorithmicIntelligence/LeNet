import numpy as np
import pickle
import Lenet5
import MNIST
import time
import os

# data pre-processing
train_data, train_labels, test_data, test_labels = MNIST.load()
train_data = np.pad(train_data, ((0, ), (2, ), (2, )), "constant")
test_data = np.pad(test_data, ((0, ), (2, ), (2, )), "constant")

normalize_min, normalize_max = -0.1, 1.175
train_data = train_data/255*(normalize_max-normalize_min) + normalize_min
test_data = test_data/255*(normalize_max-normalize_min) + normalize_min

train_data = np.expand_dims(train_data, axis=-1)
test_data = np.expand_dims(test_data, axis=-1)

epochs = 20
batch_size_SDLM = 500
batch_size = 32
learning_rate = np.array([0.0005]*2 + [0.0002]*3 + [0.0001]*3 + [0.00005]*4 + [0.00001]*8)

restore_weights_path = "./Weights/LeNetWeights_6000.pkl"
if os.path.isfile(restore_weights_path):
    with open(restore_weights_path, "rb") as f:
        lenet5 = pickle.load(f)
else:
    lenet5 = Lenet5.Lenet5()

train_loss_list = list()
test_loss_list = list()
train_data_index = np.arange(len(train_labels))

for epoch in range(epochs):
    print("=================================================")
    print("epoch \t", epoch, "\n")
    np.random.shuffle(train_data_index)

    time_start = time.time()

    lenet5.forward_propagation(train_data[train_data_index[:batch_size_SDLM]], train_labels[train_data_index[:batch_size_SDLM]])
    lenet5.SDLM(learning_rate[epoch])
    print("learning rates in trainable layers:", np.array([lenet5.C1.lr, lenet5.S2.lr, lenet5.C3.lr, lenet5.S4.lr, lenet5.C5.lr, lenet5.F6.lr]))
    
    for i in range(len(train_labels)//batch_size):
        loss, labels_ = lenet5.forward_propagation(train_data[train_data_index[i*batch_size:(i+1)*batch_size]], train_labels[train_data_index[i*batch_size:(i+1)*batch_size]])    
        lenet5.backward_propagation()
        
        time_end = time.time()
        if ((i+1) % 5) == 0:
            print("Training Data Num ", (i+1)*batch_size)
            print("labels(GT) = ", train_labels[train_data_index[i*batch_size:(i+1)*batch_size]])
            print("labels(PD) = ", labels_)
            print("Loss = ", loss, "\tTime Elapsed = ", time_end-time_start,"\n")
            if not os.path.isdir("./Weights/"):
                os.mkdir("./Weights/")
            with open("./Weights/LeNetWeights_"+str((i+1)*batch_size)+".pkl", "wb") as f:
                pickle.dump(lenet5, f)
    
    print("training time = ", time_end - time_start, "\n")
    
    loss = 0
    for i in range(len(train_labels)):
        loss_, _ = lenet5.forward_propagation(train_data[i:i+1], train_labels[i:i+1])
        loss += loss_
    loss /= len(train_data_index)
    train_loss_list.append(loss)
    print("training loss = ", loss, "\n")
    
    loss = 0
    for i in range(len(test_labels)):
        loss_, _ = lenet5.forward_propagation(test_data[i:i+1], test_labels[i:i+1])
        loss += loss_
    loss /= len(test_labels)  
    test_loss_list.append(loss)
    print("testing loss = ", loss, "\n")

with open("./Weights/LeNetWeightsFinal.pkl", "wb") as f:
    pickle.dump(lenet5, f)