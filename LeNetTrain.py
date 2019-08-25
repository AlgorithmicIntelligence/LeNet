import numpy as np
import pickle
import Lenet5
import MNIST
import time

#data preprocessing
train_data, train_labels, test_data, test_labels = MNIST.load()

train_data  = np.pad(train_data, ((0, ), (2, ), (2, ) ), "constant")
test_data  = np.pad(test_data, ((0, ), (2, ), (2, ) ), "constant")

normalize_min, normalize_max = -0.1, 1.175
train_data = train_data/255*(normalize_max-normalize_min) + normalize_min
test_data = test_data/255*(normalize_max-normalize_min) + normalize_min

train_data = np.expand_dims(train_data, axis = -1)
test_data = np.expand_dims(test_data, axis = -1)

epochs = 20
learning_rate = np.array([0.0005]*2 + [0.0002]*3 + [0.0001]*3 + [0.00005]*4 + [0.00001]*8) * 100

lenet5 = Lenet5.Lenet5()

train_loss_list = list()
test_loss_list = list()
train_data_index = np.arange(len(train_labels))

for epoch in range(epochs):
    print("=================================================")
    print("epoch \t", epoch, "\n")
    np.random.shuffle(train_data_index)
    

    time_start = time.time()
    for i in range(len(train_labels)):
        loss, labels_ = lenet5.forward_propagation(train_data[train_data_index[i]], train_labels[train_data_index[i]])
        lenet5.backward_propagation(learning_rate[epoch])
        
        time_end = time.time()
        if (i % 10) == 0:
            print("Training Data Num ", i, "\tlabels = ", train_labels[train_data_index[i]], "\tlabels_ = ", labels_, "\tLoss = ",loss, "\tTime Elapsed = ", time_end - time_start, "\n")
            with open("Weights\\LeNetWeights"+str(i)+".pkl","wb") as f:
                pickle.dump(lenet5, f)
    
    print("training time = ", time_end - time_start, "\n")
    
    loss = 0
    for i in range(len(train_labels)):
        loss += lenet5.forward_propagation(train_data[i], train_labels[i])
    loss /= len(train_data_index)
    train_loss_list.append(loss)
    print("training loss = ", loss, "\n")
    
    loss = 0
    for i in range(len(test_labels)):
        loss += lenet5.forward_propagation(test_data[i], test_labels[i])
    loss /= len(test_labels)  
    test_loss_list.append(loss)
    print("testing loss = ", loss, "\n")

