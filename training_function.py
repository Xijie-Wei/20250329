
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset

def training_loop(network,dataloader,optimizer,weight1,weight2):
    """
    This is the function of trainging a classification network.
    This function requires cuda

    Input:
        newotk: Training network
        dataloader: dataloader contains training data expect two output
            [1,0] means track is reconstructed well
            [0,1] means track is not reconstructed well
        optimizer: the optimizer used in this trainging 
        weight1: weight of output[0] in CrossEntrypyLoss
        weight2: weight of output[1] in CrossEntrypyLoss

    Output:
        nework: Trained network
        total_loss: average loss in this training loop
    """

    criterion = nn.CrossEntropyLoss(torch.tensor([weight1,weight2]).to('cuda'))
    total_loss = 0

    # Train with good set
    for i,datas in enumerate(dataloader,0):
        # Load data list
        inputs,labels = datas
        inputs = torch.tensor(inputs).to('cuda')
        labels = torch.tensor(labels).float().to('cuda')
        # Zero the gradient
        optimizer.zero_grad()

        outputs = network(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        total_loss+= loss.item()

    total_loss = total_loss / (i+1)

    return network,total_loss

def test_score (network,test_set,index = 3000):
    """
    This is a function calculate score of the network, used to help change weight in training.

    Input:
        network: test network
        testset: testset(input only) used to test the model with 
            [0,index] good track(expect output[0]>output[1]) and 
            [index,testset.shape[0]] bad track (expect output[0]<output[1])
        index: index used in defination of testset

    Output:
        accuracy_good: accuracy of good dataset
        accuracy_bad: accuracy of bad dataset
    """
    accuracy = 0
    accuracy_good = 0
    accuracy_bad = 0
    for i in range (0,index):
        inputs = test_set[i].to("cuda")
        output = network(inputs)
        if output[0]>output[1]:
            accuracy +=1
            accuracy_good+=1
    for i in range (index,test_set.shape[0]):
        inputs = test_set[i].to("cuda")
        output = network(inputs)
        if output[0]<output[1]:
            accuracy +=1
            accuracy_bad +=1
    accuracy = 100*accuracy / test_set.shape[0]
    accuracy_good = 100*accuracy_good/ index
    accuracy_bad = 100*accuracy_bad/ (test_set.shape[0] - index)
    print(f"The accuracy of the network for well reconstructed tracks is: {accuracy_good:.3f}%")
    print(f"The accuracy of the network for poor reconstructed tracks is: {accuracy_bad:.3f}%")
    print(f"The accuracy of the network is: {accuracy:.3f}%")
    return accuracy_good,accuracy_bad