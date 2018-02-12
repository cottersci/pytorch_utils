'''
    Helper functions to be used in unit testing with pytest
'''
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

def is_learning(model,batch):
    '''
        Asserts that all learnable paramters in a model update during
        optimizer step.

        :param model (torch.nn.Module): Model to test
        :param batch (torch.Tensor): Batch to train the model on
    '''
    # A large value is needed here to be sure all paramters significantly
    # change after only 1 step
    optimizer = optim.SGD(model.parameters(),lr=1e4)

    #Collect the paramaters to train and their names
    original_learnables = []
    paramater_names = []
    for i in model.named_parameters():
        original_learnables.append(i[1].clone())
        paramater_names.append(i[0])

    ## Train Model
    res = model(Variable(batch))
    loss = torch.pow(res,2).mean()
    loss.backward()
    optimizer.step()

    ## Get paramters after training
    modified_learnables = list(model.parameters())

    ## Test that paramters have changed
    for i in range(len(original_learnables)):
        assert not (original_learnables[i] == modified_learnables[i]).data.any(), "Parameter " + paramater_names[i] + " not learning."
