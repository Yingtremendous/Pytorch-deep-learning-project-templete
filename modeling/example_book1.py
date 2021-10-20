import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module): # SimpleNet inherits from nn.Module class

    def __init__(self): # typically create layers as a class attributes
        super(SimpleNet, self).__init__() # call the base's __init__() function to initialize parameters
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64,2)
        # super(): call super() function to execute the parent nn.Module class's __init__（） 
        # function to initialize the class parameters

    def forward(self, x):
        x = x.view(-1, 2048) # reshape the inpute (into a element vector)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x




simplenet = SimpleNet()
print(simplenet)


input = torch.rand(2048)
output = simplenet(input)