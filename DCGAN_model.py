import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self,latent_dim, ):
        super(Generator, self).__init__()

        # 2 linear layers
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

        
    def forward(self, z):
        x = F.leaky_relu(self.fc1(z))
        x = F.leaky_relu(self.fc2(x))
        #tanh
        x = (self.fc3(x))
        return x


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
    
    def forward(self,x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        
        return x
        

class Classifier(nn.Module):
    def __init__(self, base_model, classfier):
        super(Classifier, self).__init__()
        self.base_model = base_model
        self.fc4 = nn.Linear(32, classfier)
    
    def forward(self,x):
        x = F.leaky_relu(self.base_model(x))
        x = F.softmax(self.fc4(x))
        return x
        

class Discriminator(nn.Module):
    def __init__(self, base_model):
        super(Discriminator, self).__init__()
        self.base_model = base_model
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self,x):
        x = F.leaky_relu(self.base_model(x))
        real = F.sigmoid(self.fc4(x))
        return real
        
