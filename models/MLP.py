import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the model
class MLP(nn.Module):
    def __init__(self, unit1, unit2, unit3, drop_rate, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2048 + 4096, 6144)
        self.bn1 = nn.BatchNorm1d(6144)
        
        self.fc2 = nn.Linear(6144, unit1)
        self.bn2 = nn.BatchNorm1d(unit1)
        
        self.fc3 = nn.Linear(unit1, unit2)
        self.bn3 = nn.BatchNorm1d(unit2)
        
        self.fc4 = nn.Linear(unit2, unit3)
        self.bn4 = nn.BatchNorm1d(unit3)
        self.dropout = nn.Dropout(drop_rate)
        
        self.output = nn.Linear(unit3, num_classes)

    def forward(self, input):
        # x = torch.cat((input_f, input_b), dim=1)
        x = input
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        
        x = F.relu(self.fc4(x))
        x = self.bn4(x)
        x = self.dropout(x)
        
        x = torch.sigmoid(self.output(x))  # Binary multi-label classification
        return x
