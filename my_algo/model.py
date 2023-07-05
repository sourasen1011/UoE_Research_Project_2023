import torch.nn as nn

class MLP(nn.Module):
    def __init__(self , input_size , hidden_size , output_size , dropout_rate):
        super(MLP , self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.BatchNorm1d(hidden_size), nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.BatchNorm1d(hidden_size), nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)
        )
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Flatten(),
        #     nn.Linear(32 * 7 * 7, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 10),
        #     nn.LogSoftmax(dim=1)
        # )
    
    def forward(self , x):
        # forward pass
        output = self.network(x)
        return output
    

# self.l1 = nn.Linear(input_size , hidden_size)
# self.l2 = nn.Linear(hidden_size , hidden_size)
# self.l3 = nn.Linear(hidden_size , output_size)
# self.relu = nn.ReLU()
# self.bn = nn.BatchNorm1d(hidden_size)
# self.dropout = nn.Dropout(dropout_rate)
# stack layers
# out = self.l1(x)
# out = self.relu(out)
# out = self.bn(out)
# out = self.dropout(out)
# out = self.l2(out)
# out = self.relu(out)
# out = self.bn(out)
# out = self.dropout(out)
# out = self.l3(out)
# return out