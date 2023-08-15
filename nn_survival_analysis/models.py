from general_utils import *

class MLP(nn.Module):
    '''
    simple model class for time-invariant
    '''
    def __init__(self , input_size , hidden_size , output_size , dropout_rate = 0.1):
        super(MLP , self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.BatchNorm1d(hidden_size), nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.BatchNorm1d(hidden_size), nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self , x):
        # forward pass
        output = self.network(torch.Tensor(x).to(torch.float32))
        return output

class Net(nn.Module):
    '''
    simple model class for time-varying, just a fc net slapped onto a conv net 
    '''
    def __init__(self , fc_input_size, hidden_size , output_size , dropout_rate):
        super(Net , self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size = 3 , stride = 1 , padding = 1) , nn.ReLU() , 
            nn.Conv2d(8, 4, kernel_size = 3 , stride = 1 , padding = 1) , nn.ReLU() ,
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size), nn.ReLU(), nn.BatchNorm1d(hidden_size), nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.BatchNorm1d(hidden_size), nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self , x):
        # forward pass
        output = self.fc(self.conv_net(torch.Tensor(x).to(torch.float32)))
        return output