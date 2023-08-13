from general_utils import *

class MyDataset(Dataset):
    '''
    simple data set class - agnostic for time invaraiant and time varying, 
    however, time varying does have an exact same implementation of its own
    '''
    def __init__(self, features, duration_index , durations, events):
        self.data = features
        self.durations = durations
        self.duration_index = duration_index
        self.events = events
        assert self.data.shape[0] == self.duration_index.shape[0] == self.durations.shape[0] == self.events.shape[0] , 'shapes must match!'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        cov = self.data[index] # covariates
        dur_idx = self.duration_index[index] # duration index
        dur = self.durations[index] # durations
        eve = self.events[index] # events
        return cov , dur_idx , dur , eve

class Surv_Matrix:
    '''
    a survival matrix - helper function for the nll loss
    '''
    def __init__(self , duration_index , events , q_cuts=10):
        self.duration_index = duration_index
        self.events = events
        self.q_cuts = q_cuts
    
    def make_survival_matrix(self):
        '''
        converts durations and index into a matrix denoting time of event
        this is the y_ij matrix as shown in Kvamme's paper
        '''
        self.surv_matrix = torch.eye(self.q_cuts)[self.duration_index]
        self.surv_matrix = self.surv_matrix*self.events.reshape(-1 , 1) # censoring mask
        return self.surv_matrix

class Transforms:
    def __init__(self , durations):
        self.durations = durations
        
    def discrete_transform(self , _cuts):
        '''
        cut at even spaces
        '''
        self.bin_edges = np.linspace(self.durations.min() , self.durations.max() , _cuts) # right-side edges of the bins
        self.duration_index = np.searchsorted(a = self.bin_edges , v = self.durations)
        self.n_duration_index = self.duration_index.max()+1
        # print(f'n_duration_index {self.n_duration_index} , _cuts: {_cuts}')
        assert self.n_duration_index == _cuts , 'must match. we have self.n_duration_index , {self.n_duration_index} != _cuts {_cuts}'
        return self.duration_index