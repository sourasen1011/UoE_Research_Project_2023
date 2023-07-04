import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numba

from pycox.evaluation.concordance import concordance_td

from losses import nll_logistic_hazard
from model import MLP
from transforms import Transforms

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
import torch # For building the networks 
import torchtuples as tt # Some useful functions

from pycox.datasets import metabric
from pycox.models import LogisticHazard
# from pycox.models import PMF
# from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

# We also set some seeds to make this reproducable.
# Note that on gpu, there is still some randomness.
np.random.seed(1234)
_ = torch.manual_seed(123)


class MyDataset(Dataset):
    '''
    simple data set class
    '''
    def __init__(self, data, durations, events):
        self.data = data
        self.durations = durations
        self.events = events

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        cov = self.data[index] # covariates
        dur = self.durations[index] # durations
        eve = self.events[index] # events
        return cov , dur , eve
   
class Eval:
    '''
    Class that takes in a cohort and their predicted survival times, then outputs evaluation metrics
    
    Time dependent concorance index from
    Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A timedependent discrimination
    index for survival data. Statistics in Medicine 24:3927-3944.
    
    Implementation by 
    Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
    with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
    https://arxiv.org/pdf/1910.06724.pdf
    
    '''
    def __init__(self , durations , events , survival_pred , survival_idx):
        self.durations = durations
        self.events = events
        self.survival_pred = survival_pred
        self.survival_idx = survival_idx
    
    def concordance_time_dependent(self):
        return concordance_td(self.durations, self.events, self.survival_pred, self.survival_idx) # method is applied as adjusted anatolini