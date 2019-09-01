import torch.nn as nn
import torch.nn.functional as F


class AbstractVarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = {'qw':[]} # variational params
        self.h = {} # hyperparams

    def forward(x, theta_values):
        # params are sampled.        
        pass 


class AbstractStructureVarModel(AbstractVarModel):
    def __init__(self):
        super().__init__()
        self.theta['qG'] = {}
    
