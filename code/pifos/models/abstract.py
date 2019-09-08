import torch.nn as nn
import torch.nn.functional as F


class VarModel(nn.Module):
    def __init__(self):
        super(VarModel, self).__init__()
        self.theta = {'qw':[]} # variational params
        self.h = {} # hyperparams


class StructureVarModel(VarModel):
    def __init__(self):
        super(StructureVarModel, self).__init__()
        self.theta['qG'] = []
    
    
    
    


class MultiStartElboModel(VarModel):
    def __init__(self, multistart_models):
        super(MultiStartElboModel, self).__init__()
        self.multistart_models =multistart_models
        
    def forward(self, x):
        return [m(x) for m in self.multistart_models]
        




