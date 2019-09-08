import numpy as np
import torch.nn as nn
import torch as t

LOG_SQRT_2PI = np.log(np.sqrt(2*np.pi))
def neg_log_ll_softmax(X, Y):
    pass 

def neg_log_ll_regression( y, predicted, log_eps=0):
    """
    Distribution = const * (y-predicted)^2/2eps^2
    """
    eps_2 = t.exp(2*log_eps)    
    nll =  t.sum((predicted-y)**2)/2/eps_2 + y.shape[0]*(log_eps + LOG_SQRT_2PI)
    return nll

def simple_neg_log_ll_regression( y, predicted, log_eps=0):
    """
    Distribution = const * (y-predicted)^2/2eps^2
    """
    
    nll =  t.sum((predicted-y)**2)
    return nll



def kld_diag(m1, m2, log_s1, log_s2, weights=1.0):
    """
    See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    """
    part1 = t.sum(weights*t.exp(log_s1).view(-1, 1)/t.exp(log_s2).view(-1, 1))
    part2 = t.dot(weights*(m2-m1).view(m1.shape[0])*t.exp(-log_s2).view(m1.shape[0]),(m2-m1).view(m1.shape[0]))
    part3 = -m1.shape[0]
    part4 = t.sum(weights*log_s2)-t.sum(weights*log_s1)
    return 0.5*(part1+part2+part3+part4)



def kld_gs(qG, priorG, sample_num):
    """
    Simple MC-like estimation
    """
    result = 0
    for _ in xrange(sample_num):
        sample = qG.rsample()
        result+= qG.log_prob(sample) - priorG.log_prob(sample)
    return result




