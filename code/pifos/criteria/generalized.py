def generalized_one_level(Y, Y0, nll = None, kld = None, priorG = None,  logph = None, structures = [], c_ll=0, c_prior=0, c_structures=[], c_logph=0):
    """
    Simple generalization of ELBO
    Y - is a prediction of the model
    Y0 - is a ground truth
    ll - YxY0 -> R,  log likelihood 
    kld - () -> R KL divergence term
    priorG - prior distribution on the structure
    logph - () -> R, hyperprior
    """
    result = 0
    if c_ll!=0:
        result+=c_ll * nll(Y,Y0)
    if c_prior!=0:
        result+=c_prior * kld()
    if len(c_structures)>0:
        sample = priorG.rsample()
        for  l, G in zip(c_sturctures, structures):
            result+=1.0 * l * G.logprob(sample)
    if c_logph !=0:
        raise NotImplementedError()
    return result
        