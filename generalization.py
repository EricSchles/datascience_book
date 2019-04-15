import numpy as np 
from scipy import stats 
from statsmodels.base.model import GenericLikelihoodModel 



class AlmostMinimalScipyMLE(GenericLikelihoodModel): 

    def __init__(self, endog, distr, nparams): 
        self.distr = distr 
        self.nparams = nparams 
        super(AlmostMinimalScipyMLE, self).__init__(endog, None) 
        self.df_model = nparams 
        self.df_resid = self.endog.shape[0] - nparams 


    def loglike(self, params): 
        return self.distr.logpdf(self.endog, *params).sum() 

nobs = 100 
mu, sig = 2, 1.5 
x = mu + sig * np.random.randn(nobs) 

mod = AlmostMinimalScipyMLE(x, stats.norm, 2) 
res = mod.fit() 
print(res.summary()) 

mod = AlmostMinimalScipyMLE(x, stats.t, 3) 
res = mod.fit() 
print(res.summary()) 

mod = AlmostMinimalScipyMLE(x, stats.binom, 1)
res = mod.fit()
print(res.summary())
