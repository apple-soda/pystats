import numpy as np
import math

class NegativeBinomial:
    '''
    Models k successes until observing r failures, n is not fixed
    Negative Binomial can also be modeled in the opposite manner:
    Models k failures until observing r failures, n is not fixed
    '''
    def __init__(self, p_success, p_fail):
        assert p_success + p_fail == 1
        self.p_success = p_success
        self.p_fail = p_fail
        
    def E(self, distribution, manner='fail'): 
        assert len(distribution) != 0
        # E(X) = r/p where r is the # of successes/fails, p is the probability of success/fail
        if manner == 'success': 
            E = sum(distribution == 1) / self.p_success
        else:
            E = sum(distribution == 0) / self.p_fail
        return E
    
    def Var(self, distribution, manner='fail'):
        assert len(distribution) != 0
        # Var(X) = (r(1-p))/p^2 
        if manner == 'success':
            Var = (sum(distribution == 1) * (1 - self.p_success)) / (self.p_success ** 2)
        else:
            Var = (sum(distribution == 0) * (1 - self.p_fail)) / (self.p_fail ** 2)
        return Var
    
    def SD(self, distribution, manner='fail'):
        assert len(distribution) != 0
        SD = self.Var(distribution, manner) ** 0.5
        return SD
    
    def PMF(self, distribution, manner='fail'):
        assert len(distribution) != 0
        # P(X) = (n-1) choose (r-1) * (1-p)^(n-r) * p^r
        if manner == 'success':
            r = sum(distribution == 1)
        else:
            r = sum(distribution == 0)
            
        n = len(distribution)
        bi_coef = math.factorial(n-1) / (math.factorial(r-1) * math.factorial((n-1) - (r-1)))
        
        if manner == 'success':
            p = bi_coef * (self.p_fail ** (n-r)) * (self.p_success ** r)
        else:
            p = bi_coef * (self.p_success ** (n-r)) * (self.p_fail ** r)
        return p