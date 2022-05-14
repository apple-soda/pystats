import numpy as np
import math

class Binomial:
    '''
    Models k successes in n draws with replacement, where n is fixed
    '''
    def __init__(self, p_success, p_fail):
        assert p_success + p_fail == 1
        
        self.p_success = p_success
        self.p_fail = p_fail
        
    def E(self, distribution): 
        # E(X) = np : number of draws * probability of success
        E = len(distribution) * self.p_success
        return E
    
    def Var(self, distribution):
        # Var(X) = npq : number of draws * probability of success * probability of failure
        Var = len(distribution) * self.p_success * self.p_fail
        return Var
    
    def SD(self, distribution):
        SD = self.Var(distribution) ** 0.5
        return SD
    
    def PMF(self, distribution):
        k = sum(distribution == 1)
        n = len(distribution)
        bi_coef = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
        
        p = bi_coef * self.p_success**k * self.p_fail**(n-k)
        return p