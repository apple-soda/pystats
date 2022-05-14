import numpy as np
import math

class Geometric:
    '''
    Models n draws until observing first success, where n is not fixed
    '''
    def __init__(self, p_success, p_fail):
        assert p_success + p_fail == 1
        self.p_success = p_success
        self.p_fail = p_fail
        
    def E(self, distribution):
        assert len(distribution) != 0
        # E(X) = 1/p
        E = 1 / self.p_success
        
    def Var(self, distribution):
        assert len(distribution) != 0
        # Var(X) = (1-p)/p^2
        Var = (1 - self.p_success) / (self.p_success ** 2)
        return Var
    
    def SD(self, distribution):
        assert len(distribution) != 0
        SD = self.Var(distribution) ** 0.5
        return SD
    
    def PMF(self, distribution):
        assert len(distribution) != 0
        # P(X) = (1-p)^(n-1) * p
        n = len(distribution)
        p = (self.p_fail ** (n-1)) * self.p_success
        
        return p